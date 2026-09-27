#!/usr/bin/env python3
"""Analysis A: tensor trace of one real edit forward pass (digit 5 + "rotate 90 degrees clockwise", 512x512, 40 steps).

Deck: p.4 (Tensor flow), p.5 (Two preprocessing paths for the reference image), p.6 (Block-causal attention);
figures with scripts/plot_trace.py. Needs one CUDA GPU with >= 40 GB (peak 33 GiB).

Experiment A: tensor trace of one real Qwen-Image 2.1 editing forward pass (MNIST rot90, 512x512).

Mirrors diffusers QwenImage21Pipeline.__call__ (commit e0abab83, pipeline_qwenimage21.py) step by step,
calling the pipeline's own helpers, and records shape / dtype / stats at every stage. Transformer
internals are captured with forward hooks. Afterwards the real `pipe(...)` is run with the same seed
to confirm the mirror matches, and again with use_kv_cache=False for timing.

Usage:
  python scripts/trace_forward.py --model MODEL_DIR --out outputs/trace --check   # imports + processor only
  python scripts/trace_forward.py --model MODEL_DIR --out outputs/trace           # full trace (1 GPU, ~35 GiB)
Writes trace.json, trace_arrays.npz, final_*.png into --out; plot with scripts/plot_trace.py.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("QIE_RES", "512")
from qie_mnist import data as me_data  # noqa: E402
from qie_mnist import evaluate as me_eval  # noqa: E402


class me:  # namespace shim: me.make_pairs / me.evaluate
    make_pairs = staticmethod(me_data.make_pairs)
    evaluate = staticmethod(me_eval.evaluate)


CKPT = None   # set from --model
OUT = None    # set from --out
SEED = 0
RES = 512
STEPS = 40
X0_STEPS = [1, 5, 10, 20, 40]  # 1-indexed steps at which x0_hat is decoded

TRACE = {"stages": {}}
NPZ = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def st(x, name=None):
    """shape / dtype / stats summary of a tensor or array."""
    if isinstance(x, torch.Tensor):
        shape, dtype = list(x.shape), str(x.dtype).replace("torch.", "")
        if x.dtype == torch.bool:
            d = {"shape": shape, "dtype": dtype, "num_true": int(x.sum())}
        else:
            xf = x.detach().float()
            d = {"shape": shape, "dtype": dtype, "mean": float(xf.mean()), "std": float(xf.std()) if xf.numel() > 1 else 0.0,
                 "min": float(xf.min()), "max": float(xf.max())}
    else:
        a = np.asarray(x)
        d = {"shape": list(a.shape), "dtype": str(a.dtype), "mean": float(a.mean()), "std": float(a.std()),
             "min": float(a.min()), "max": float(a.max())}
    if name:
        log(f"  {name}: {d}")
    return d


def stage(key, **kw):
    TRACE["stages"].setdefault(key, {}).update(kw)


def dump():
    with open(OUT / "trace.json", "w") as f:
        json.dump(TRACE, f, ensure_ascii=False, indent=1, default=str)
    np.savez_compressed(OUT / "trace_arrays.npz", **NPZ)
    log(f"dumped trace.json / trace_arrays.npz to {OUT}")


def pick_sample():
    samples = me.make_pairs("test", n_per_task=4, seed=0)
    s = next(x for x in samples if x["task"] == "rot90")
    return s


def segments_of(ids_list, tok):
    """Human-readable token list."""
    return [tok.decode([i]) for i in ids_list]


def check_only():
    import diffusers, transformers
    from diffusers import QwenImage21Pipeline  # noqa: F401
    from transformers import Qwen3VLProcessor
    log(f"diffusers {diffusers.__version__} transformers {transformers.__version__} torch {torch.__version__}")
    log(f"cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    log(f"ckpt listing: {sorted(os.listdir(CKPT))}")
    log(open(os.path.join(CKPT, "model_index.json")).read())
    proc = Qwen3VLProcessor.from_pretrained(os.path.join(CKPT, "processor"))
    s = pick_sample()
    tmpl = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")
    mi = proc(text=[tmpl.format(s["prompt"])], images=[s["ref"]], padding=True, padding_side="left", return_tensors="pt")
    ids = mi.input_ids[0].tolist()
    log(f"sample task={s['task']} label={s['src_label']} prompt={s['prompt']}")
    log(f"input_ids len {len(ids)}; image_pad count {ids.count(151655)}; grid_thw {mi.image_grid_thw.tolist()}; "
        f"pixel_values {list(mi.pixel_values.shape)}")
    log("tokens: " + " | ".join(repr(t) for t in segments_of(ids, proc.tokenizer) if t != "<|image_pad|>"))
    log("CHECK OK")


def main():
    import diffusers, transformers
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, calculate_dimensions, retrieve_timesteps
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache, QwenImage21Transformer2DModel, _IMG_TOKENS_PER_SLOT)
    from PIL import Image

    dev = torch.device("cuda")
    torch.set_grad_enabled(False)  # encode_prompt / prepare_latents are not no_grad-decorated
    TRACE["env"] = {"diffusers": diffusers.__version__, "transformers": transformers.__version__,
                    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "diffusers_commit": "e0abab83b5df05de9e7abd788643c1a7c1e42e28", "hf_revision": "790c9263",
                    "seed": SEED, "res": RES, "steps": STEPS}

    # ------------------------------------------------------------------ sample
    s = pick_sample()
    TRACE["sample"] = {"split": "test", "make_pairs": "make_pairs('test', n_per_task=4, seed=0), first rot90",
                       "task": s["task"], "prompt": s["prompt"], "src_label": s["src_label"], "want_label": s["want_label"]}
    NPZ["ref28"], NPZ["tgt28"] = s["ref28"], s["tgt28"]
    log(f"sample: {TRACE['sample']}")

    # ------------------------------------------------------------------ load
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    load_s = time.time() - t0
    log(f"loaded pipeline in {load_s:.0f}s")
    tr, te, vae = pipe.transformer, pipe.text_encoder, pipe.vae

    def nparams(m):
        return int(sum(p.numel() for p in m.parameters()))
    te_vis = getattr(te.model, "visual", None)
    te_lang = getattr(te.model, "language_model", None)
    TRACE["params"] = {
        "transformer": nparams(tr), "text_encoder": nparams(te),
        "text_encoder_visual": nparams(te_vis) if te_vis is not None else None,
        "text_encoder_language": nparams(te_lang) if te_lang is not None else None,
        "text_encoder_lm_head": nparams(te.lm_head) if hasattr(te, "lm_head") else None,
        "vae": nparams(vae), "vae_encoder": nparams(vae.encoder), "vae_decoder": nparams(vae.decoder),
        "transformer_blocks": len(tr.transformer_blocks),
        "transformer_one_block": nparams(tr.transformer_blocks[0]),
        "dtypes": {"transformer": str(tr.dtype), "text_encoder": str(te.dtype), "vae": str(vae.dtype)},
        "load_seconds": load_s,
    }
    TRACE["attn_processor"] = type(tr.transformer_blocks[0].attn.processor).__name__
    log(f"params: {TRACE['params']}")
    dump()

    # ================================================================== stage 1: image preprocessing
    # pipeline_qwenimage21.py:598-663 (output_resolution=RES so the condition image is RES x RES)
    image = [s["ref"]]
    calc_w, calc_h, _ = calculate_dimensions(RES * RES, image[-1].size[0] / image[-1].size[1])
    height, width = calc_h, calc_w
    multiple_of = pipe.vae_scale_factor * 2
    width, height = width // multiple_of * multiple_of, height // multiple_of * multiple_of
    img = image[0]
    mode_in = img.mode
    img = img.convert("RGBA")                                         # :653-654
    input_w, input_h, _ = calculate_dimensions(RES * RES, img.size[0] / img.size[1])  # :656
    input_images = [pipe.image_processor.resize(img, width=input_w, height=input_h)]  # :660
    vae_images = [pipe.image_processor.preprocess(img, width=input_w, height=input_h).unsqueeze(2)]  # :661-663
    rgba_np = np.asarray(input_images[0])
    NPZ["ref_rgba"] = rgba_np
    stage("1_image", pil_mode_in=mode_in, pil_mode_after_convert="RGBA", pil_size=list(img.size),
          target_hw=[height, width], input_image_size=[input_w, input_h],
          pil_rgba_uint8=st(rgba_np, "PIL RGBA"), alpha_unique=[int(v) for v in np.unique(rgba_np[..., 3])],
          vae_image_tensor=st(vae_images[0], "vae_image [B,C,T,H,W]"),
          vae_image_per_channel_mean=[float(vae_images[0][0, c].mean()) for c in range(4)],
          note="VaeImageProcessor.preprocess: PIL uint8 [0,255] -> float32 [-1,1], 4 channels RGBA; unsqueeze(2) adds a frame axis")

    # ================================================================== stage 2: VAE encode
    # pipeline :423-444 _encode_vae_image
    with torch.no_grad():
        x = vae_images[0].to(dev, torch.bfloat16)
        post = vae.encode(x).latent_dist
        raw_params = post.parameters
        z_raw = post.mode()
        mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1).to(dev, z_raw.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1).to(dev, z_raw.dtype)
        z_norm = (z_raw - mean) / std
        z_pipe = pipe._encode_vae_image(x, None)
        # round trip for the latent figure
        rec = vae.decode(z_norm * std + mean, return_dict=False)[0][:, :, 0]
    NPZ["z_raw"] = z_raw[0, :, 0].float().cpu().numpy()
    NPZ["z_norm"] = z_norm[0, :, 0].float().cpu().numpy()
    NPZ["vae_recon_rgba"] = ((rec[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    stage("2_vae_encode", encoder_output_params=st(raw_params, "posterior params (mean|logvar)"),
          latent_raw=st(z_raw, "z_raw"), latent_norm=st(z_norm, "z_norm"),
          latent_raw_channel_mean_range=[float(z_raw[0, :, 0].float().mean((1, 2)).min()), float(z_raw[0, :, 0].float().mean((1, 2)).max())],
          latent_norm_channel_std=[float(v) for v in z_norm[0, :, 0].float().std((1, 2))],
          matches_pipe_encode_vae_image=float((z_pipe - z_norm).abs().max()),
          vae_roundtrip_rgb_mae_uint8=float(np.abs(NPZ["vae_recon_rgba"][..., :3].astype(float) - rgba_np[..., :3]).mean()),
          vae_roundtrip_alpha=st(NPZ["vae_recon_rgba"][..., 3]),
          sample_mode="argmax (posterior mean, no sampling)")

    # ================================================================== stage 4: prompt encoding (mirror of :234-331)
    prompt = s["prompt"]
    tmpl = pipe.prompt_template_ti2i
    prompt_text = tmpl.format(prompt)
    cond_pil = []
    for im in input_images:
        if im.mode == "RGBA":
            white = Image.new("RGB", im.size, (255, 255, 255))
            white.paste(im, mask=im.getchannel("A"))
            im = white
        cond_pil.append(im)
    mi = pipe.processor(text=[prompt_text], images=cond_pil, padding=True, padding_side="left", return_tensors="pt").to(dev)
    fwd = {"input_ids": mi.input_ids, "attention_mask": mi.attention_mask, "output_hidden_states": True,
           "pixel_values": mi.pixel_values, "image_grid_thw": mi.image_grid_thw}
    if hasattr(mi, "mm_token_type_ids"):
        fwd["mm_token_type_ids"] = mi.mm_token_type_ids
    text_model = getattr(te.model, "language_model", te.model)
    vis_out = {}
    hv = te.model.visual.register_forward_hook(lambda m, a, o: vis_out.setdefault("o", o)) if te_vis is not None else None
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    with torch.no_grad():
        try:
            outputs = te(**fwd)
        finally:
            handle.remove()
            if hv: hv.remove()
        hs_all = outputs.hidden_states
        hs = hs_all[-1]
        hs_normed = text_model.norm(hs)
    ids = mi.input_ids[0].tolist()
    toks = segments_of(ids, pipe.processor.tokenizer)
    drop = pipe._drop_idx
    img_tok = pipe._img_token_id
    kept_ids = ids[drop:]
    kept_toks = toks[drop:]
    ipm = torch.tensor([i == img_tok for i in kept_ids])
    first_pad = kept_ids.index(img_tok)
    n_pad = int(ipm.sum())
    # segment runs over the kept sequence
    runs, start = [], 0
    for k in range(1, len(kept_ids) + 1):
        if k == len(kept_ids) or (kept_ids[k] == img_tok) != (kept_ids[start] == img_tok):
            runs.append({"start": start, "end": k, "kind": "image_pad" if kept_ids[start] == img_tok else "text",
                         "text": "".join(kept_toks[start:k]) if kept_ids[start] != img_tok else f"<|image_pad|> x{k-start}"})
            start = k
    vis_shape = None
    try:
        o = vis_out.get("o")
        cands = list(o) if isinstance(o, (tuple, list)) else [getattr(o, k, None) for k in ("pooler_output", "last_hidden_state")]
        vis_shape = [list(c.shape) for c in cands if isinstance(c, torch.Tensor)]
    except Exception as e:
        vis_shape = f"n/a ({type(e).__name__})"
    prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
        image=input_images, prompt=prompt, device=dev, num_images_per_prompt=1)
    stage("4_prompt", template_literal=tmpl, prompt=prompt, full_text=prompt_text,
          input_ids_len=len(ids), n_image_pad=ids.count(img_tok), image_grid_thw=mi.image_grid_thw.tolist(),
          pixel_values=st(mi.pixel_values, "pixel_values"), vision_tower_output_shape=vis_shape,
          drop_idx=drop, dropped_tokens=toks[:drop], kept_len=len(kept_ids),
          tokens_all=[t if t != "<|image_pad|>" else "P" for t in toks], token_ids_all=ids,
          kept_runs=runs, first_pad_in_kept=first_pad,
          n_hidden_state_entries=len(hs_all), hidden_last_prenorm=st(hs, "hidden_states[-1] (pre-norm)"),
          hidden_last_normed_std=float(hs_normed.float().std()),
          prompt_embeds=st(prompt_embeds, "prompt_embeds"), prompt_embeds_mask=None if prompt_embeds_mask is None else st(prompt_embeds_mask),
          image_pad_mask=st(image_pad_mask, "image_pad_mask"),
          matches_mirror=float((prompt_embeds[0].float() - hs[0, drop:].float()).abs().max()),
          slots_explained=f"VLM patch 16px, merge 2x2 -> 1 slot = 32x32px; VAE token = 16x16px -> 4 latent tokens per slot; "
                          f"{RES}x{RES}: {(RES//32)**2} slots = {(RES//16)**2}/4")
    NPZ["prompt_embeds_rowstd"] = prompt_embeds[0].float().std(-1).cpu().numpy()
    NPZ["prompt_embeds_rownorm"] = prompt_embeds[0].float().norm(dim=-1).cpu().numpy()
    dump()

    # ================================================================== stage 3 + 5: pack + noise, timesteps
    gen = torch.Generator(device=dev).manual_seed(SEED)
    latents, cond_tokens = pipe.prepare_latents(vae_images, 1, 64, height, width, prompt_embeds.dtype, dev, gen, None)
    lh, lw = height // 16, width // 16
    pack_ok = float((cond_tokens[0, 5 * lw + 7] - z_norm[0, :, 0, 5, 7]).abs().max())
    stage("3_flatten", cond_tokens=st(cond_tokens, "cond tokens"), latent_hw=[lh, lw],
          check_token_index=f"token[h*{lw}+w] == latent[:, h, w]; max diff at (5,7) = {pack_ok}")
    stage("5_noise", target_latents=st(latents, "x_T noise tokens"),
          randn_shape=[1, 1, 64, lh, lw], generator=f"torch.Generator(cuda).manual_seed({SEED})")
    img_shapes = [[*[(1, vh // 16, vw // 16) for vw, vh in [(input_w, input_h)]], (1, height // 16, width // 16)]]
    sig_in = np.linspace(1.0, 1 / STEPS, STEPS)
    mu = calculate_shift(latents.shape[1], pipe.scheduler.config.get("base_image_seq_len", 256),
                         pipe.scheduler.config.get("max_image_seq_len", 4096), pipe.scheduler.config.get("base_shift", 0.5),
                         pipe.scheduler.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(pipe.scheduler, STEPS, dev, sigmas=sig_in, mu=mu)
    sig = pipe.scheduler.sigmas.float().cpu().numpy()
    NPZ["sigmas"] = sig
    NPZ["timesteps"] = timesteps.float().cpu().numpy()
    stage("5_noise", mu=float(mu), image_seq_len=int(latents.shape[1]), scheduler_config=dict(pipe.scheduler.config),
          sigmas=[float(v) for v in sig], timesteps=[float(v) for v in timesteps.cpu()],
          sigma_formula="exp(mu)/(exp(mu)+(1/s-1)), then stretch so last=shift_terminal, append 0")

    # ================================================================== stage 6: joint sequence
    def append_target_slots(mask):
        return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)
    img_mask = append_target_slots(image_pad_mask)
    repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
    ipm_joint = torch.repeat_interleave(img_mask[0], repeats)
    image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(ipm_joint, img_shapes[0])
    L = int(ipm_joint.shape[0])
    ii = image_ids.cpu().numpy()
    segs, start = [], 0
    for k in range(1, L + 1):
        if k == L or ii[k] != ii[start]:
            kind = "text" if ii[start] < 0 else ("target" if ii[start] == len(img_shapes[0]) - 1 else f"cond{ii[start]}")
            segs.append({"kind": kind, "start": start, "end": k, "len": k - start})
            start = k
    # text content of each text segment, from the kept VLM tokens
    txt_iter = iter([t for t, m in zip(kept_toks, ipm.tolist()) if not m])
    for sg in segs:
        if sg["kind"] == "text":
            sg["text"] = "".join(next(txt_iter) for _ in range(sg["len"]))
    NPZ["image_ids"] = ii
    stage("6_joint", img_shapes=img_shapes, img_mask=st(img_mask, "img_mask (vlm-level, +target slots)"),
          target_slots_appended=int(latents.shape[1] // 4), joint_len=L, segments=segs,
          prefix_len=int((~target_token_mask).sum()), n_target=int(target_token_mask.sum()))
    log(f"segments: {segs}")

    # ================================================================== stage 7: mask (dense replica of mask_mod, :291-296)
    q = torch.arange(L, device=dev)
    iid = image_ids.to(dev)
    same = (iid[:, None] == iid[None, :]) & (iid[:, None] >= 0)
    M = (q[:, None] >= q[None, :]) | same
    checks = {}
    for sg in segs:
        a, b = sg["start"], sg["end"]
        blk = M[a:b, a:b]
        if sg["kind"] == "text":
            checks[f"{sg['kind']}@{a}"] = {"is_lower_triangular": bool(torch.equal(blk, torch.tril(torch.ones_like(blk)))),
                                           "sees_all_before": bool(M[a:b, :a].all()), "sees_nothing_after": not bool(M[a:b, b:].any())}
        else:
            checks[f"{sg['kind']}@{a}"] = {"fully_bidirectional": bool(blk.all()), "sees_all_before": bool(M[a:b, :a].all()),
                                           "sees_nothing_after": not bool(M[a:b, b:].any())}
    ds = 8
    Lp = (L + ds - 1) // ds * ds
    Mp = torch.zeros(Lp, Lp, device=dev); Mp[:L, :L] = M.float()
    NPZ["mask_ds"] = Mp.view(Lp // ds, ds, Lp // ds, ds).mean((1, 3)).cpu().numpy().astype(np.float16)
    stage("7_mask", rule="allowed = (q_idx >= kv_idx) | (same image block); padding keys removed (none here, batch 1)",
          dense_shape=[L, L], allowed_fraction=float(M.float().mean()), block_checks=checks, downsample=ds,
          processor_path="QwenImage21AttnProcessor: one SDPA call per prefix segment (keys [0,end), causal triangle on text) + one for target (all keys)")
    log(f"mask checks: {checks}")

    # ================================================================== stage 9: RoPE ids (mirror of Rope.forward :682-708)
    def rope_ids(img_shapes_, image_pad_mask_):
        fi, ihi, iwi = [], [], []
        cursor, position = 0, 0
        total = image_pad_mask_.shape[-1]
        is_img = image_pad_mask_.tolist()
        for _, hh, ww in img_shapes_:
            bs = is_img.index(True, cursor)
            tl = bs - cursor
            fi.extend(range(position, position + tl)); position += tl
            cursor = bs + hh * ww
            fi.extend([position] * (hh * ww)); position += max(hh, ww)
            ihi.extend([h for h in range(-(hh - hh // 2), hh // 2) for _ in range(ww)])
            iwi.extend([w for _ in range(hh) for w in range(-(ww - ww // 2), ww // 2)])
        if cursor < total:
            fi.extend(range(position, position + total - cursor))
        fi = torch.tensor(fi); h_ = fi.clone(); w_ = fi.clone()
        h_[image_pad_mask_.cpu()] = torch.tensor(ihi); w_[image_pad_mask_.cpu()] = torch.tensor(iwi)
        return fi, h_, w_
    fi, hi, wi = rope_ids(img_shapes[0], ipm_joint)
    with torch.no_grad():
        rot_real = tr.pos_embed(img_shapes[0], ipm_joint, device=dev)
    fr = tr.pos_embed.freqs
    rot_mine = torch.cat([fr[0][fi.to(dev)], fr[1][hi.to(dev)], fr[2][wi.to(dev)]], dim=-1)
    NPZ["rope_f"], NPZ["rope_h"], NPZ["rope_w"] = fi.numpy(), hi.numpy(), wi.numpy()
    rope_tab = []
    for sg in segs:
        a, b = sg["start"], sg["end"]
        rope_tab.append({"kind": sg["kind"], "start": a, "end": b,
                         "frame": [int(fi[a:b].min()), int(fi[a:b].max())],
                         "h": [int(hi[a:b].min()), int(hi[a:b].max())], "w": [int(wi[a:b].min()), int(wi[a:b].max())],
                         "first3": [[int(fi[k]), int(hi[k]), int(wi[k])] for k in range(a, min(a + 3, b))]})
    stage("9_rope", axes_dims=list(tr.config.axes_dims_rope), theta=10000, rotary_emb=st(rot_real.real if rot_real.is_complex() else rot_real),
          rotary_emb_shape=list(rot_real.shape), rotary_emb_dtype=str(rot_real.dtype),
          mirror_matches_real=bool(torch.equal(rot_real, rot_mine)), per_segment=rope_tab)
    log(f"rope: {rope_tab}")
    dump()

    # ================================================================== hooks on transformer internals
    calls = []
    cap = {"want": False}

    def pre_block0(mod, args, kwargs):
        hsx = kwargs.get("hidden_states", args[0] if args else None)
        rec = {"kv_mode": kwargs.get("kv_cache_mode"), "seq_len": int(hsx.shape[1]),
               "has_segments": kwargs.get("segments") is not None,
               "tmask_true": None if kwargs.get("target_token_mask") is None else int(kwargs["target_token_mask"].sum()),
               "tmask_len": None if kwargs.get("target_token_mask") is None else int(kwargs["target_token_mask"].shape[0])}
        if cap["want"]:
            cap["block0_in"] = hsx.detach().clone()
            cap["modulation"] = kwargs["modulation"].detach().clone()
            cap["segments"] = kwargs.get("segments")
            cap["target_token_mask"] = kwargs.get("target_token_mask")
        calls.append(rec)

    def pre_temb(mod, args):
        if cap["want"]:
            cap["timestep_in"] = args[0].detach().clone()

    def post_img_in(mod, a, o):
        if cap["want"]:
            cap["img_in_out"] = o.detach().clone()

    def post_txt_in(mod, a, o):
        if cap["want"]:
            cap["txt_in_out"] = o.detach().clone()

    def post_last(mod, a, o):
        if cap["want"]:
            cap["last_block_out"] = o.detach().clone()

    hooks = [tr.transformer_blocks[0].register_forward_pre_hook(pre_block0, with_kwargs=True),
             tr.time_text_embed.register_forward_pre_hook(pre_temb),
             tr.img_in.register_forward_hook(post_img_in), tr.txt_in.register_forward_hook(post_txt_in),
             tr.transformer_blocks[-1].register_forward_hook(post_last)]

    # ================================================================== denoising loop (mirror of :747-822)
    image_pad_mask_full = img_mask
    cache = QwenImage21KVCache(len(tr.transformer_blocks))
    pipe.scheduler.set_begin_index(0)
    step_log, x0_imgs, x0_lat = [], {}, {}

    def decode_tokens(tok):
        lat = pipe._unpack_latents(tok, height, width, pipe.vae_scale_factor).to(vae.dtype)
        lat = lat * std + mean
        im = vae.decode(lat, return_dict=False)[0][:, :, 0]
        return im

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            kv_mode = "extract" if i == 0 else "cached"
            lmi = torch.cat([cond_tokens, latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            cap["want"] = i in (0, 1)
            torch.cuda.synchronize(); t0 = time.time()
            out = tr(hidden_states=lmi, timestep=timestep / 1000, encoder_hidden_states=prompt_embeds,
                     encoder_hidden_states_mask=prompt_embeds_mask, img_shapes=img_shapes, img_mask=image_pad_mask_full,
                     kv_cache=cache, kv_cache_mode=kv_mode, return_dict=False)[0]
            torch.cuda.synchronize(); dt = time.time() - t0
            v = out[:, -latents.size(1):]
            sigma = float(pipe.scheduler.sigmas[i]); sigma_next = float(pipe.scheduler.sigmas[i + 1])
            x0_hat = latents.float() - sigma * v.float()
            rec = {"step": i + 1, "t": float(t), "t_bf16": float(timestep[0]), "timestep_passed": float(timestep[0] / 1000), "sigma": sigma, "sigma_next": sigma_next,
                   "kv_mode": kv_mode, "tokens_through_blocks": calls[-1]["seq_len"], "transformer_out_shape": list(out.shape),
                   "v": st(v), "x_t": st(latents), "x0_hat": st(x0_hat), "seconds": dt}
            if i == 0:
                stage("10_transformer_out", step1_full_out=st(out, "transformer out step1"), step1_tail=st(v, "v = out[:, -N:]"))
                cap0 = {k: v_ for k, v_ in cap.items() if k != "want"}
            if i == 1:
                stage("10_transformer_out", step2_full_out=st(out, "transformer out step2 (cached)"),
                      note_cached="with KV cache only the target rows go through the blocks, so out is already [1,N,64] and the tail slice is a no-op")
                cap1 = {k: v_ for k, v_ in cap.items() if k != "want"}
            if (i + 1) in X0_STEPS:
                im = decode_tokens(x0_hat.to(latents.dtype))
                x0_imgs[i + 1] = ((im[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                x0_lat[i + 1] = pipe._unpack_latents(x0_hat, height, width, 16)[0, :, 0].cpu().numpy()
            lat_before = latents
            latents = pipe.scheduler.step(v, t, latents, return_dict=False)[0]
            if i == 0:
                manual = lat_before.float() + (sigma_next - sigma) * v.float()
                stage("11_euler", formula="x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v", step1_scheduler_vs_manual_maxdiff=float((latents.float() - manual).abs().max()),
                      latents_dtype_after_step=str(latents.dtype))
            step_log.append(rec)
            log(f"step {i+1:2d} t={float(t):7.2f} sigma={sigma:.4f} mode={kv_mode} tokens={rec['tokens_through_blocks']} "
                f"out={rec['transformer_out_shape']} v.std={rec['v']['std']:.3f} x0.std={rec['x0_hat']['std']:.3f} {dt*1000:.0f}ms")
    for h in hooks:
        h.remove()
    TRACE["steps"] = step_log
    NPZ["x0_steps"] = np.array(sorted(x0_imgs))
    NPZ["x0_imgs"] = np.stack([x0_imgs[k] for k in sorted(x0_imgs)])
    NPZ["x0_lat"] = np.stack([x0_lat[k] for k in sorted(x0_lat)])
    NPZ["v_std"] = np.array([r["v"]["std"] for r in step_log]); NPZ["x0_std"] = np.array([r["x0_hat"]["std"] for r in step_log])

    # ---- stage 6/8/10 facts from captured internals at step 1
    b0 = cap0["block0_in"]; tmask = cap0["target_token_mask"]
    img_pos = ipm_joint.nonzero(as_tuple=True)[0]
    raster_ok = float((b0[0, img_pos] - cap0["img_in_out"][0]).abs().max())
    txt_pos = (~ipm_joint).nonzero(as_tuple=True)[0]
    # txt_in projects all VLM positions (incl. the image_pad slots, later overwritten); compare the non-pad rows
    txt_ok = float((b0[0, txt_pos] - cap0["txt_in_out"][0][~image_pad_mask[0]]).abs().max())
    stage("6_joint", block0_input=st(b0, "joint hidden states into block 0 (step1)"),
          img_in_out=st(cap0["img_in_out"]), txt_in_out=st(cap0["txt_in_out"]),
          image_positions_hold_img_in_in_raster_order_maxdiff=raster_ok, text_positions_hold_txt_in_maxdiff=txt_ok,
          prefix_segments_passed_to_processor=cap0["segments"])
    ts_in = cap0["timestep_in"]
    mod = cap0["modulation"]
    stage("8_modulation", timestep_into_time_embed_step1=[float(v) for v in ts_in], timestep_into_time_embed_step2=[float(v) for v in cap1["timestep_in"]],
          modulation=st(mod, "modulation (rows: real t, t=0)"), rows_differ_maxabs=float((mod[0] - mod[1]).abs().max()),
          target_token_mask_true=int(tmask.sum()), target_token_mask_len=int(tmask.shape[0]),
          tokens_on_t0_row=int((~tmask).sum()), tokens_on_real_t_row=int(tmask.sum()),
          step2_mask_len=int(cap1["target_token_mask"].shape[0]),
          chunks="modulation [B+1, 4*4096] -> mod1(scale,gate), mod2(scale,gate); gate passes through tanh")
    stage("12_kv_cache", per_call=calls[:3] + [{"...": len(calls)}], modes=[r["kv_mode"] for r in step_log],
          tokens_through_blocks=[r["tokens_through_blocks"] for r in step_log],
          seconds_per_step=[r["seconds"] for r in step_log],
          cached_prefix_len=int(cache.get_layer(0).k.shape[1]), cache_k_shape=list(cache.get_layer(0).k.shape),
          cache_bytes_total=int(sum(c.k.numel() * c.k.element_size() * 2 for c in cache.layer_caches)))
    dump()

    # ================================================================== stage 11: final decode
    with torch.no_grad():
        lat5 = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        lat5v = lat5.to(vae.dtype)
        den = lat5v * std + mean
        dec = vae.decode(den, return_dict=False)[0]
        im = dec[:, :, 0]
        pil = pipe.image_processor.postprocess(im, output_type="pil")[0]
    out_rgb = pil.convert("RGB")
    NPZ["final_rgba"] = np.asarray(pil)
    NPZ["final_rgb"] = np.asarray(out_rgb)
    ev = me.evaluate([s], [out_rgb])
    stage("11_euler", final_tokens=st(latents, "final tokens"), unpacked=st(lat5, "unpacked [B,C,T,h,w]"),
          denormalized=st(den, "x*std+mean"), vae_decode=st(dec, "vae.decode"), postprocess_mode=pil.mode, postprocess_size=list(pil.size),
          alpha_channel=st(np.asarray(pil)[..., 3]) if pil.mode == "RGBA" else None, evaluate=ev)
    log(f"eval: {ev}")
    out_rgb.save(OUT / "final_manual.png")
    dump()

    # ================================================================== stage 7 empirical: perturbation checks (no cache, full seq)
    try:
        perturbation_checks(tr, timesteps, latents, cond_tokens, prompt_embeds, img_shapes, image_pad_mask_full, segs,
                            first_pad, n_pad, kept_toks)
    except Exception as e:
        import traceback; traceback.print_exc(); stage("7_mask", empirical_error=repr(e))
    dump()
    try:
        pipeline_timing(pipe, s, prompt, dev)
    except Exception as e:
        import traceback; traceback.print_exc(); stage("12_kv_cache", pipeline_timing_error=repr(e))
    dump()
    log("TRACE DONE")


def perturbation_checks(tr, timesteps, latents, cond_tokens, prompt_embeds, img_shapes, image_pad_mask_full, segs,
                        first_pad, n_pad, kept_toks):
    with torch.no_grad():
        lastcap = {}
        hk = tr.transformer_blocks[-1].register_forward_hook(lambda m, a, o: lastcap.__setitem__("o", o.detach().float()))
        tt = timesteps[10].expand(1).to(latents.dtype) / 1000
        noise0 = torch.cat([cond_tokens, latents], 1)

        def run(hs_in, pe):
            tr(hidden_states=hs_in, timestep=tt, encoder_hidden_states=pe, encoder_hidden_states_mask=None,
               img_shapes=img_shapes, img_mask=image_pad_mask_full, return_dict=False)
            return lastcap["o"][0]
        base = run(noise0, prompt_embeds)
        cs = [sg for sg in segs if sg["kind"] == "cond0"][0]; tg = [sg for sg in segs if sg["kind"] == "target"][0]
        t_pre = [sg for sg in segs if sg["kind"] == "text"][0]; t_post = [sg for sg in segs if sg["kind"] == "text"][1]
        res = {}
        # (a) change target tokens -> prefix unchanged
        pert = noise0.clone(); pert[:, -latents.shape[1]:] += torch.randn_like(pert[:, -latents.shape[1]:])
        o = run(pert, prompt_embeds)
        res["perturb_target"] = {"prefix_maxdiff": float((o[:tg["start"]] - base[:tg["start"]]).abs().max()),
                                 "target_maxdiff": float((o[tg["start"]:] - base[tg["start"]:]).abs().max())}
        # (b) change only the LAST condition-latent token -> first cond token changes (bidirectional), pre-image text unchanged
        pert = noise0.clone(); pert[:, cs["len"] - 1] += 3.0
        o = run(pert, prompt_embeds)
        res["perturb_last_cond_token"] = {"text_before_maxdiff": float((o[:cs["start"]] - base[:cs["start"]]).abs().max()),
                                          "first_cond_token_maxdiff": float((o[cs["start"]] - base[cs["start"]]).abs().max()),
                                          "text_after_maxdiff": float((o[t_post["start"]:t_post["end"]] - base[t_post["start"]:t_post["end"]]).abs().max()),
                                          "target_maxdiff": float((o[tg["start"]:] - base[tg["start"]:]).abs().max())}
        # (c) change one text token in the middle of the post-image text -> earlier text unchanged, later changed
        j_vlm = first_pad + n_pad + (t_post["len"] // 2)  # index in kept VLM sequence
        j_joint = t_post["start"] + t_post["len"] // 2
        pe = prompt_embeds.clone(); pe[:, j_vlm] += 3.0 * pe[:, j_vlm].std()
        o = run(noise0, pe)
        res["perturb_mid_text_token"] = {"joint_index": j_joint, "token": kept_toks[j_vlm],
                                         "earlier_text_maxdiff": float((o[t_post["start"]:j_joint] - base[t_post["start"]:j_joint]).abs().max()),
                                         "cond_maxdiff": float((o[cs["start"]:cs["end"]] - base[cs["start"]:cs["end"]]).abs().max()),
                                         "later_text_maxdiff": float((o[j_joint + 1:t_post["end"]] - base[j_joint + 1:t_post["end"]]).abs().max()),
                                         "target_maxdiff": float((o[tg["start"]:] - base[tg["start"]:]).abs().max())}
        hk.remove()
    stage("7_mask", empirical_perturbation=res, empirical_note="last block output, full-sequence (no-cache) forward at step 11")
    log(f"perturbation: {res}")


def pipeline_timing(pipe, s, prompt, dev):

    # ================================================================== real pipeline: match + timing with/without cache
    def timed_pipe(use_cache):
        ts = []
        def cb(p, i, t, kw):
            torch.cuda.synchronize(); ts.append(time.time()); return kw
        g = torch.Generator(device=dev).manual_seed(SEED)
        torch.cuda.synchronize(); t0 = time.time()
        r = pipe(prompt=prompt, image=s["ref"], height=RES, width=RES, output_resolution=RES, num_inference_steps=STEPS,
                 generator=g, use_kv_cache=use_cache, callback_on_step_end=cb)
        torch.cuda.synchronize(); total = time.time() - t0
        per = np.diff([t0] + ts)
        return r.images[0], total, per
    res_pipe = {}
    for name, uc in [("cache_warm", True), ("cache", True), ("nocache", False)]:
        im_p, total, per = timed_pipe(uc)
        res_pipe[name] = {"total_s": total, "step1_s": float(per[0]), "median_later_step_s": float(np.median(per[1:])),
                          "per_step_s": [float(x) for x in per]}
        rgb = np.asarray(im_p.convert("RGB")).astype(float)
        res_pipe[name]["mae_vs_manual_uint8"] = float(np.abs(rgb - NPZ["final_rgb"].astype(float)).mean())
        res_pipe[name]["evaluate"] = me.evaluate([s], [im_p.convert("RGB")])
        im_p.convert("RGB").save(OUT / f"final_pipe_{name}.png")
        if name == "nocache":
            NPZ["final_rgb_nocache"] = np.asarray(im_p.convert("RGB"))
        log(f"pipe {name}: {res_pipe[name]['total_s']:.2f}s step1 {per[0]*1000:.0f}ms later {np.median(per[1:])*1000:.0f}ms "
            f"mae_vs_manual {res_pipe[name]['mae_vs_manual_uint8']:.3f} eval {res_pipe[name]['evaluate']}")
    stage("12_kv_cache", pipeline_timing=res_pipe, peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Qwen-Image 2.1 diffusers snapshot dir")
    ap.add_argument("--out", default="outputs/trace")
    ap.add_argument("--check", action="store_true", help="cheap check: imports + processor/tokenizer, no weights")
    a = ap.parse_args()
    CKPT = a.model
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        check_only()
    else:
        main()
