"""Training hooks: logging, the fixed held-out probe and periodic validation for the vendored diffusers
Qwen-Image 2.1 img2img LoRA trainer.

Deck: p.8 (Training loss) <- train.jsonl, p.9 (Loss by sigma) <- probe.jsonl,
p.13 (Success rate) and p.14 (Before vs after training) <- val.jsonl, val/step_*/.

The trainer (train_dreambooth_lora_qwenimage21_img2img.py, diffusers e0abab8) calls into this module at four
marked points ("[qie-mnist hook n/4]"). Nothing here changes what is trained: the hooks only read the model,
in eval mode under torch.no_grad, and write logs / images / LoRA files. (The trainer's only training change,
the optional --train_shift, is patch 5 in the trainer itself.)

  prepare()         encode the fixed test set once, while the text encoder and VAE are loaded
  log_train_step()  one JSON line per optimizer step: loss, lr, sampled sigma          -> train.jsonl
  on_step()         every HOOK_PROBE_EVERY steps: held-out loss at fixed sigmas with fixed noise -> probe.jsonl
                    every HOOK_VAL_EVERY steps (and step 0 / last step): 40-step pipeline sampling on the test
                    set, scored with qie_mnist.evaluate()                                -> val.jsonl, val/step_*/
                    every HOOK_CKPT_EVERY steps (and last step): save the LoRA with the same call the script
                    uses for its final save                                              -> <output_dir>/ckpt-<step>/

Environment variables (set by scripts/train_lora.sh):
  HOOK_OUT            output dir for logs and images (default <output_dir>/hook_logs)
  HOOK_N_VAL (16)     test pairs per task         HOOK_N_GRID (2)  grid examples per task
  HOOK_VAL_EVERY (200)  HOOK_PROBE_EVERY (50)  HOOK_CKPT_EVERY (500)
  HOOK_NEXT_TARGET    random | proto: next targets of the fixed test set; must match the training data
                      (train_lora.sh reads it from DATA_DIR/summary.json). The test inputs are the same either way.
  HOOK_DATA_SUMMARY   the training data's summary.json (only copied into meta.json / FINAL.json)
"""
import json
import os
import time

import numpy as np
import torch

from qie_mnist import data as D
from qie_mnist import evaluate as E
from qie_mnist.probe import fixed_noise, probe_loss, to_pixels

PROBE_SIGMAS = [0.1, 0.3, 0.5, 0.7, 0.9]
S = {}


def _env(k, d):
    return type(d)(os.environ.get(k, d))


def _log(msg):
    print(f"[train_hooks {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _append(name, rec):
    with open(os.path.join(S["out"], name), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _data_summary():
    p = os.environ.get("HOOK_DATA_SUMMARY", "")
    if p and os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return {k: d.get(k) for k in ("n", "n_per_task", "tasks", "next_target", "per_task", "seed")}
    return None


def prepare(args, accelerator, text_encoding_pipeline, compute_text_embeddings, vae, latents_mean, latents_std,
            processor, weight_dtype, validation_pipeline_cls, noise_scheduler):
    from diffusers.training_utils import offload_models

    S.update(out=os.environ.get("HOOK_OUT", os.path.join(args.output_dir, "hook_logs")),
             n_val=_env("HOOK_N_VAL", 16), n_grid=_env("HOOK_N_GRID", 2),
             val_every=_env("HOOK_VAL_EVERY", 200), probe_every=_env("HOOK_PROBE_EVERY", 50),
             ckpt_every=_env("HOOK_CKPT_EVERY", 500), next_target=os.environ.get("HOOK_NEXT_TARGET", "random"),
             data_summary=_data_summary(),
             device=accelerator.device, dtype=weight_dtype, args=args, processor=processor,
             pipe_cls=validation_pipeline_cls, pipe=None, t_start=time.time(), train_buf=[])
    os.makedirs(os.path.join(S["out"], "val"), exist_ok=True)

    samples = D.make_pairs("test", S["n_val"], next_target=S["next_target"])
    S["samples"] = samples
    dev = accelerator.device
    t0 = time.time()
    embeds, masks = [], []
    with offload_models(text_encoding_pipeline, device=dev, offload=args.offload):
        for s in samples:
            e, m, pad = compute_text_embeddings(s["prompt"], text_encoding_pipeline, s["ref"].convert("RGBA"))
            embeds.append((e.to(dev), pad.to(dev)))
            masks.append(None if m is None else m.to(dev))
    S["embeds"], S["masks"] = embeds, masks
    with torch.no_grad(), offload_models(vae, device=dev, offload=args.offload):
        tgt, cond = [], []
        for s in samples:
            x = vae.encode(to_pixels(s["target"]).to(dev, vae.dtype)).latent_dist.mode()
            c = vae.encode(to_pixels(s["ref"]).to(dev, vae.dtype)).latent_dist.mode()
            # The trainer's latents_std is already the reciprocal (1 / std), hence the multiplication.
            tgt.append(((x - latents_mean) * latents_std).to(weight_dtype))
            cond.append(((c - latents_mean) * latents_std).to(weight_dtype))
    S["tgt_lat"], S["cond_lat"] = tgt, cond
    S["noise"] = fixed_noise([x.shape for x in tgt], 1234, dev, weight_dtype)
    # Grid examples: the first n_grid test pairs of each task.
    grid = []
    for task in D.TASK_LIST:
        grid += [i for i, s in enumerate(samples) if s["task"] == task][: S["n_grid"]]
    S["grid"] = grid
    meta = dict(n_val=len(samples), tasks=D.TASK_LIST, grid=grid, probe_sigmas=PROBE_SIGMAS,
                val_every=S["val_every"], probe_every=S["probe_every"], ckpt_every=S["ckpt_every"],
                test_next_target=S["next_target"], train_data=S["data_summary"],
                train_shift=float(getattr(args, "train_shift", 1.0)),
                samples=[dict(i=i, task=s["task"], prompt=s["prompt"], src_label=s["src_label"],
                              want_label=s["want_label"], seed=1000 + i) for i, s in enumerate(samples)],
                args={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                      for k, v in vars(args).items()},
                scheduler_config=dict(noise_scheduler.config))
    with open(os.path.join(S["out"], "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    for i in grid:
        samples[i]["ref"].resize((256, 256)).save(os.path.join(S["out"], "val", f"ref_{i:02d}.png"))
        samples[i]["target"].resize((256, 256)).save(os.path.join(S["out"], "val", f"tgt_{i:02d}.png"))
    _log(f"prepared {len(samples)} test pairs in {time.time() - t0:.0f}s; prompt len "
         f"{sorted({e[0].shape[1] for e in embeds})}; pad slots {sorted({int(e[1].sum()) for e in embeds})}")


def log_train_step(step, logs, sigmas):
    rec = dict(step=int(step), loss=float(logs["loss"]), lr=float(logs["lr"]),
               sigma=[round(float(x), 5) for x in sigmas.flatten().tolist()], t=round(time.time() - S["t_start"], 2))
    S["train_buf"].append(rec)
    if len(S["train_buf"]) >= 25:
        _flush_train()


def _flush_train():
    with open(os.path.join(S["out"], "train.jsonl"), "a") as f:
        for r in S["train_buf"]:
            f.write(json.dumps(r) + "\n")
    S["train_buf"] = []


@torch.no_grad()
def _probe(step, transformer):
    t0 = time.time()
    per = probe_loss(transformer, S["samples"], S["embeds"], S["tgt_lat"], S["cond_lat"], [S["noise"]],
                     PROBE_SIGMAS, device=S["device"])
    per = {t: {s: v[0] for s, v in d.items()} for t, d in per.items()}
    rec = dict(step=int(step), sec=round(time.time() - t0, 1),
               by_task={t: {s: float(np.mean(v)) for s, v in d.items()} for t, d in per.items()},
               by_sigma={str(s): float(np.mean([np.mean(per[t][str(s)]) for t in D.TASK_LIST])) for s in PROBE_SIGMAS})
    rec["mean"] = float(np.mean(list(rec["by_sigma"].values())))
    _append("probe.jsonl", rec)
    _log(f"probe step {step}: mean {rec['mean']:.4f} " + " ".join(f"{k}:{v:.3f}" for k, v in rec["by_sigma"].items())
         + f" ({rec['sec']}s)")


def _pipeline(transformer):
    if S["pipe"] is None:
        args = S["args"]
        pipe = S["pipe_cls"].from_pretrained(args.pretrained_model_name_or_path, text_encoder=None,
                                              processor=S["processor"], transformer=transformer,
                                              torch_dtype=S["dtype"])
        pipe.vae.to(S["device"], S["dtype"])
        pipe.set_progress_bar_config(disable=True)
        S["pipe"] = pipe
    return S["pipe"]


@torch.no_grad()
def _validate(step, transformer):
    t0 = time.time()
    pipe = _pipeline(transformer)
    outs = []
    d = os.path.join(S["out"], "val", f"step_{step:05d}")
    os.makedirs(d, exist_ok=True)
    nsteps = S["args"].validation_num_inference_steps
    for i, s in enumerate(S["samples"]):
        e, pad = S["embeds"][i]
        pipe.cached_image_pad_mask = pad
        with torch.autocast("cuda", dtype=torch.bfloat16):
            im = pipe(prompt_embeds=e, prompt_embeds_mask=S["masks"][i], image=s["ref"], num_inference_steps=nsteps,
                      true_cfg_scale=1.0, output_resolution=S["args"].resolution,
                      generator=torch.Generator(device=S["device"]).manual_seed(1000 + i)).images[0]
        im = im.convert("RGB")
        outs.append(im)
        im.resize((256, 256) if i in S["grid"] else (128, 128)).save(os.path.join(d, f"{i:02d}.png"))
    res = E.evaluate(S["samples"], outs)
    out28 = [D.from_pil(o) for o in outs]
    np.save(os.path.join(d, "out28.npy"), np.stack(out28))
    pred = E.classify([E.undo(s["task"], o) for s, o in zip(S["samples"], out28)])
    rec = dict(step=int(step), sec=round(time.time() - t0, 1), steps=nsteps, cfg=1.0, metrics=res,
               mean_success=float(np.mean([r["success"] for r in res.values()])),
               pred=[int(p) for p in pred])
    _append("val.jsonl", rec)
    _log(f"val step {step}: mean success {rec['mean_success']:.3f} "
         + " ".join(f"{t}:{r['success']:.2f}" for t, r in res.items()) + f" ({rec['sec']}s)")


def _save_ckpt(step, transformer, final):
    from diffusers import QwenImage21Pipeline
    from diffusers.training_utils import _collate_lora_metadata
    from peft.utils import get_peft_model_state_dict
    args = S["args"]
    local = os.path.join(args.output_dir, f"ckpt-{step}")
    # Same call the trainer makes for its final save (weights kept in the training dtype).
    QwenImage21Pipeline.save_lora_weights(save_directory=local,
                                          transformer_lora_layers=get_peft_model_state_dict(transformer),
                                          **_collate_lora_metadata({"transformer": transformer}))
    _log(f"saved LoRA -> {local}")
    if final:
        keys = ["rank", "lora_alpha", "lora_layers", "learning_rate", "lr_scheduler", "max_train_steps",
                "train_batch_size", "gradient_accumulation_steps", "resolution", "mixed_precision", "weighting_scheme",
                "logit_mean", "logit_std", "seed", "optimizer", "adam_weight_decay", "max_grad_norm"]
        marker = dict(checkpoint=f"ckpt-{step}/", weight_file="pytorch_lora_weights.safetensors", step=int(step),
                      base_model="Qwen/Qwen-Image-2.1 rev 790c92633540aa0cb11d9abf19eb46d861714758",
                      diffusers_commit="e0abab83b5df05de9e7abd788643c1a7c1e42e28",
                      trainer="train/train_dreambooth_lora_qwenimage21_img2img.py (+4 logging hooks, +--train_shift)",
                      load="pipe.load_lora_weights('<output_dir>/ckpt-<step>')",
                      sigma_sampling=_sigma_desc(float(getattr(args, "train_shift", 1.0))),
                      hyperparameters={k: getattr(args, k, None) for k in keys + ["train_shift"]},
                      data=S["data_summary"] or "unknown (no HOOK_DATA_SUMMARY)",
                      test_next_target=S["next_target"],
                      written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(os.path.join(args.output_dir, "FINAL.json"), "w") as f:
            json.dump(marker, f, ensure_ascii=False, indent=1)


def _sigma_desc(shift):
    if shift == 1.0:
        return ("weighting_scheme=none: sigma ~ U(0,1) at train time, no shift "
                "(scheduler use_dynamic_shifting; shift applied only at sampling)")
    return f"sigma ~ U(0,1) then train-time shift s={shift:g}: s*sigma/(1+(s-1)*sigma)"


def on_step(step, transformer, args):
    last = step >= args.max_train_steps
    ckpt = step > 0 and (step % S["ckpt_every"] == 0 or last)
    probe = step % S["probe_every"] == 0 or last
    val = step % S["val_every"] == 0 or last
    if not (ckpt or probe or val):
        return
    _flush_train()
    was_training = transformer.training
    transformer.eval()
    try:
        if step == 0:
            lp = [p for n, p in transformer.named_parameters() if "lora" in n]
            _log(f"LoRA tensors {len(lp)}, params {sum(p.numel() for p in lp) / 1e6:.2f}M, dtype {lp[0].dtype}")
        if ckpt:
            _save_ckpt(step, transformer, final=last)
        if probe:
            _probe(step, transformer)
        if val:
            _validate(step, transformer)
        if step == 0:
            _log(f"cuda max mem so far {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    finally:
        if was_training:
            transformer.train()
        torch.cuda.empty_cache()
