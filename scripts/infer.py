#!/usr/bin/env python3
"""Inference with Qwen-Image 2.1 on the MNIST edit tasks: one edit, or a configuration sweep (experiment C).

Single edit (a test pair picked by task, or your own image + instruction):
  python scripts/infer.py edit --model MODEL_DIR --task rot90 --index 0 --out outputs/edit
  python scripts/infer.py edit --model MODEL_DIR --image my.png --prompt "把图片旋转 180 度" --lora LORA_DIR

Sweep (every config sees the same 16 test pairs per task and the same per-sample noise):
  python scripts/infer.py sweep --model MODEL_DIR --phase 1 --out outputs/sweep/phase1
  python scripts/infer.py sweep --model MODEL_DIR --phase 2 --lora LORA_DIR --out outputs/sweep/phase2

Baseline = pipeline defaults (40 steps, true_cfg_scale 1, dynamic shift mu, use_kv_cache=True,
causal_condition=True), except output_resolution=512 (the task resolution; the pipeline default is 1024).
How each knob is overridden:
  steps       num_inference_steps=N
  cfg         true_cfg_scale=s, negative_prompt=""   (only when s > 1)
  shift       pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_cfg, use_dynamic_shifting=False,
                shift=s); the pipeline still passes mu, which the scheduler then ignores. shift_terminal=0.02 is kept.
  kv cache    use_kv_cache=False
  causal off  pipe.transformer.register_to_config(causal_condition=False); the pipeline then disables the KV cache
                by itself (cache_enabled = use_kv_cache and causal_condition)
  resolution  output_resolution=height=width=1024 (the condition image is also resized to 1024)
  lora scale  pipe.set_adapters(["lora"], [scale])
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qie_mnist import data as D
from qie_mnist import evaluate as E

RES = 512


def C(name, sweep, steps=40, cfg=1.0, shift=None, kv=True, causal=True, res=RES, subset="all", lora=None):
    return dict(name=name, sweep=sweep, steps=steps, cfg=cfg, shift=shift, kv=kv, causal=causal,
                res=res, subset=subset, lora=lora)


# Priority order: if the time budget runs out, later configs are skipped.
PHASE1 = [
    C("base", "baseline"),
    C("steps2", "steps", steps=2), C("steps4", "steps", steps=4),
    C("steps8", "steps", steps=8), C("steps16", "steps", steps=16),
    C("kvoff", "kv", kv=False),
    C("causaloff", "causal", kv=False, causal=False),
    C("cfg4", "cfg", cfg=4.0), C("cfg7", "cfg", cfg=7.0), C("cfg2", "cfg", cfg=2.0),
    C("shift1", "shift", shift=1.0), C("shift3", "shift", shift=3.0), C("shift6", "shift", shift=6.0),
    C("res1024", "res", res=1024, subset="display"),
]
PHASE2 = [
    C("lora1.0", "lora_scale", lora=1.0),
    C("lora0.0", "lora_scale", lora=0.0),
    C("lora0.5", "lora_scale", lora=0.5),
    C("lora1.5", "lora_scale", lora=1.5),
    C("lora1.0_steps4", "lora_steps", steps=4, lora=1.0),
    C("lora1.0_steps8", "lora_steps", steps=8, lora=1.0),
    C("lora1.0_cfg4", "lora_cfg", cfg=4.0, lora=1.0),
]


def img_stats(arr):
    """arr uint8 [H,W,3]. gray_frac: share of mid-gray pixels (blur / averaging); chroma: mean channel spread
    (MNIST is gray, so colour = artefact); clip: share of channels at 0 or 255."""
    small = np.asarray(Image.fromarray(arr).resize((128, 128), Image.BILINEAR)).astype(np.int16)
    g = small.mean(-1)
    return dict(gray_frac=float(((g > 48) & (g < 208)).mean()),
                chroma=float((small.max(-1) - small.min(-1)).mean()),
                clip=float(((small == 0) | (small == 255)).mean()))


def load_pipe(model):
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_lora(pipe, path, name="lora"):
    """A LoRA .safetensors file or a directory holding pytorch_lora_weights.safetensors."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.rglob("*.safetensors"))
        pref = [c for c in cands if c.name == "pytorch_lora_weights.safetensors"]
        if not (pref or cands):
            raise FileNotFoundError(f"no .safetensors under {p}")
        p = (pref or cands)[0]
    print("lora file:", p)
    pipe.load_lora_weights(str(p), adapter_name=name)
    return str(p.name)


def edit(pipe, image, prompt, steps=40, cfg=1.0, res=RES, kv=True, seed=0):
    kw = dict(prompt=prompt, image=image, num_inference_steps=steps, true_cfg_scale=cfg, output_resolution=res,
              height=res, width=res, use_kv_cache=kv, generator=torch.Generator("cuda").manual_seed(seed))
    if cfg > 1:
        kw["negative_prompt"] = ""
    return pipe(**kw).images[0].convert("RGB")


def cmd_edit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipe(a.model)
    if a.lora:
        load_lora(pipe, a.lora)
        pipe.set_adapters(["lora"], [a.lora_scale])
    sample = None
    if a.image:
        ref = Image.open(a.image).convert("RGB").resize((RES, RES), Image.BICUBIC)
        prompt = a.prompt or D.TASKS[a.task]
    else:
        samples = [s for s in D.make_pairs("test", a.n_per_task) if s["task"] == a.task]
        sample = samples[a.index]
        ref, prompt = sample["ref"], a.prompt or sample["prompt"]
    t0 = time.time()
    im = edit(pipe, ref, prompt, steps=a.steps, cfg=a.cfg, res=a.res, seed=a.seed)
    dt = time.time() - t0
    ref.save(out / "ref.png")
    im.save(out / "out.png")
    rec = dict(prompt=prompt, steps=a.steps, cfg=a.cfg, res=a.res, seed=a.seed, lora=bool(a.lora),
               lora_scale=a.lora_scale if a.lora else None, sec=round(dt, 2))
    if sample is not None:
        sample["target"].save(out / "target.png")
        rec.update(task=sample["task"], src_label=sample["src_label"], want_label=sample["want_label"],
                   metrics=E.evaluate([sample], [im]))
    json.dump(rec, open(out / "edit.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rec, ensure_ascii=False))


def cmd_sweep(a):
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    samples = D.make_pairs("test", a.n_per_task)
    display = []
    for task in D.TASK_LIST:
        display += [i for i, s in enumerate(samples) if s["task"] == task][:2]
    ref_dir = out / "images" / "_ref"
    ref_dir.mkdir(exist_ok=True)
    for i, s in enumerate(samples):
        s["ref"].resize((256, 256)).save(ref_dir / f"{i:02d}_ref.png")
        s["target"].resize((256, 256)).save(ref_dir / f"{i:02d}_tgt.png")
    json.dump([dict(i=i, task=s["task"], src_label=s["src_label"], want_label=s["want_label"])
               for i, s in enumerate(samples)], open(out / "eval_set.json", "w"), indent=1)

    from diffusers import FlowMatchEulerDiscreteScheduler
    import diffusers
    import transformers
    pipe = load_pipe(a.model)
    base_sched_cfg = dict(pipe.scheduler.config)
    lora_file = None
    if a.phase == 2:
        if not a.lora:
            raise SystemExit("--phase 2 needs --lora")
        lora_file = load_lora(pipe, a.lora)

    seq = {}

    def pre_hook(mod, args, kwargs):
        if "hs" not in seq:
            im = kwargs["img_mask"]
            seq["hs"] = int(kwargs["hidden_states"].shape[1])
            seq["enc"] = int(kwargs["encoder_hidden_states"].shape[1])
            seq["joint"] = int(im.shape[1] + 3 * int(im[0].sum()))
    pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    cfgs = PHASE1 if a.phase == 1 else PHASE2
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only.split(",")]
    meta = dict(phase=a.phase, n_per_task=a.n_per_task, seed0=a.seed0, display=display,
                versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__,
                              transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0)),
                scheduler_config=base_sched_cfg, lora_file=lora_file, results=[])
    mpath = out / f"metrics_phase{a.phase}.json"

    def run_one(i, c):
        s = samples[i]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        im = edit(pipe, s["ref"], s["prompt"], steps=c["steps"], cfg=c["cfg"], res=c["res"], kv=c["kv"],
                  seed=a.seed0 + i)
        torch.cuda.synchronize()
        return im, time.perf_counter() - t0

    run_one(0, C("warm", "warm", steps=2))  # warm-up (kernels / allocator); not timed

    keep = {}  # full-res uint8 outputs of reference configs, for exact-diff comparisons
    for c in cfgs:
        el = (time.time() - t_start) / 60
        if el > a.max_minutes:
            print(f"SKIP {c['name']} (budget {el:.1f} > {a.max_minutes} min)")
            meta["results"].append(dict(c, skipped=True))
            continue
        if c["shift"] is None:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_sched_cfg)
        else:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                base_sched_cfg, use_dynamic_shifting=False, shift=c["shift"])
        pipe.transformer.register_to_config(causal_condition=c["causal"])
        if c["lora"] is not None:
            pipe.set_adapters(["lora"], [c["lora"]])
        ids = list(range(len(samples))) if c["subset"] == "all" else display
        d = out / "images" / c["name"]
        d.mkdir(exist_ok=True)
        outs, times, stats = [], [], []
        seq.clear()
        for i in ids:
            im, dt = run_one(i, c)
            outs.append(im)
            times.append(dt)
            arr = np.asarray(im)
            stats.append(img_stats(arr))
            im.resize((256, 256), Image.BICUBIC).save(d / f"{i:02d}.png")
            if i in display:
                im.save(d / f"{i:02d}_full.png")
            if c["name"] in ("base", "kvoff", "lora1.0"):
                keep.setdefault(c["name"], {})[i] = arr
        sub = [samples[i] for i in ids]
        ev = E.evaluate(sub, outs)
        r = dict(c, n=len(ids), metrics=ev,
                 success_mean=float(np.mean([v["success"] for v in ev.values()])),
                 sec_per_img=float(np.mean(times)), sec_per_img_median=float(np.median(times)),
                 img_stats={k: float(np.mean([s_[k] for s_ in stats])) for k in stats[0]},
                 sigmas=[round(float(x), 5) for x in pipe.scheduler.sigmas.cpu()], seq=dict(seq))
        out28 = [D.from_pil(o) for o in outs]
        pred = E.classify([E.undo(s_["task"], o) for s_, o in zip(sub, out28)])
        r["per_sample_success"] = {int(i): bool(p == s_["want_label"]) for i, p, s_ in zip(ids, pred, sub)}
        np.save(d / "out28.npy", np.stack(out28))
        if c["name"] == "kvoff" and "base" in keep:
            diffs = [np.abs(keep["base"][i].astype(np.int16) - keep["kvoff"][i].astype(np.int16)) for i in ids]
            r["vs_base"] = dict(max_abs_diff=int(max(x.max() for x in diffs)),
                                mean_abs_diff=float(np.mean([x.mean() for x in diffs])),
                                identical_imgs=int(sum(int(x.max() == 0) for x in diffs)), n=len(ids))
        if c["name"] == "causaloff" and "kvoff" in keep:
            arrs = {i: np.asarray(o) for i, o in zip(ids, outs)}
            diffs = [np.abs(keep["kvoff"][i].astype(np.int16) - arrs[i].astype(np.int16)) for i in ids]
            r["vs_kvoff"] = dict(max_abs_diff=int(max(x.max() for x in diffs)),
                                 mean_abs_diff=float(np.mean([x.mean() for x in diffs])))
        meta["results"].append(r)
        meta["elapsed_min"] = (time.time() - t_start) / 60
        json.dump(meta, open(mpath, "w"), indent=1)
        print(f"[{meta['elapsed_min']:.1f} min] {c['name']}: success {r['success_mean']:.3f} "
              f"{r['sec_per_img']:.2f}s/img seq={r['seq']} stats={r['img_stats']}", flush=True)
    json.dump(meta, open(mpath, "w"), indent=1)
    print("DONE", mpath)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit", help="one edit")
    e.add_argument("--model", required=True, help="local Qwen-Image 2.1 snapshot dir (or HF repo id)")
    e.add_argument("--out", default="outputs/edit")
    e.add_argument("--task", default="rot90", choices=D.TASK_LIST)
    e.add_argument("--index", type=int, default=0, help="which test pair of that task")
    e.add_argument("--n-per-task", type=int, default=16)
    e.add_argument("--image", default=None, help="your own input image instead of a test pair")
    e.add_argument("--prompt", default=None, help="instruction (default: the task's fixed instruction)")
    e.add_argument("--lora", default=None)
    e.add_argument("--lora-scale", type=float, default=1.0)
    e.add_argument("--steps", type=int, default=40)
    e.add_argument("--cfg", type=float, default=1.0)
    e.add_argument("--res", type=int, default=RES)
    e.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sweep", help="experiment C sweep")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1: base model knobs, 2: LoRA knobs")
    s.add_argument("--lora", default=None)
    s.add_argument("--only", default="", help="comma-separated config names")
    s.add_argument("--n-per-task", type=int, default=16)
    s.add_argument("--seed0", type=int, default=1000, help="sample i uses torch.Generator('cuda').manual_seed(seed0+i)")
    s.add_argument("--max-minutes", type=float, default=75)
    a = ap.parse_args()
    cmd_edit(a) if a.cmd == "edit" else cmd_sweep(a)


if __name__ == "__main__":
    main()
