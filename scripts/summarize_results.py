#!/usr/bin/env python3
"""Step 4 - evaluation summary: print the small JSON files in results/ (or your own outputs) as the tables of the
README's results section. CPU, seconds.

Deck: p.13 (Success rate) and p.15 (Learnable vs not learnable) for B, p.11 (loss = x0 error x 1/sigma^2) for D,
p.16-19 (CFG, Shift, LoRA scale, Sampling steps) for C, plus the README's comparison of the training variants E and F.

Only 16 test pairs per task: the success rate moves in steps of 1/16 = 6.25 points, so one image is about 6 points.

Usage:
  python scripts/summarize_results.py                          # everything, from results/
  python scripts/summarize_results.py b e f                    # training only
  python scripts/summarize_results.py c --sweep outputs/sweep  # your own sweep outputs
  python scripts/summarize_results.py b --train outputs/lora_b/hook_logs
"""
import argparse
import json
import os

TASKS = ["rot90", "rot180", "next", "invert"]
ZH = {"rot90": "rot90", "rot180": "rot180", "next": "next", "invert": "invert"}


def jl(path):
    with open(path) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    by = {r["step"]: r for r in recs}  # a resumed run replays some steps: keep the last record per step
    return [by[k] for k in sorted(by)]


def pct(x):
    return f"{100 * x:.0f}%"


def pct1(x):
    return f"{100 * x:.1f}%"


def cell(m, t):
    r = m[t]
    extra = f" / {r['iou']:.2f}" if "iou" in r else f" / copied {r['copied_source']:.2f}"
    return pct(r["success"]) + extra


def first_ge(val, t, thr=0.9):
    for r in val:
        if r["step"] > 0 and r["metrics"][t]["success"] >= thr:
            return r["step"]
    return None


def table(rows, head):
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def show_train(d, name, steps=(0, 200, 400, 600, 1000, 2000, 3000)):
    val, probe = jl(os.path.join(d, "val.jsonl")), jl(os.path.join(d, "probe.jsonl"))
    pm = {r["step"]: r["mean"] for r in probe}
    print(f"### {name}: validation success / IoU (next: success / share copied from the source), 40 steps, CFG 1, "
          f"16 pairs per task\n")
    rows = [[r["step"]] + [cell(r["metrics"], t) for t in TASKS] + [pct1(r["mean_success"]),
                                                                    f"{pm.get(r['step'], float('nan')):.4f}"]
            for r in val if r["step"] in steps]
    table(rows, ["step"] + [ZH[t] for t in TASKS] + ["mean", "probe loss"])
    return val, probe


def cmd_b(a):
    val, probe = show_train(a.train, "B (baseline)")
    print("first step >= 90%:", {ZH[t]: first_ge(val, t) for t in TASKS})
    f0, f1 = probe[0]["by_sigma"], probe[-1]["by_sigma"]
    print("probe loss by sigma, step 0 -> last step:",
          {s: f"{f0[s]:.4f} -> {f1[s]:.4f} (-{100 * (1 - f1[s] / f0[s]):.0f}%)" for s in f0})
    print()


def variant(a, d, name):
    b_val = jl(os.path.join(a.train, "val.jsonl"))
    v_val, v_probe = show_train(d, name)
    b_probe = jl(os.path.join(a.train, "probe.jsonl"))
    print(f"first step >= 90% (B -> {name[0]}):", {ZH[t]: f"{first_ge(b_val, t)} -> {first_ge(v_val, t)}" for t in TASKS})
    mid = lambda v, k: sum(r["metrics"]["next"][k] for r in v if r["step"] >= 200) / sum(r["step"] >= 200 for r in v)  # noqa: E731
    print(f"next, mean over steps 200-3000: success B {pct1(mid(b_val, 'success'))} vs {name[0]} {pct1(mid(v_val, 'success'))}; "
          f"copied B {mid(b_val, 'copied_source'):.3f} vs {name[0]} {mid(v_val, 'copied_source'):.3f}")
    s200 = {r["step"]: r for r in b_val}.get(200), {r["step"]: r for r in v_val}.get(200)
    if all(s200):
        print(f"mean success at step 200: B {pct1(s200[0]['mean_success'])} vs {name[0]} {pct1(s200[1]['mean_success'])}")
    bl, vl = b_probe[-1]["by_task"], v_probe[-1]["by_task"]
    print("last-step probe loss (sigma 0.1 / 0.9): " + "; ".join(
        f"{ZH[t]} B {bl[t]['0.1']:.4f}/{bl[t]['0.9']:.4f} vs {name[0]} {vl[t]['0.1']:.4f}/{vl[t]['0.9']:.4f}"
        for t in TASKS))
    sp = os.path.join(d, "sigma_summary.json")
    if os.path.exists(sp):
        bs, es = json.load(open(os.path.join(a.train, "sigma_summary.json"))), json.load(open(sp))
        print(f"sigma actually used in training: mean B {bs['mean_sigma']:.3f} vs {name[0]} {es['mean_sigma']:.3f}; "
              f"share > 0.9 B {pct(bs['share_sigma_gt_0_9'])} vs {pct(es['share_sigma_gt_0_9'])}; "
              f"share < 0.3 B {pct(bs['share_sigma_lt_0_3'])} vs {pct(es['share_sigma_lt_0_3'])}")
    print()


def load_sweep(a, phase):
    p = os.path.join(a.sweep, f"phase{phase}", f"metrics_phase{phase}.json")
    m = json.load(open(p))
    return {r["name"]: r for r in m["results"] if not r.get("skipped")}


def sweep_rows(R, names):
    rows = []
    for lab, n in names:
        if n not in R:
            continue
        r = R[n]
        rows.append([lab, pct1(r["success_mean"])] + [cell(r["metrics"], t) for t in TASKS]
                    + [f"{r['img_stats']['chroma']:.1f}", f"{r['sec_per_img']:.2f}"])
    return rows


def cmd_c(a):
    head = ["config", "mean"] + [ZH[t] for t in TASKS] + ["colour artefact", "s/img"]
    try:
        R2 = load_sweep(a, 2)
        print("### C phase 2: LoRA scale (step-3000 LoRA, 40 steps, CFG 1)\n")
        table(sweep_rows(R2, [(f"scale {s}", f"lora{s}") for s in ("0.0", "0.5", "1.0", "1.5")]), head)
    except FileNotFoundError:
        pass
    R3 = load_sweep(a, 3)
    print("### C phase 3: inference settings with the LoRA (scale 1.0)\n")
    names = ([("CFG 1 (default)", "lora1.0")] + [(f"CFG {c}", f"lora_cfg{c}") for c in (2, 4, 7)]
             + [(f"shift {s}", f"lora_shift{s}") for s in (1, 3, 6)]
             + [(f"{k} steps", f"lora_steps{k}") for k in (2, 4, 8, 16)])
    table(sweep_rows(R3, names), head)
    print("Note: \"colour artefact\" is img_stats.chroma in the metrics (RGB channel spread of the full output resized to "
          "128 px). The deck figures recompute it per task from the 256 px thumbnails (plot_sweep.py), so the values "
          "differ slightly.\n")


def cmd_d(a):
    R = json.load(open(os.path.join(a.probe, "results.json")))
    L = R.get("probe_lora") or R["probe_base"]
    sig = [str(s) for s in R["sigmas"]]
    print(f"### D: held-out loss vs sigma ({'LoRA' if 'probe_lora' in R else 'base model'}, 16 pairs x {R['n_seeds']} "
          f"noise draws)\n")
    table([[ZH[t]] + [f"{L[t][s]['mean']:.4f}" for s in sig] for t in TASKS], ["task"] + [f"σ={s}" for s in sig])
    p1 = R["part1"]
    print(f"D_next (within-class latent variance on the train split, mean over 10 classes) = {p1['D_next_train']:.4f}; "
          f"D = 0 for the deterministic tasks; next loss at sigma = 1 = {L['next']['1.0']['mean']:.3f} "
          f"≈ {L['next']['1.0']['mean'] / p1['D_next_train']:.1f} x D\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", default=["b", "e", "f", "c", "d"], help="any of b e f c d (default: all)")
    ap.add_argument("--train", default="results/train", help="B hook_logs (val.jsonl, probe.jsonl)")
    ap.add_argument("--e", default="results/train_e_shift5", help="E hook_logs (train-time sigma shift 5)")
    ap.add_argument("--f", default="results/train_f_proto", help="F hook_logs (fixed next prototypes)")
    ap.add_argument("--sweep", default="results/sweep", help="dir holding phase2/ and phase3/")
    ap.add_argument("--probe", default="results/probe", help="output dir of probe_loss.py measure")
    a = ap.parse_args()
    for p in a.parts:
        if p == "b":
            cmd_b(a)
        elif p == "e":
            variant(a, a.e, "E (train-time sigma shift 5)")
        elif p == "f":
            variant(a, a.f, "F (fixed next prototypes)")
        elif p == "c":
            cmd_c(a)
        elif p == "d":
            cmd_d(a)
        else:
            raise SystemExit(f"unknown part {p}")


if __name__ == "__main__":
    main()
