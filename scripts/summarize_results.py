#!/usr/bin/env python3
"""Step 4 - evaluation summary: print the small JSON files in results/ (or your own outputs) as the tables of the
README's results section. CPU, seconds. The printed tables are in Chinese, like the README.

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
ZH = {"rot90": "90°", "rot180": "180°", "next": "next", "invert": "反色"}


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
    extra = f" / {r['iou']:.2f}" if "iou" in r else f" / 抄 {r['copied_source']:.2f}"
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
    print(f"### {name}：验证成功率 / IoU（next 为成功率 / 抄原数字比例），40 步，CFG 1，每任务 16 张\n")
    rows = [[r["step"]] + [cell(r["metrics"], t) for t in TASKS] + [pct1(r["mean_success"]),
                                                                    f"{pm.get(r['step'], float('nan')):.4f}"]
            for r in val if r["step"] in steps]
    table(rows, ["step"] + [ZH[t] for t in TASKS] + ["平均", "probe loss"])
    return val, probe


def cmd_b(a):
    val, probe = show_train(a.train, "B（基线）")
    print("首次 >= 90%：", {ZH[t]: first_ge(val, t) for t in TASKS})
    f0, f1 = probe[0]["by_sigma"], probe[-1]["by_sigma"]
    print("probe loss 按 σ，第 0 步 → 最后一步：",
          {s: f"{f0[s]:.4f} → {f1[s]:.4f}（降 {100 * (1 - f1[s] / f0[s]):.0f}%）" for s in f0})
    print()


def variant(a, d, name):
    b_val = jl(os.path.join(a.train, "val.jsonl"))
    v_val, v_probe = show_train(d, name)
    b_probe = jl(os.path.join(a.train, "probe.jsonl"))
    print(f"首次 >= 90%（B → {name[0]}）：", {ZH[t]: f"{first_ge(b_val, t)} → {first_ge(v_val, t)}" for t in TASKS})
    mid = lambda v, k: sum(r["metrics"]["next"][k] for r in v if r["step"] >= 200) / sum(r["step"] >= 200 for r in v)  # noqa: E731
    print(f"next 第 200–3000 步平均：成功率 B {pct1(mid(b_val, 'success'))} vs {name[0]} {pct1(mid(v_val, 'success'))}；"
          f"抄原数字 B {mid(b_val, 'copied_source'):.3f} vs {name[0]} {mid(v_val, 'copied_source'):.3f}")
    s200 = {r["step"]: r for r in b_val}.get(200), {r["step"]: r for r in v_val}.get(200)
    if all(s200):
        print(f"第 200 步平均成功率：B {pct1(s200[0]['mean_success'])} vs {name[0]} {pct1(s200[1]['mean_success'])}")
    bl, vl = b_probe[-1]["by_task"], v_probe[-1]["by_task"]
    print(f"最后一步 probe loss（σ=0.1 / 0.9）：" + "；".join(
        f"{ZH[t]} B {bl[t]['0.1']:.4f}/{bl[t]['0.9']:.4f} vs {name[0]} {vl[t]['0.1']:.4f}/{vl[t]['0.9']:.4f}"
        for t in TASKS))
    sp = os.path.join(d, "sigma_summary.json")
    if os.path.exists(sp):
        bs, es = json.load(open(os.path.join(a.train, "sigma_summary.json"))), json.load(open(sp))
        print(f"训练时实际的 σ：平均 B {bs['mean_sigma']:.3f} vs {name[0]} {es['mean_sigma']:.3f}；"
              f"σ > 0.9 占 B {pct(bs['share_sigma_gt_0_9'])} vs {pct(es['share_sigma_gt_0_9'])}；"
              f"σ < 0.3 占 B {pct(bs['share_sigma_lt_0_3'])} vs {pct(es['share_sigma_lt_0_3'])}")
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
    head = ["配置", "平均"] + [ZH[t] for t in TASKS] + ["彩色伪影", "s/张"]
    try:
        R2 = load_sweep(a, 2)
        print("### C 第 2 轮：LoRA 强度（第 3000 步 LoRA，40 步，CFG 1）\n")
        table(sweep_rows(R2, [(f"强度 {s}", f"lora{s}") for s in ("0.0", "0.5", "1.0", "1.5")]), head)
    except FileNotFoundError:
        pass
    R3 = load_sweep(a, 3)
    print("### C 第 3 轮：带 LoRA（强度 1.0）的推理设置\n")
    names = ([("CFG 1（默认）", "lora1.0")] + [(f"CFG {c}", f"lora_cfg{c}") for c in (2, 4, 7)]
             + [(f"shift {s}", f"lora_shift{s}") for s in (1, 3, 6)]
             + [(f"{k} 步", f"lora_steps{k}") for k in (2, 4, 8, 16)])
    table(sweep_rows(R3, names), head)
    print("注：「彩色伪影」是 metrics 里 img_stats.chroma（全分辨率输出缩到 128 px 后的 RGB 通道差）。"
          "讲义图里的数字由 plot_sweep.py 从 256 px 缩略图重算（按任务拆开），数值略有不同。\n")


def cmd_d(a):
    R = json.load(open(os.path.join(a.probe, "results.json")))
    L = R.get("probe_lora") or R["probe_base"]
    sig = [str(s) for s in R["sigmas"]]
    print(f"### D：固定测试集 loss vs σ（{'LoRA' if 'probe_lora' in R else '底模'}，16 张 × {R['n_seeds']} 组噪声平均）\n")
    table([[ZH[t]] + [f"{L[t][s]['mean']:.4f}" for s in sig] for t in TASKS], ["任务"] + [f"σ={s}" for s in sig])
    p1 = R["part1"]
    print(f"D_next（训练集 latent 类内方差，10 类平均）= {p1['D_next_train']:.4f}；确定性任务 D = 0；"
          f"σ=1 时 next 的 loss = {L['next']['1.0']['mean']:.3f} ≈ {L['next']['1.0']['mean'] / p1['D_next_train']:.1f} × D\n")


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
            variant(a, a.e, "E（训练时 σ shift 5）")
        elif p == "f":
            variant(a, a.f, "F（next 用固定原型）")
        elif p == "c":
            cmd_c(a)
        elif p == "d":
            cmd_d(a)
        else:
            raise SystemExit(f"unknown part {p}")


if __name__ == "__main__":
    main()
