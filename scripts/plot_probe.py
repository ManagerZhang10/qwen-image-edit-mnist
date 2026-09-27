#!/usr/bin/env python3
"""Step 5 - figures for experiment D (loss vs sigma), from scripts/probe_loss.py output. CPU.

Deck page -> output file:
  p.10 (Sigma sweep of the x0 guess)          qi21_b_sigma_vis.png     needs --vis (probe_loss.py vis thumbnails + vis_numbers.json)
  p.11 (loss = x0 error x 1/sigma^2)          qi21_b_sigma_decomp.png  needs only results.json (shipped in results/probe/)
Also writes qi21_b_loss_vs_sigma_by_task.png (README figure: deterministic tasks vs next, dashed line = floor D).

Usage:
  python scripts/plot_probe.py                                   # results/probe/: p.11 + the README figure
  python scripts/plot_probe.py --src outputs/probe --vis outputs/probe/vis   # your own outputs, adds p.10
"""
import argparse
import json
import os

import numpy as np

from qie_mnist.plotting import ACCENT, GRAY, LIGHT, MUTED, TEXT, plt, save, set_style  # noqa: F401

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--src", default="results/probe", help="output dir of probe_loss.py measure (results.json)")
_ap.add_argument("--vis", default=None, help="output dir of probe_loss.py vis (PNGs + vis_numbers.json); enables deck p.10")
_ap.add_argument("--out", default="outputs/figures")
_A = _ap.parse_args()
RESULTS = os.path.join(_A.src, "results.json")
VIS = _A.vis
MEDIA = _A.out
VIS_SIGMAS = ["0.1", "0.5", "0.95", "0.99", "1.0"]   # same as the probe_loss.py vis default
VIS_EXTRA = 3                                        # extra noise draws at sigma = 1
TASKS = ["rot90", "rot180", "invert", "next"]
TASK_ZH = {"rot90": "顺时针转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}
TASK_C = {"rot90": ACCENT, "rot180": "#eb6834", "next": "#1baf7a", "invert": "#eda100"}
TASK_M = {"rot90": "o", "rot180": "s", "next": "^", "invert": "D"}


def fig_loss_vs_sigma():
    R = json.load(open(RESULTS))
    set_style()
    sig = R["sigmas"]
    L = R.get("probe_lora") or R["probe_base"]
    D = R["part1"]["D_next_train"]
    y = lambda d, t: [d[t][str(s)]["mean"] for s in sig]  # noqa: E731

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    ax = axes[0]
    for t in ("rot90", "rot180", "invert"):
        ax.plot(sig, y(L, t), color=TASK_C[t], lw=2.6, marker=TASK_M[t], ms=8, label=TASK_ZH[t], zorder=3)
    ax.axhline(0, color=GRAY, lw=1.6, ls="--", zorder=1)
    ax.text(0.12, 0.0008, "D = 0", color=MUTED, fontsize=15, va="bottom")
    ax.set_title("确定性任务：目标由输入唯一确定", loc="left", fontsize=19, color=TEXT)
    ax.legend(loc="upper right", fontsize=14)

    ax = axes[1]
    ax.plot(sig, y(L, "next"), color=TASK_C["next"], lw=2.6, marker=TASK_M["next"], ms=8,
            label=TASK_ZH["next"], zorder=3)
    ax.axhline(D, color=GRAY, lw=1.6, ls="--", zorder=1)
    ax.text(0.12, D + 0.015, f"D = 同一类数字之间的差异 ≈ {D:.2f}", color=MUTED, fontsize=15, va="bottom")
    ax.set_title("不确定性任务：同一输入有多种正确答案", loc="left", fontsize=19, color=TEXT)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=14)
    ax.set_ylim(-0.02, max(y(L, "next")) * 1.12)

    det_top = max(max(y(L, t)) for t in ("rot90", "rot180", "invert"))
    axes[0].set_ylim(-0.002, det_top * 1.15)
    for ax in axes:
        ax.set_xlim(0.05, 1.03)
        ax.set_xlabel("噪声强度 σ（1 = 纯噪声）")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        ax.grid(axis="y", color=LIGHT, lw=0.8, alpha=0.6)
    axes[0].set_ylabel("固定测试集 loss")
    fig.tight_layout()
    save(fig, MEDIA, "qi21_b_loss_vs_sigma_by_task.png")


def fig_decomp():
    """v loss = x0 error x 1/sigma^2 (exact, since x0' - x0 = sigma * (v - v')): top v loss, middle 1/sigma^2, bottom x0 error."""
    set_style()
    R0 = json.load(open(RESULTS))
    R = R0.get("probe_lora") or R0["probe_base"]
    S = [s for s in ["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"] if s in R["next"]]
    sig = np.array([float(s) for s in S])
    fig, (axC, axB, axA) = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True,
                                        gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.34))
    L1, L2 = "#EEF3FB", "#EAF7F1"
    for ax in (axA, axB, axC):
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(LIGHT)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#E4E4EA", lw=1)
        ax.set_yscale("log"); ax.set_xlim(0.03, 1.04); ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    axA.set_xlabel("σ（噪声占比）")
    axC.axvspan(0.03, 0.5, color=L1, zorder=0); axC.axvspan(0.5, 1.04, color=L2, zorder=0)
    axB.set_facecolor(L1); axA.set_facecolor(L2)   # the middle panel explains the left half of the top panel, the bottom panel the right half
    for t in TASKS:
        v = np.array([R[t][s]["mean"] for s in S])
        det = t != "next"
        kw = dict(color=TASK_C[t], lw=2.2 if det else 3.2, marker="o", ms=5 if det else 7, alpha=0.85 if det else 1)
        axA.plot(sig, v * sig ** 2, **kw)
        axC.plot(sig, v, label=TASK_ZH[t], **kw)
    axB.plot(sig, 1 / sig ** 2, color=TEXT, lw=3, marker="o", ms=6)
    for x, y in zip(sig[:5], 1 / sig[:5] ** 2):
        axB.text(x + 0.015, y * 1.15, f"{y:.0f}" if y >= 10 else f"{y:.1f}", fontsize=12.5, color=MUTED)
    axC.set_title("v loss：训练时看到的", loc="left", fontsize=18, color=TEXT, fontweight=600)
    axB.set_title("放大倍数 1/σ²　（左半边的主导项）", loc="left", fontsize=17, color=TEXT)
    axA.set_title("x0 误差：猜终点偏多少　（右半边的主导项）", loc="left", fontsize=17, color=TEXT)
    axC.legend(frameon=False, fontsize=12, loc="lower right", ncol=4, bbox_to_anchor=(1.0, 1.0), handlelength=1.4,
               columnspacing=1.0, borderaxespad=0.2)
    axC.text(0.25, 0.95, "左半边：放大倍数主导，所有任务都在降", transform=axC.transAxes, ha="center", va="top",
             fontsize=13.5, color="#3A5A8C")
    axC.text(0.76, 0.95, "右半边：x0 误差主导，答案不唯一的回升", transform=axC.transAxes, ha="center", va="top",
             fontsize=13.5, color="#137a55")
    axB.text(0.97, 0.85, "σ 越小，放得越大", transform=axB.transAxes, ha="right", fontsize=13, color=MUTED)
    axA.text(0.03, 0.9, "答案唯一：一直很小\n答案不唯一：看不清后猛涨", transform=axA.transAxes, ha="left", va="top",
             fontsize=13, color=MUTED)
    lo = min(axA.get_ylim()[0], axC.get_ylim()[0]); hi = max(axA.get_ylim()[1], axC.get_ylim()[1])
    axA.set_ylim(lo, hi); axC.set_ylim(lo, hi * 8); axB.set_ylim(0.7, 300)
    save(fig, MEDIA, "qi21_b_sigma_decomp.png", facecolor="white", bbox_inches="tight")


def xt_pixel(tag, s):
    """Pixel-space illustration of (1 - sigma) * x0 + sigma * eps (the model's real x_t lives in latent space and decodes to
    colour blotches). Built in memory; nothing is written."""
    from PIL import Image
    x0 = np.asarray(Image.open(os.path.join(VIS, f"{tag}_x0.png")).convert("L"), dtype=np.float32) / 127.5 - 1
    eps = np.random.default_rng(0).standard_normal(x0.shape).astype(np.float32)
    x = (1 - float(s)) * x0 + float(s) * eps
    return np.asarray(Image.fromarray(np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)).convert("RGB"))


def fig_sigma_vis():
    """Per group: reference / ground truth, x_t (pixel illustration) and the one-step guess x0' at each sigma, plus 3 more
    noise draws at sigma = 1."""
    from PIL import Image
    TASK_C2 = {"rot90": ACCENT, "next": "#1baf7a"}
    TITLE = {"rot90": "顺时针转 90°（确定性）", "next": "换成下一个数字（不确定性）"}
    nums = json.load(open(os.path.join(VIS, "vis_numbers.json")))
    R0 = json.load(open(RESULTS))
    avg = R0.get("probe_lora") or R0["probe_base"]
    set_style()
    W, H = 16, 9
    fig = plt.figure(figsize=(W, H))
    iw, ih = 1.4 / W, 1.4 / H
    col0 = 0.085
    xs = [0.262 + k * 0.126 for k in range(5)]
    group_h = 0.475

    def img(x, y, src, edge=None, sc=1.0):
        ax = fig.add_axes([x, y, iw * sc, ih * sc])
        ax.imshow(Image.open(src) if isinstance(src, str) else src)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(edge is not None); s.set_color(edge or "none"); s.set_linewidth(2)

    for g, tag in enumerate(sorted(nums, key=lambda t: 0 if t.startswith("rot90") else 1)):
        r = nums[tag]
        task = r["task"]
        top = 0.985 - g * group_h
        fig.patches.append(plt.Rectangle((0.012, top - 0.032), 0.005, 0.028, transform=fig.transFigure,
                                         color=TASK_C2[task]))
        fig.text(0.024, top - 0.018, TITLE[task], fontsize=19, color=TEXT, va="center")
        fig.text(0.262, top - 0.018, f"指令：{r['prompt']}", fontsize=15, color=MUTED, va="center")
        y1 = top - 0.045 - ih          # x_t row
        y2 = y1 - 0.008 - ih           # x0' row
        img(col0, y1, os.path.join(VIS, f"{tag}_ref.png"))
        img(col0, y2, os.path.join(VIS, f"{tag}_x0.png"), edge=TASK_C2[task])
        fig.text(col0 - 0.008, y1 + ih / 2, "参考图", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(col0 - 0.008, y2 + ih / 2, "正确答案 x0\n（模型看不到）", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(0.250, y1 + ih / 2, "x_t\n（像素示意）", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(0.250, y2 + ih / 2, "x0′\n模型的猜测", ha="right", va="center", fontsize=14, color=TEXT)
        for k, s in enumerate(VIS_SIGMAS):
            img(xs[k], y1, xt_pixel(tag, s))
            img(xs[k], y2, os.path.join(VIS, f"{tag}_x0hat_{s}.png"))
            n = r["sigma"][s]
            cx = xs[k] + iw / 2
            fig.text(cx, y2 - 0.010, f"σ = {s}", ha="center", va="top", fontsize=14, color=TEXT)
            fig.text(cx, y2 - 0.035, f"x0 误差 {n['x0_err']:.4f}", ha="center", va="top", fontsize=13, color=TEXT)
            fig.text(cx, y2 - 0.059, f"v loss {n['v_loss']:.3f}", ha="center", va="top", fontsize=13, color=TEXT)
            fig.text(cx, y2 - 0.083, f"16 张平均 v loss {avg[task][s]['mean']:.3f}", ha="center", va="top",
                     fontsize=12, color=MUTED)
        # sigma = 1: 3 more noise draws, does the guess change?
        ex, sc = 0.895, 0.62
        fig.text(ex + iw * sc / 2, y1 + ih + 0.006, "σ = 1 再换 3 组噪声", ha="center", va="bottom", fontsize=13, color=MUTED)
        step = (2 * ih + 0.008 - 3 * ih * sc) / 2
        for k in range(VIS_EXTRA):
            yy = y1 + ih - (k + 1) * ih * sc - k * step
            img(ex, yy, os.path.join(VIS, f"{tag}_x0hat_1.0_s{k + 1}.png"), sc=sc)
    save(fig, MEDIA, "qi21_b_sigma_vis.png")


if __name__ == "__main__":
    fig_loss_vs_sigma()
    fig_decomp()
    if VIS:
        fig_sigma_vis()
    else:
        print("skip qi21_b_sigma_vis.png (page 10): pass --vis <probe_loss.py vis output>; see deck/media/")
