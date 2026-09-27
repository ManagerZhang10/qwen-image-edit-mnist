#!/usr/bin/env python3
"""Figures for experiment D (loss vs sigma and the sigma = 1 floor), from scripts/probe_loss.py output.

Usage: python scripts/plot_probe.py [--src results/probe] [--vis outputs/probe/vis] [--out figures]
  --src  dir with results.json (probe_loss.py measure); the shipped results/probe/ works out of the box
  --vis  dir with the PNGs + vis_numbers.json from probe_loss.py vis (optional; enables qi21_b_sigma_vis.png)
Writes qi21_b_loss_vs_sigma_by_task.png, qi21_b_sigma_decomp.png (and qi21_b_sigma_vis.png).
"""
import argparse
import json
import os

import numpy as np

from qie_mnist.plotting import ACCENT, GRAY, LIGHT, MUTED, TEXT, plt, save, set_style  # noqa: F401

_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default="results/probe")
_ap.add_argument("--vis", default=None)
_ap.add_argument("--out", default="figures")
_A = _ap.parse_args()
RESULTS = os.path.join(_A.src, "results.json")
VIS = _A.vis
MEDIA = _A.out
SIGMAS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
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
    set_style()
    R0 = json.load(open(RESULTS))
    R = R0.get("probe_lora") or R0["probe_base"]
    S = [s for s in ["0.1", "0.3", "0.5", "0.7", "0.9", "0.95", "0.99"] if s in R["next"]]
    sig = np.array([float(s) for s in S])
    fig = plt.figure(figsize=(18, 6.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.16, 1, 0.16, 1], wspace=0.12)
    axA, axB, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
    for ax in (axA, axB, axC):
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(LIGHT)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#EDEDF0", lw=1)
        ax.set_yscale("log"); ax.set_xlim(0.03, 1.04); ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_xlabel("σ（噪声占比）")
    for t in TASKS:
        v = np.array([R[t][s]["mean"] for s in S])
        det = t != "next"
        kw = dict(color=TASK_C[t], lw=2.2 if det else 3.2, marker="o", ms=5 if det else 7, alpha=0.85 if det else 1)
        axA.plot(sig, v * sig ** 2, label=TASK_ZH[t], **kw)
        axC.plot(sig, v, **kw)
    axB.plot(sig, 1 / sig ** 2, color=TEXT, lw=3, marker="o", ms=6)
    for x, y in zip(sig[:5], 1 / sig[:5] ** 2):
        axB.text(x + 0.03, y * 1.12, f"{y:.0f}" if y >= 10 else f"{y:.1f}", fontsize=12.5, color=MUTED)
    axA.set_title("x0 误差：猜终点偏多少", loc="left", fontsize=17, color=TEXT)
    axB.set_title("放大倍数 1/σ²", loc="left", fontsize=17, color=TEXT)
    axC.set_title("v loss：训练时看到的", loc="left", fontsize=17, color=TEXT)
    axA.legend(frameon=False, fontsize=13, loc="upper left")
    axA.text(0.98, 0.04, "答案唯一：一直很小\n答案不唯一：看不清目标后猛涨", transform=axA.transAxes, ha="right", va="bottom", fontsize=13, color=MUTED)
    axB.text(0.98, 0.9, "σ 越小，放得越大", transform=axB.transAxes, ha="right", fontsize=13, color=MUTED)
    axC.axvspan(0.03, 0.5, color="#F4F6FA", zorder=0); axC.axvspan(0.5, 1.04, color="#F1FAF6", zorder=0)
    axC.text(0.27, 0.97, "阶段一：放大倍数主导\n所有任务都在降", transform=axC.get_xaxis_transform(), ha="center", va="top", fontsize=12.5, color=MUTED)
    axC.text(0.77, 0.97, "阶段二：x0 误差主导\n不唯一的任务明显回升", transform=axC.get_xaxis_transform(), ha="center", va="top", fontsize=12.5, color="#1baf7a")
    lo = min(axA.get_ylim()[0], axC.get_ylim()[0]); hi = max(axA.get_ylim()[1], axC.get_ylim()[1])
    axA.set_ylim(lo, hi); axC.set_ylim(lo, hi * 4)
    for g, sym in ((gs[1], "×"), (gs[3], "=")):
        a = fig.add_subplot(g); a.axis("off")
        a.text(0.5, 0.5, sym, ha="center", va="center", fontsize=46, color=GRAY)
    save(fig, MEDIA, "qi21_b_sigma_decomp.png", facecolor="white", bbox_inches="tight")


def xt_pixel(tag, s):
    """Pixel-space illustration of (1-σ)·x0 + σ·ε. The model's real x_t lives in latent space and decodes to colour blotches."""
    import numpy as np
    from PIL import Image
    out = os.path.join(VIS, f"{tag}_xtpix_{s}.png")
    x0 = np.asarray(Image.open(os.path.join(VIS, f"{tag}_x0.png")).convert("L"), dtype=np.float32) / 127.5 - 1
    eps = np.random.default_rng(0).standard_normal(x0.shape).astype(np.float32)
    x = (1 - float(s)) * x0 + float(s) * eps
    Image.fromarray(np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)).convert("RGB").save(out)
    return out


def fig_sigma_vis():
    from PIL import Image
    TASK_C = {"rot90": ACCENT, "next": "#1baf7a"}
    TITLE = {"rot90": "顺时针转 90°（确定性）", "next": "换成下一个数字（不确定性）"}
    nums = json.load(open(os.path.join(VIS, "vis_numbers.json")))
    R0 = json.load(open(RESULTS))
    avg = R0.get("probe_lora") or R0["probe_base"]
    set_style()
    W, H = 16, 9
    fig = plt.figure(figsize=(W, H))
    iw, ih = 1.4 / W, 1.4 / H
    col0 = 0.085
    xs = [0.285 + k * 0.143 for k in range(5)]
    group_h = 0.475

    def img(x, y, path, edge=None):
        ax = fig.add_axes([x, y, iw, ih])
        ax.imshow(Image.open(path))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(edge is not None); s.set_color(edge or "none"); s.set_linewidth(2)

    for g, tag in enumerate(sorted(nums, key=lambda t: 0 if t.startswith("rot90") else 1)):
        r = nums[tag]
        task = r["task"]
        top = 0.985 - g * group_h
        fig.patches.append(plt.Rectangle((0.012, top - 0.032), 0.005, 0.028, transform=fig.transFigure,
                                         color=TASK_C[task]))
        fig.text(0.024, top - 0.018, TITLE[task], fontsize=19, color=TEXT, va="center")
        fig.text(0.285, top - 0.018, f"指令：{r['prompt']}", fontsize=15, color=MUTED, va="center")
        y1 = top - 0.045 - ih          # x_t row
        y2 = y1 - 0.008 - ih           # x0' row
        img(col0, y1, os.path.join(VIS, f"{tag}_ref.png"))
        img(col0, y2, os.path.join(VIS, f"{tag}_x0.png"), edge=TASK_C[task])
        fig.text(col0 - 0.008, y1 + ih / 2, "参考图", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(col0 - 0.008, y2 + ih / 2, "正确答案\nx0", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(0.268, y1 + ih / 2, "目标加噪\n（像素示意）", ha="right", va="center", fontsize=14, color=TEXT)
        fig.text(0.268, y2 + ih / 2, "模型对 x0\n的猜测", ha="right", va="center", fontsize=14, color=TEXT)
        for k, s in enumerate(SIGMAS):
            img(xs[k], y1, xt_pixel(tag, s))
            img(xs[k], y2, os.path.join(VIS, f"{tag}_x0hat_{s}.png"))
            n = r["sigma"][s]
            cx = xs[k] + iw / 2
            inv = 1 / float(s) ** 2
            fig.text(cx, y2 - 0.010, f"σ = {s}  ·  1/σ² = {inv:.0f}" if inv >= 10 else f"σ = {s}  ·  1/σ² = {inv:.1f}",
                     ha="center", va="top", fontsize=14, color=TEXT)
            fig.text(cx, y2 - 0.035, f"x0 误差 {n['x0_err']:.4f}", ha="center", va="top", fontsize=13, color=TEXT)
            fig.text(cx, y2 - 0.059, f"v loss {n['v_loss']:.3f}", ha="center", va="top", fontsize=13, color=TEXT)
            fig.text(cx, y2 - 0.083, f"16 张平均 v loss {avg[task][s]['mean']:.3f}", ha="center", va="top",
                     fontsize=12, color=MUTED)
    save(fig, MEDIA, "qi21_b_sigma_vis.png")


if __name__ == "__main__":
    fig_loss_vs_sigma()
    fig_decomp()
    if VIS:
        fig_sigma_vis()
