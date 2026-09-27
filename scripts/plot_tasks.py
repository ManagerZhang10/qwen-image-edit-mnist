#!/usr/bin/env python3
"""Step 5 - figures of the task examples: reference -> target for the four edits. CPU, needs no experiment output.

Deck page -> output file:
  p.1 (Cover)            practice_strip.png (one row, four pairs)
  p.2 (Four edit tasks)  qi21_tasks.png (4 rows with the fixed instructions)

Usage: python scripts/plot_tasks.py [--out outputs/figures] [--mnist-root data]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qie_mnist.data import TASK_LIST, TASKS, make_pairs  # noqa: E402
from qie_mnist.plotting import cjk_fonts, save  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/figures")
ap.add_argument("--mnist-root", default=None)
A = ap.parse_args()
plt.rcParams["font.family"] = ["Hiragino Sans GB"] + cjk_fonts() + ["DejaVu Sans"]  # the deck figure uses Hiragino Sans GB
S = make_pairs("test", 30, seed=3, root=A.mnist_root)


def fig_tasks():
    fig, axes = plt.subplots(4, 8, figsize=(15, 7.6), gridspec_kw=dict(width_ratios=[1, .4, 1, .3, 1, .4, 1, 2.1]))
    for r, task in enumerate(TASK_LIST):
        ex = [s for s in S if s["task"] == task][:3]
        cols = [(0, 2), (4, 6)]
        for (ca, cb), s in zip(cols, ex):
            for c, img in ((ca, s["ref28"]), (cb, s["tgt28"])):
                axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255)
        for c in (1, 5):
            axes[r, c].text(.5, .5, "→", ha="center", va="center", fontsize=26, color="#8E8E93", transform=axes[r, c].transAxes)
        axes[r, 7].text(0.08, .5, "「" + TASKS[task] + "」", ha="left", va="center", fontsize=16, color="#1D1D1F", transform=axes[r, 7].transAxes)
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    for r in range(4):
        for c in (0, 2, 4, 6):
            for sp in axes[r, c].spines.values(): sp.set_visible(True); sp.set_color("#C7C7CC")
    for c, t in ((0, "参考图"), (2, "目标图"), (4, "参考图"), (6, "目标图"), (7, "指令")):
        axes[0, c].set_title(t, fontsize=15, color="#6E6E73")
    plt.subplots_adjust(wspace=.06, hspace=.12, left=.01, right=.99, top=.94, bottom=.01)
    save(fig, A.out, "qi21_tasks.png", facecolor="white")


def fig_strip():
    LABEL = {"rot90": "顺时针旋转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}
    # per group: reference | arrow | target | gap
    W = [1, .38, 1, .42] * 4
    fig, axes = plt.subplots(1, len(W), figsize=(15, 2.5), gridspec_kw=dict(width_ratios=W))
    for g, task in enumerate(TASK_LIST):
        s = [x for x in S if x["task"] == task][0]
        a, arr, b = axes[4 * g], axes[4 * g + 1], axes[4 * g + 2]
        for ax, img in ((a, s["ref28"]), (b, s["tgt28"])):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#C7C7CC")
        arr.text(.5, .5, "→", ha="center", va="center", fontsize=30, color="#8E8E93", transform=arr.transAxes)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        if not ax.images: [sp.set_visible(False) for sp in ax.spines.values()]
    plt.subplots_adjust(wspace=.05, left=.01, right=1.0, top=.98, bottom=.2)
    fig.canvas.draw()
    for g, task in enumerate(TASK_LIST):  # label centred under the pair, from the actual image positions (not the axes slots)
        pa, pb = axes[4 * g].get_position(), axes[4 * g + 2].get_position()
        fig.text((pa.x0 + pb.x1) / 2, pa.y0 - .14, LABEL[task], ha="center", va="center", fontsize=19, color="#1D1D1F")
    save(fig, A.out, "practice_strip.png", dpi=200, facecolor="white")


if __name__ == "__main__":
    fig_tasks()
    fig_strip()
