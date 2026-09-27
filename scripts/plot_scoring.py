#!/usr/bin/env python3
"""Step 5 - figure "how success is scored", from real outputs of the phase-3 sweep (LoRA 1.0, 40 steps, defaults). CPU.

Deck page -> output file: p.12 (How success is scored)  qi21_c_scoring.png

Per output: shrink to 28x28 gray -> undo the edit (rotate back / invert back; "next" is left as is) -> MNIST
classifier -> compare with the wanted digit. Exactly the steps of qie_mnist.evaluate.evaluate (from_pil, undo, classify).
Needs your own `infer.py sweep --phase 3` output (256 px images under images/lora1.0/); the repo ships no images,
the finished figure is in deck/media/.

Usage: python scripts/plot_scoring.py --src outputs/sweep/phase3 [--out outputs/figures] [--mnist-root data]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from qie_mnist import data as D
from qie_mnist import evaluate as E
from qie_mnist.plotting import ACCENT, LIGHT, MUTED, TEXT, plt, save, set_style

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, help="output dir of infer.py sweep --phase 3 (eval_set.json, images/lora1.0/)")
ap.add_argument("--config", default="lora1.0", help="which config's outputs to use (default lora1.0)")
ap.add_argument("--out", default="outputs/figures")
ap.add_argument("--mnist-root", default=None)
A = ap.parse_args()
SRC = Path(A.src)
# rows: test index and how the edit is undone (one per edit, two for next: one right, one wrong)
ROWS = [(8, "逆时针转回 90°"), (0, "再转 180°"), (5, "再反色一次"), (14, "不做逆变换"), (26, "不做逆变换")]
TASK_ZH = {"rot90": "顺时针转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}
RED = "#D0342C"


def main():
    set_style()
    samples = D.make_pairs("test", 16, root=A.mnist_root)
    ev = json.load(open(SRC / "eval_set.json"))
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(len(ROWS), 6, width_ratios=[1, 1, 0.55, 1, 1.1, 1.3], hspace=0.3, wspace=0.12,
                          left=0.13, right=0.99, top=0.85, bottom=0.02)
    heads = ["参考图", "模型输出", "", "缩回 28×28\n并做逆变换", "分类器认出", "期望数字 → 判定"]
    for r, (i, how) in enumerate(ROWS):
        s = samples[i]; assert s["task"] == ev[i]["task"]
        out = Image.open(SRC / "images" / A.config / f"{i:02d}.png")
        u = E.undo(s["task"], D.from_pil(out))
        p = int(E.classify([u])[0]); ok = p == s["want_label"]
        tiles = [np.asarray(s["ref"].convert("RGB")), np.asarray(out.convert("RGB")), None, u]
        for c in range(6):
            ax = fig.add_subplot(gs[r, c]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and heads[c]:
                ax.set_title(heads[c], fontsize=17, color=TEXT, pad=10)
            if c in (0, 1, 3):
                im = tiles[c]
                ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest" if c == 3 else "bilinear")
                for sp in ax.spines.values(): sp.set_color(LIGHT)
                if c == 0:
                    lab = TASK_ZH[s["task"]] + (f"\n{s['src_label']} → {s['want_label']}" if s["task"] == "next" else "")
                    ax.set_ylabel(lab, fontsize=16, color=TEXT, rotation=0, ha="right", va="center", labelpad=14)
                continue
            ax.axis("off")
            if c == 2:
                ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
                ax.text(0.5, 0.62, how, ha="center", va="bottom", fontsize=13, color=MUTED, transform=ax.transAxes)
            elif c == 4:
                ax.text(0.5, 0.5, str(p), ha="center", va="center", fontsize=54, color=TEXT, fontweight=600, transform=ax.transAxes)
            else:
                ax.text(0.28, 0.5, str(s["want_label"]), ha="center", va="center", fontsize=40, color=MUTED, transform=ax.transAxes)
                ax.text(0.72, 0.5, "✓" if ok else "✗", ha="center", va="center", fontsize=48,
                        color=ACCENT if ok else RED, fontweight=700, transform=ax.transAxes)
                if not ok and p == s["src_label"]:
                    ax.text(0.5, 0.08, "照抄了原数字", ha="center", fontsize=13, color=RED, transform=ax.transAxes)
    fig.suptitle("先把输出变回「黑底白字、正立」，再让分类器认数字；认出的数字等于期望数字才算成功",
                 fontsize=20, y=0.985, x=0.13, ha="left")
    save(fig, A.out, "qi21_c_scoring.png")


if __name__ == "__main__":
    main()
