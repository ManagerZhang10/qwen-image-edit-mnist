#!/usr/bin/env python3
"""Figures for experiment C (inference sweeps), from scripts/infer.py sweep output.

Usage:
  python scripts/plot_sweep.py --phase 1 --src outputs/sweep [--out figures]
  python scripts/plot_sweep.py --phase 2 --src outputs/sweep
Expects <src>/phase1/ (and <src>/phase2/ for --phase 2) with metrics_phase*.json, eval_set.json and images/.
Writes qi21_c_*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from qie_mnist.plotting import ACCENT, GRAY, LIGHT, MUTED, TEXT, plt, save, set_style  # noqa: F401

_ap = argparse.ArgumentParser()
_ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
_ap.add_argument("--src", default="outputs/sweep", help="dir holding phase1/ and phase2/")
_ap.add_argument("--phase1", default="phase1", help="phase-1 subdir name")
_ap.add_argument("--phase2", default="phase2", help="phase-2 subdir name")
_ap.add_argument("--out", default="figures")
A = _ap.parse_args()
OUT = Path(A.src)
MEDIA = Path(A.out)
TASK_ZH = {"rot90": "顺时针转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}
SERIES = [ACCENT, "#5E9EF0", "#A7C8F5", GRAY, "#1D1D1F"]


def local_stats(phase, name):
    """From the saved 256 px outputs. stroke_blur: share of mid-gray among non-background pixels
    (soft edges / averaged strokes); peak: 99th-percentile brightness (dim = washed-out mean image);
    chroma: mean RGB spread (MNIST is gray, so any colour is an artefact)."""
    d = (OUT / (A.phase1 if phase == 1 else A.phase2)) / "images" / name
    b, m, pk, ch = [], [], [], []
    for p in sorted(d.glob("[0-9][0-9].png")):
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
        g = a.mean(-1)
        m.append(((g > 48) & (g < 208)).sum()); b.append((g >= 208).sum())
        pk.append(np.percentile(g, 99)); ch.append((a.max(-1) - a.min(-1)).mean())
    m, b = np.array(m, float), np.array(b, float)
    return dict(stroke_blur=float(np.mean(m / np.maximum(m + b, 1))), peak=float(np.mean(pk)), chroma=float(np.mean(ch)))


def load(phase):
    m = json.load(open((OUT / (A.phase1 if phase == 1 else A.phase2)) / f"metrics_phase{phase}.json"))
    return m, {r["name"]: r for r in m["results"] if not r.get("skipped")}


def img(phase, name, i, full=False):
    p = (OUT / (A.phase1 if phase == 1 else A.phase2)) / "images" / name / (f"{i:02d}_full.png" if full else f"{i:02d}.png")
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC))


def ref(i, kind="ref"):
    return np.asarray(Image.open(OUT / A.phase1 / "images" / "_ref" / f"{i:02d}_{kind}.png").convert("RGB"))


def pick_rows(meta, per_task=1):
    """one display example per task (first of the two)"""
    ev = json.load(open(OUT / A.phase1 / "eval_set.json"))
    rows, seen = [], {}
    for i in meta["display"]:
        t = ev[i]["task"]
        if seen.get(t, 0) < per_task:
            rows.append((i, t)); seen[t] = seen.get(t, 0) + 1
    return rows


def strip(fig, gs_top, rows, cols, phase):
    """cols: list of (title, name|'_ref'|'_tgt'). Draws a rows×cols image grid into gs_top."""
    sub = gs_top.subgridspec(len(rows), len(cols), wspace=0.04, hspace=0.06)
    for r, (i, task) in enumerate(rows):
        for c, (title, name) in enumerate(cols):
            ax = fig.add_subplot(sub[r, c])
            if name == "_ref":
                im = ref(i)
            elif name == "_tgt":
                im = ref(i, "tgt")
            else:
                im = img(phase, name, i)
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(name not in ("_ref", "_tgt")); s.set_color(LIGHT)
            if r == 0:
                ax.set_title(title, fontsize=17, color=TEXT if name[0] != "_" else MUTED, pad=8)
            if c == 0:
                ax.set_ylabel(TASK_ZH[task], fontsize=16, color=MUTED, rotation=0, ha="right", va="center", labelpad=10)


def bars(ax, labels, vals, title, fmt="{:.0%}", color=ACCENT, ylim=None, highlight=None):
    cs = [color if (highlight is None or k == highlight) else LIGHT for k in range(len(vals))] if highlight is not None else color
    b = ax.bar(labels, vals, color=cs, width=0.6)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=14, color=TEXT)
    ax.set_title(title, fontsize=17, loc="left")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


def per_task_lines(ax, xs, recs, xlabel, logx=False, note=True):
    for k, t in enumerate(TASK_ZH):
        inv = note and t == "invert"
        ax.plot(xs, [r["metrics"][t]["success"] for r in recs], "--o" if inv else "-o", color=SERIES[k], lw=1.6 if inv else 2.2,
                ms=5 if inv else 6, label=TASK_ZH[t] + ("*" if inv else ""))
    ax.plot(xs, [r["success_mean"] for r in recs], "-o", color=TEXT, lw=3, ms=7, label="平均")
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_title("编辑成功率（分类器判定）", fontsize=17, loc="left")
    ax.legend(fontsize=12, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    if note:
        ax.text(1.02, -0.02, "* 模型从未真正反色，\n  这条线是分类器噪声", transform=ax.transAxes, fontsize=11, color=MUTED, va="top")


def fig_steps(meta, R, phase=1, prefix="", names=None, fname="qi21_c_steps.png", ttl=""):
    names = names or [("2 步", "steps2"), ("4 步", "steps4"), ("8 步", "steps8"), ("16 步", "steps16"), ("40 步", "base")]
    names = [(a, b) for a, b in names if b in R]
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.1, right=0.86, top=0.895, bottom=0.07)
    strip(fig, gs[0], rows, [("输入", "_ref")] + names + [("标准答案", "_tgt")], phase)
    bot = gs[1].subgridspec(1, 2, wspace=0.55)
    xs = [R[n]["steps"] for _, n in names]
    per_task_lines(fig.add_subplot(bot[0]), xs, [R[n] for _, n in names], "采样步数", logx=True, note=phase == 1)
    ax = fig.add_subplot(bot[1])
    ls = [local_stats(phase, n) for _, n in names]
    ax.plot(xs, [l["peak"] / 255 for l in ls], "-o", color=ACCENT, lw=3, ms=7)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0.5, 1.05); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("采样步数"); ax.set_title("笔画最亮处亮度（越低越灰暗；点上标每张耗时）", fontsize=17, loc="left")
    for x, l, n in zip(xs, ls, [n for _, n in names]):
        ax.annotate(f"{R[n]['sec_per_img']:.1f}s", (x, l["peak"] / 255), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=12, color=MUTED)
    fig.suptitle(ttl or "只走 2 步 ≈ 从纯噪声一步猜「平均图」：又暗又糊；8 步起基本成形", fontsize=22, y=0.975, x=0.1, ha="left")
    save(fig, MEDIA, fname)


def fig_cfg(meta, R, phase=1, names=None, fname="qi21_c_cfg.png", ttl=""):
    names = names or [("CFG 1（默认）", "base"), ("CFG 2", "cfg2"), ("CFG 4", "cfg4"), ("CFG 7", "cfg7")]
    names = [(a, b) for a, b in names if b in R]
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(17, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.12, right=0.97, top=0.895, bottom=0.05)
    strip(fig, gs[0], rows, [("输入", "_ref")] + names + [("标准答案", "_tgt")], phase)
    bot = gs[1].subgridspec(1, 3, wspace=0.25)
    lab = [a.replace("（默认）", "") for a, _ in names]
    bars(fig.add_subplot(bot[0]), lab, [R[n]["success_mean"] for _, n in names], "平均编辑成功率", ylim=(0, 1.15))
    cv = [local_stats(phase, n)["chroma"] for _, n in names]
    bars(fig.add_subplot(bot[1]), lab, cv, "彩色伪影（RGB 通道差，越低越干净）", fmt="{:.1f}", color=GRAY, ylim=(0, 1.25 * max(cv)))
    bars(fig.add_subplot(bot[2]), lab, [R[n]["sec_per_img"] for _, n in names], "每张耗时", fmt="{:.1f}s", color=GRAY,
         ylim=(0, 1.2 * max(R[n]["sec_per_img"] for _, n in names)))
    fig.suptitle(ttl or "CFG > 1 让旋转更听话（47% → 61%），CFG 7 开始冒色块；每步要算两遍", fontsize=22, y=0.975, x=0.12, ha="left")
    save(fig, MEDIA, fname)


def fig_shift(meta, R):
    names = [("动态≈1.7（默认）", "base"), ("shift 1", "shift1"), ("shift 3", "shift3"), ("shift 6", "shift6")]
    names = [(a, b) for a, b in names if b in R]
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], width_ratios=[1.6, 1], hspace=0.22, wspace=0.12,
                          left=0.1, right=0.97, top=0.895, bottom=0.07)
    strip(fig, gs[0, :], rows, [("输入", "_ref")] + names + [("标准答案", "_tgt")], 1)
    ax = fig.add_subplot(gs[1, 0])
    for k, (a, n) in enumerate(names):
        s = R[n]["sigmas"]
        ax.plot(range(len(s)), s, "-", color=SERIES[k], lw=2.5, label=a)
    ax.set_xlabel("第几步（共 40 步）"); ax.set_ylabel("噪声强度 σ")
    ax.set_title("σ 表：shift 越大，越多步花在高噪声段", fontsize=17, loc="left")
    ax.legend(fontsize=13)
    bars(fig.add_subplot(gs[1, 1]), [a.replace("（默认）", "") for a, _ in names], [R[n]["success_mean"] for _, n in names],
         "平均编辑成功率", ylim=(0, 1.15))
    fig.suptitle("shift 只改 σ 怎么分配：在这个任务上几乎没影响", fontsize=22, y=0.975, x=0.1, ha="left")
    save(fig, MEDIA, "qi21_c_shift.png")


def fig_kv(meta, R):
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.14, right=0.97, top=0.895, bottom=0.05)
    sub = gs[0].subgridspec(len(rows), 4, wspace=0.04, hspace=0.06)
    for r, (i, task) in enumerate(rows):
        a, b = img(1, "base", i, full=True).astype(int), img(1, "kvoff", i, full=True).astype(int)
        d = np.abs(a - b).max(-1)
        for c, (t, im) in enumerate([("输入", ref(i)), ("缓存开", a.astype(np.uint8)), ("缓存关", b.astype(np.uint8)),
                                     ("逐像素差 ×4", np.clip(d * 4, 0, 255).astype(np.uint8))]):
            ax = fig.add_subplot(sub[r, c]); ax.imshow(im, cmap="magma" if c == 3 else None, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(t, fontsize=17)
            if c == 0: ax.set_ylabel(TASK_ZH[task], fontsize=16, color=MUTED, rotation=0, ha="right", va="center", labelpad=10)
    bot = gs[1].subgridspec(1, 2, wspace=0.3)
    k, b = R["kvoff"], R["base"]
    bars(fig.add_subplot(bot[0]), ["缓存开", "缓存关"], [b["sec_per_img"], k["sec_per_img"]], "每张耗时（40 步）", fmt="{:.1f}s",
         ylim=(0, 1.25 * k["sec_per_img"]))
    bars(fig.add_subplot(bot[1]), ["缓存开", "缓存关"], [b["success_mean"], k["success_mean"]],
         f"成功率（像素非逐位相同：平均差 {k['vs_base']['mean_abs_diff']:.1f}/255）", ylim=(0, 1.15))
    sp = k["sec_per_img"] / b["sec_per_img"]
    fig.suptitle(f"条件 token 只算一次：提速 {sp:.1f}×，64 张的对错判定完全一致", fontsize=22, y=0.975, x=0.14, ha="left")
    save(fig, MEDIA, "qi21_c_kvcache.png")


def fig_causal(meta, R):
    names = [("条件用 t=0（默认）", "kvoff"), ("条件用真实 t", "causaloff")]
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(14, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.16, right=0.97, top=0.895, bottom=0.05)
    strip(fig, gs[0], rows, [("输入", "_ref")] + names + [("标准答案", "_tgt")], 1)
    bot = gs[1].subgridspec(1, 2, wspace=0.3)
    lab = ["t=0", "真实 t"]
    bars(fig.add_subplot(bot[0]), lab, [R[n]["success_mean"] for _, n in names], "平均编辑成功率", ylim=(0, 1.15))
    cv = [local_stats(1, n)["chroma"] for _, n in names]
    bars(fig.add_subplot(bot[1]), lab, cv, "彩色伪影（RGB 通道差）", fmt="{:.1f}", color=GRAY, ylim=(0, 1.25 * max(cv)))
    fig.suptitle("关掉 causal_condition：慢了一倍，画面却更干净——和预期相反", fontsize=22, y=0.975, x=0.16, ha="left")
    save(fig, MEDIA, "qi21_c_causal.png")


def fig_res(meta, R):
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(14, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.18, right=0.97, top=0.895, bottom=0.05)
    sub = gs[0].subgridspec(len(rows), 3, wspace=0.04, hspace=0.06)
    for r, (i, task) in enumerate(rows):
        for c, (t, im) in enumerate([("输入", ref(i)), ("512×512", img(1, "base", i, full=True)), ("1024×1024", img(1, "res1024", i, full=True))]):
            ax = fig.add_subplot(sub[r, c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(t, fontsize=17)
            if c == 0: ax.set_ylabel(TASK_ZH[task], fontsize=16, color=MUTED, rotation=0, ha="right", va="center", labelpad=10)
    bot = gs[1].subgridspec(1, 3, wspace=0.35)
    a, b = R["base"], R["res1024"]
    d = meta["display"]
    sa = sum(a["per_sample_success"][str(i)] for i in d); sb = sum(b["per_sample_success"][str(i)] for i in d)
    bars(fig.add_subplot(bot[2]), ["512", "1024"], [sa, sb], f"这 {len(d)} 张里编辑成功", fmt="{:d} 张", color=ACCENT, ylim=(0, len(d) * 1.2))
    bars(fig.add_subplot(bot[0]), ["512", "1024"], [a["seq"]["joint"], b["seq"]["joint"]], "序列长度（token）", fmt="{:,}",
         ylim=(0, 1.25 * b["seq"]["joint"]))
    bars(fig.add_subplot(bot[1]), ["512", "1024"], [a["sec_per_img"], b["sec_per_img"]], "每张耗时", fmt="{:.1f}s", color=GRAY,
         ylim=(0, 1.25 * b["sec_per_img"]))
    fig.suptitle(f"回到模型默认的 1024：更干净，但 token ×{b['seq']['joint'] / a['seq']['joint']:.1f}，耗时 ×{b['sec_per_img'] / a['sec_per_img']:.1f}", fontsize=22, y=0.975, x=0.18, ha="left")
    save(fig, MEDIA, "qi21_c_resolution.png")


def fig_lora(meta, R):
    names = [("LoRA 0", "lora0.0"), ("0.5", "lora0.5"), ("1.0", "lora1.0"), ("1.5", "lora1.5")]
    names = [(a, b) for a, b in names if b in R]
    rows = pick_rows(meta)
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22, left=0.1, right=0.86, top=0.895, bottom=0.07)
    strip(fig, gs[0], rows, [("输入", "_ref")] + names + [("标准答案", "_tgt")], 2)
    bot = gs[1].subgridspec(1, 2, wspace=0.55)
    xs = [R[n]["lora"] for _, n in names]
    ax = fig.add_subplot(bot[0]); per_task_lines(ax, xs, [R[n] for _, n in names], "LoRA 强度", note=False)
    ax.set_xticks(xs)
    ax2 = fig.add_subplot(bot[1])
    ax2.plot(xs, [local_stats(2, n)["chroma"] for _, n in names], "-o", color=ACCENT, lw=3)
    ax2.set_xlabel("LoRA 强度"); ax2.set_title("彩色伪影（RGB 通道差）", fontsize=17, loc="left")
    fig.suptitle("LoRA 是个旋钮：0 就是原模型；0.5 已经学会旋转和反色，彩色伪影消失", fontsize=22, y=0.975, x=0.1, ha="left")
    save(fig, MEDIA, "qi21_c_lora_scale.png")


def main():
    a = A
    set_style()
    if a.phase == 1:
        meta, R = load(1)
        fig_steps(meta, R)
        if "cfg4" in R: fig_cfg(meta, R)
        if "shift1" in R: fig_shift(meta, R)
        if "kvoff" in R: fig_kv(meta, R)
        if "causaloff" in R: fig_causal(meta, R)
        if "res1024" in R: fig_res(meta, R)
    else:
        meta, R = load(2)
        fig_lora(meta, R)
        fig_steps(meta, R, phase=2, names=[("4 步", "lora1.0_steps4"), ("8 步", "lora1.0_steps8"), ("40 步", "lora1.0")],
                  fname="qi21_c_lora_steps.png", ttl="微调后，答案唯一的编辑 4 步就够；答案不唯一的「换数字」少步就糊成平均图")
        fig_cfg(meta, R, phase=2, names=[("CFG 1", "lora1.0"), ("CFG 4", "lora1.0_cfg4")], fname="qi21_c_lora_cfg.png",
                ttl="微调后再加 CFG：77% → 81%，只有「换数字」受益，时间翻倍")


if __name__ == "__main__":
    main()
