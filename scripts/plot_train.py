#!/usr/bin/env python3
"""Step 5 - figures for experiment B (training), from the hook_logs written by scripts/train_lora.sh. CPU.

Deck page -> output file:
  p.8  (Training loss)                qi21_b_loss.png           needs train.jsonl, probe.jsonl
  p.9  (Loss by sigma)                qi21_b_probe_by_task.png  needs probe.jsonl
  p.13 (Success rate)                 qi21_b_success.png        needs val.jsonl, probe.jsonl
  p.14 (Before vs after training)     qi21_b_samples.png        needs meta.json and the val/ images (own runs only)
  p.15 (Learnable vs not learnable)   qi21_b_task_order.png     needs val.jsonl, probe.jsonl
Also writes qi21_b_loss_by_sigma.png (probe loss by sigma, not in the deck). The shipped results/train/ draws
everything except p.14.

Usage:
  python scripts/plot_train.py                                        # from results/train/
  python scripts/plot_train.py --src outputs/lora_b/hook_logs         # your own run (adds p.14)
  python scripts/plot_train.py --src results/train_e_shift5 --out outputs/figures/e   # same figures for E / F
"""
import argparse
import json
import os

import numpy as np

from qie_mnist.plotting import ACCENT, GRAY, LIGHT, MUTED, TEXT, set_style  # noqa: F401
from qie_mnist.plotting import save as _save
import matplotlib  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--src", default="results/train", help="hook_logs dir (train.jsonl, probe.jsonl, val.jsonl, meta.json, val/)")
_ap.add_argument("--out", default="outputs/figures")
_ARGS = _ap.parse_args()
SRC = _ARGS.src
MEDIA = _ARGS.out
TASKS = ["rot90", "rot180", "next", "invert"]
TASK_ZH = {"rot90": "顺时针转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}
TASK_C = {"rot90": ACCENT, "rot180": "#eb6834", "next": "#1baf7a", "invert": "#eda100"}
TASK_M = {"rot90": "o", "rot180": "s", "next": "^", "invert": "D"}
SIGMA_C = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#104281"]  # one hue, light -> dark = small -> large sigma


def jl(name):
    # Deduplicate by step, keeping the last record: a Spot restart replays steps after the resumed checkpoint.
    with open(os.path.join(SRC, name)) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    by = {r["step"]: r for r in recs}
    return [by[k] for k in sorted(by)]


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LIGHT)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#EDEDF0", lw=1)
    ax.set_axisbelow(True)


def save(fig, name):
    _save(fig, MEDIA, name, facecolor="white", bbox_inches="tight")


def ema(x, a=0.98):
    out, m = [], x[0]
    for v in x:
        m = a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


def fig_loss(train, probe, title):
    st = np.array([r["step"] for r in train])
    lo = np.array([r["loss"] for r in train])
    sg = np.array([r["sigma"][0] for r in train])
    ext = (sg < 0.05) | (sg > 0.95)
    fig, ax = plt.subplots(figsize=(12, 6.4))
    style_ax(ax)
    ax.plot(st, lo, color=LIGHT, lw=0.7, label="每一步的 loss（原始）", zorder=1)
    ax.scatter(st[ext], lo[ext], s=14, color="#eb6834", label="σ < 0.05 或 σ > 0.95 的步", zorder=2)
    e = ema(lo)
    ax.plot(st, e, color=ACCENT, lw=2.5, label="滑动平均（EMA 0.98）", zorder=3)
    ps = [r["step"] for r in probe]
    pm = [r["mean"] for r in probe]
    ax.plot(ps, pm, color=TEXT, lw=2, ls="--", label="固定测试集 loss（σ=0.1…0.9 平均）", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("MSE loss（对数坐标）")
    ax.legend(frameon=False, loc="upper right", fontsize=14, ncol=2)
    ax.set_ylim(top=lo.max() * 6)
    ax.set_title(title, loc="left", color=TEXT)
    save(fig, "qi21_b_loss.png")
    return e


def fig_sigma(probe, train):
    sig = [str(s) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    style_ax(ax)
    ps = [r["step"] for r in probe]
    for c, s in zip(SIGMA_C, sig):
        y = [r["by_sigma"][s] for r in probe]
        ax.plot(ps, y, color=c, lw=2.2, label=f"σ = {s}")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("固定测试集上的 loss")
    ax.set_title("σ 接近 0 时 loss 最高，中间的 σ 最低", loc="left", color=TEXT)
    ax.legend(frameon=False, ncol=5, fontsize=13, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax = axes[1]
    style_ax(ax)
    first, last = probe[0]["by_sigma"], probe[-1]["by_sigma"]
    drop = [100 * (first[s] - last[s]) / first[s] for s in sig]
    bars = ax.bar(range(5), drop, color=SIGMA_C, width=0.7, edgecolor="white", linewidth=2)
    for b, d in zip(bars, drop):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.4 if d >= 0 else -1.5), f"{d:.0f}%",
                ha="center", color=TEXT, fontsize=14)
    ax.set_xticks(range(5), [f"σ={s}" for s in sig])
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_ylabel("训练前后 loss 下降比例")
    ax.set_title(f"第 0 步 → 第 {probe[-1]['step']} 步：大 σ 降得最多", loc="left", color=TEXT)
    fig.tight_layout()
    save(fig, "qi21_b_loss_by_sigma.png")
    return dict(zip(sig, drop))


def fig_success(val, probe):
    st = [r["step"] for r in val]
    fig, ax = plt.subplots(figsize=(13, 6.6))
    style_ax(ax)
    ax.axvspan(200, 600, color="#F2F2F7", zorder=0)
    ax.text(400, 111, "成功率跳升：第 200–600 步", ha="center", va="bottom", fontsize=14, color=MUTED)
    for t in TASKS:
        y = [100 * r["metrics"][t]["success"] for r in val]
        ax.plot(st, y, color=TASK_C[t], lw=2.4, marker=TASK_M[t], ms=8, label=TASK_ZH[t], zorder=3)
    ax.set_ylim(-3, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("训练步数")
    ax.set_ylabel("编辑成功率（%，实线）")
    # right axis: held-out probe loss
    ax2 = ax.twinx()
    ps = np.array([r["step"] for r in probe]); pm = np.array([r["mean"] for r in probe])
    ax2.plot(ps, pm, color=TEXT, lw=2.2, ls="--", label="测试集 loss（右轴）", zorder=2)
    ax2.set_ylabel("固定测试集 loss（虚线，σ = 0.1…0.9 平均）", color=TEXT)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    m0, mf = pm[0], pm[-1]
    k = int(np.where(ps == 200)[0][0]); frac = (m0 - pm[k]) / (m0 - mf)
    ax2.annotate(f"第 200 步：loss 已完成\n全程降幅的 {100 * frac:.0f}%", xy=(200, pm[k]), xytext=(1000, m0 * 0.78),
                 fontsize=14, color=TEXT, arrowprops=dict(arrowstyle="->", color=TEXT))
    ax2.set_ylim(mf * 0.8, m0 * 1.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.42), fontsize=13)
    ax.set_title("loss 前 200 步就降完大半，成功率在 200–600 步才跳上去", loc="left", color=TEXT, pad=26)
    ax.text(0, -0.2, "每个任务 16 张测试图，40 步采样、cfg 1.0；第 0 步 = 未训练的底模", transform=ax.transAxes,
            color=MUTED, fontsize=13)
    save(fig, "qi21_b_success.png")


def fig_grid(val, meta, steps):
    grid = meta["grid"]
    vdir = os.path.join(SRC, "val")
    cols = ["参考图", "目标"] + [f"第 {s} 步" for s in steps]
    pred = {r["step"]: r["pred"] for r in val}
    # grouped by task: two rows per group, a blank row between groups, task name once per group
    groups = []
    for i in grid:
        t = meta["samples"][i]["task"]
        if not groups or groups[-1][0] != t:
            groups.append((t, []))
        groups[-1][1].append(i)
    hr, rows = [], []
    for g, (t, ids) in enumerate(groups):
        if g:
            hr.append(0.28); rows.append(None)
        for i in ids:
            hr.append(1); rows.append(i)
    fig = plt.figure(figsize=(1.9 * len(cols) + 1.2, 1.95 * sum(hr)))
    gs = fig.add_gridspec(len(rows), len(cols), height_ratios=hr, wspace=0.05, hspace=0.06, left=0.17, right=0.99, top=0.95, bottom=0.01)
    first_row = True
    for r, i in enumerate(rows):
        if i is None:
            continue
        s = meta["samples"][i]
        ims = [Image.open(os.path.join(vdir, f"ref_{i:02d}.png")), Image.open(os.path.join(vdir, f"tgt_{i:02d}.png"))]
        ims += [Image.open(os.path.join(vdir, f"step_{st:05d}", f"{i:02d}.png")) for st in steps]
        for c, im in enumerate(ims):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c >= 2:
                ok = pred[steps[c - 2]][i] == s["want_label"]
                ax.text(0.95, 0.06, "✓" if ok else "✗", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=15, color=ACCENT if ok else "#eb6834", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            if first_row:
                ax.set_title(cols[c], fontsize=15, color=TEXT)
        first_row = False
    # one task name per group, vertically centred on its two rows, with a coloured bar
    for t, ids in groups:
        rs = [rows.index(i) for i in ids]
        top = fig.add_subplot(gs[rs[0], 0]).get_position(); bot = fig.add_subplot(gs[rs[-1], 0]).get_position()
        for ax in fig.axes[-2:]:
            ax.remove()
        y0, y1 = bot.y0, top.y1
        fig.patches.append(plt.Rectangle((0.155, y0), 0.006, y1 - y0, transform=fig.transFigure, color=TASK_C[t]))
        fig.text(0.145, (y0 + y1) / 2, TASK_ZH[t], ha="right", va="center", fontsize=15, color=TEXT)
    save(fig, "qi21_b_samples.png")


def fig_task_order(probe, val):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    ax = axes[0]
    style_ax(ax)
    ps = [r["step"] for r in probe]
    for t in TASKS:
        y = np.array([np.mean(list(r["by_task"][t].values())) for r in probe])
        ax.plot(ps, 100 * y / y[0], color=TASK_C[t], lw=2.4, label=TASK_ZH[t])
    ax.set_xlabel("训练步数")
    ax.set_ylabel("测试 loss（第 0 步 = 100）")
    ax.set_title("测试 loss：学会的任务降得多", loc="left", color=TEXT)
    ax.legend(frameon=False, fontsize=13)
    ax = axes[1]
    style_ax(ax)
    first = {}
    for t in TASKS:
        hit = [r["step"] for r in val if r["metrics"][t]["success"] >= 0.9]
        first[t] = hit[0] if hit else None
    order = sorted(TASKS, key=lambda t: (first[t] is None, first[t] or 0))
    mx = max(r["step"] for r in val)
    for k, t in enumerate(order):
        v = first[t]
        ax.barh(k, v if v is not None else mx, color=TASK_C[t] if v is not None else "#F0F0F2",
                height=0.6, edgecolor="white", linewidth=2)
        ax.text((v if v is not None else mx) + mx * 0.01, k,
                (f"{v} 步" if v else "底模已达到") if v is not None else "训练结束仍未达到", va="center", color=TEXT, fontsize=14)
    ax.set_yticks(range(len(order)), [TASK_ZH[t] for t in order])
    ax.invert_yaxis()
    ax.set_xlim(0, mx * 1.3)
    ax.set_xlabel("成功率首次达到 90% 的步数")
    ax.set_title("谁先学会（成功率 ≥ 90%）", loc="left", color=TEXT)
    ax.grid(False)
    fig.tight_layout()
    save(fig, "qi21_b_task_order.png")
    return first


def fig_dims(train, probe):
    """Loss mixes three factors: sigma, task, training step. Split them."""
    sg = [r["sigma"][0] for r in train]
    st = np.array([r["step"] for r in train]); lo = np.array([r["loss"] for r in train]); sg = np.array(sg)
    SIG = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    X = np.log(np.array([[[r["by_task"][t][g] for g in SIG] for t in TASKS] for r in probe]))  # step, task, sigma
    g = X.mean(); tot = ((X - g) ** 2).sum()
    e_step = X.mean(axis=(1, 2), keepdims=True) - g
    e_task = X.mean(axis=(0, 2), keepdims=True) - g
    e_sig = X.mean(axis=(0, 1), keepdims=True) - g
    e_ts = X.mean(axis=0, keepdims=True) - e_task - e_sig - g
    share = lambda e: float((np.broadcast_to(e, X.shape) ** 2).sum() / tot)
    parts = [("σ（噪声强度）", share(e_sig), ACCENT), ("任务", share(e_task), "#1baf7a"),
             ("交互", share(e_ts), "#7fcfae"), ("训练步数", share(e_step), GRAY)]
    parts.append(("其余", 1 - sum(p[1] for p in parts), LIGHT))
    lg = np.log(lo); bins = np.linspace(0, 1, 41); b = np.digitize(sg, bins)
    m = np.array([lg[b == k].mean() for k in b]); r2 = 1 - ((lg - m) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.28)
    # A: raw training loss vs sigma, colored by step
    ax = fig.add_subplot(gs[0, :2]); style_ax(ax)
    ax.scatter(sg, lo, color="#C9CCD3", s=8, lw=0, label="每一步（3000 个点）")
    for lo_s, hi_s, c, lab in [(0, 300, GRAY, "第 1–300 步"), (2700, 3000, ACCENT, "第 2701–3000 步")]:
        k = (st > lo_s) & (st <= hi_s)
        cen = (bins[:-1] + bins[1:]) / 2
        med = [np.median(lo[k & (sg >= a) & (sg < a + 0.025)]) if (k & (sg >= a) & (sg < a + 0.025)).sum() else np.nan for a in bins[:-1]]
        ok = ~np.isnan(med)
        ax.plot(cen[ok], np.array(med)[ok], color=c, lw=2.4, label=f"{lab} 中位数")
    ax.set_yscale("log"); ax.set_xlabel("这一步抽到的 σ"); ax.set_ylabel("单步训练 loss（对数）")
    ax.set_title(f"训练曲线为什么乱跳：只看 σ 就解释了 {r2:.0%} 的起伏", loc="left", color=TEXT, fontsize=16)
    ax.legend(frameon=False, fontsize=13, loc="upper center")
    # B: variance decomposition on the probe grid
    ax = fig.add_subplot(gs[0, 2:]); style_ax(ax); ax.grid(False)
    left = 0
    for n, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=2)
        if v > 0.06:
            ax.text(left + v / 2, 0, f"{n}\n{v:.0%}", ha="center", va="center", fontsize=13,
                    color="white" if c in (ACCENT, "#1baf7a", GRAY) else TEXT)
        left += v
    ax.text(0.985, -0.42, f"训练步数 {share(e_step):.0%}", ha="right", va="top", fontsize=12.5, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.5); ax.set_yticks([]); ax.set_xticks([])
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    ax.set_title("固定测试集 log loss 的起伏，各因素各占多少", loc="left", color=TEXT, fontsize=16)
    ax.text(0, -0.62, "4 任务 × 5 个 σ × 61 次测量；按主效应拆方差。交互 = 不同任务的 σ 曲线形状不一样", fontsize=12.5, color=MUTED, va="top")
    # C: per task, loss vs step with one line per sigma
    ps = [r["step"] for r in probe]
    lo_all, hi_all = np.exp(X.min()) * 0.8, np.exp(X.max()) * 1.2
    for j, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[1, j]); style_ax(ax)
        for i, (c, s_) in enumerate(zip(SIGMA_C, SIG)):
            ax.plot(ps, np.exp(X[:, j, i]), color=c, lw=2, label=f"σ={s_}")
        ax.set_yscale("log"); ax.set_ylim(lo_all, hi_all)
        ax.set_title(TASK_ZH[t], loc="left", color=TASK_C[t], fontsize=15, fontweight="bold")
        ax.set_xlabel("训练步数", fontsize=12)
        if j == 0:
            ax.set_ylabel("测试 loss（对数）", fontsize=12)
        else:
            ax.set_yticklabels([])
        if j == 3:
            ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=1)
    save(fig, "qi21_b_loss_dims.png")


def fig_train_by_sigma(train):
    """Split the raw training curve by the sigma each step drew: per band, median loss in 250-step windows."""
    st = np.array([r["step"] for r in train]); lo = np.array([r["loss"] for r in train])
    sg = np.array([r["sigma"][0] for r in train])
    bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.8), (0.8, 0.9), (0.9, 1.0001)]
    labs = ["σ 0–0.1", "σ 0.1–0.3", "σ 0.3–0.8", "σ 0.8–0.9", "σ 0.9–1"]
    cols = ["#eb6834", "#86b6ef", "#5598e7", "#104281", "#b0126a"]
    edges = np.arange(0, 3001, 250)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 5, figsize=(19, 6.2))
    for ax, (a, b), lab, c in zip(axes, bands, labs, cols):
        style_ax(ax)
        k = (sg >= a) & (sg < b)
        ax.scatter(st[k], lo[k], s=6, color=c, alpha=0.18, lw=0)
        med = np.array([np.median(lo[k & (st > e0) & (st <= e1)]) for e0, e1 in zip(edges[:-1], edges[1:])])
        ax.plot(mid, med, "-o", color=c, lw=2.6, ms=5)
        drop = 100 * (1 - med[-1] / med[0])
        ax.set_yscale("log")
        lo_k, hi_k = np.percentile(lo[k], [3, 97])
        ax.set_ylim(lo_k / 1.6, hi_k * 2.2)
        ax.set_title(f"{lab}（{k.sum()} 步）", loc="left", color=c, fontsize=15, fontweight="bold")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.text(0.97, 0.95, f"{med[0]:.3f} → {med[-1]:.3f}\n" + (f"降 {drop:.0f}%" if drop > 5 else "基本不降"), transform=ax.transAxes, ha="right", va="top",
                fontsize=14, color=TEXT, fontweight="bold" if drop > 50 else "normal")
        ax.set_xticks([0, 1000, 2000, 3000]); ax.set_xlabel("训练步数", fontsize=12.5)
    axes[0].set_ylabel("单步训练 loss（对数，每图刻度不同）")
    fig.tight_layout(w_pad=1.0)
    save(fig, "qi21_b_train_by_sigma.png")

def fig_probe_by_task(probe):
    SIG = ["0.1", "0.5", "0.9"]
    SC = {"0.1": "#eb6834", "0.5": "#8E8E93", "0.9": "#0071E3"}
    ps = [r["step"] for r in probe]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.4), sharey=True)
    for t, ax in zip(TASKS, axes):
        style_ax(ax)
        ends = {}
        for s_ in SIG:
            y = [r["by_task"][t][s_] for r in probe]
            ax.plot(ps, y, color=SC[s_], lw=2.8, label=f"σ = {s_}")
            ends[s_] = y[-1]
        if abs(np.log(ends["0.1"] / ends["0.9"])) < 0.25:   # overlapping ends: push labels apart
            hi, lo_ = ("0.1", "0.9") if ends["0.1"] >= ends["0.9"] else ("0.9", "0.1")
            ends[hi] *= 1.15; ends[lo_] /= 1.15
        for s_, y1 in ends.items():
            ax.text(ps[-1] + 60, y1, s_, color=SC[s_], fontsize=12.5, va="center", fontweight="bold")
        ax.set_yscale("log")
        ax.set_title(TASK_ZH[t] + ("（答案不唯一）" if t == "next" else ""), loc="left", color=TASK_C[t], fontsize=16, fontweight="bold")
        ax.set_xlabel("训练步数", fontsize=13)
        ax.set_xticks([0, 1000, 2000, 3000]); ax.set_xlim(-60, 3450)
    axes[0].set_ylabel("固定测试集 loss（对数）")
    axes[0].legend(frameon=False, fontsize=13, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "qi21_b_probe_by_task.png")

def fig_train_dist(train):
    """Distribution of per-step training loss: one panel per sigma band, one violin per step window."""
    st = np.array([r["step"] for r in train]); lo = np.array([r["loss"] for r in train])
    sg = np.array([r["sigma"][0] for r in train])
    bands = [(0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 0.95), (0.95, 1.0001)]
    labs = ["σ < 0.05", "σ 0.05–0.3", "σ 0.3–0.7", "σ 0.7–0.95", "σ > 0.95"]
    cols = ["#eb6834", "#86b6ef", "#5598e7", "#104281", "#b0126a"]
    wins = [(0, 300), (300, 1000), (1000, 2000), (2000, 3000)]
    wl = ["前 300", "~1000", "~2000", "~3000"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 6.6), sharey=True)
    for ax, (a, b), lab, c in zip(axes, bands, labs, cols):
        style_ax(ax)
        data, pos = [], []
        for i, (w0, w1) in enumerate(wins):
            k = (sg >= a) & (sg < b) & (st > w0) & (st <= w1)
            if k.sum() >= 3:
                data.append(np.log10(lo[k])); pos.append(i)
        vp = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(c); body.set_edgecolor("none"); body.set_alpha(0.35)
        for d, x in zip(data, pos):
            q1, q2, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color=c, lw=5, solid_capstyle="butt")
            ax.plot(x, q2, "o", color="white", ms=5, mec=c, mew=2)
            ax.text(x, np.max(d) + 0.12, f"{10 ** q2:.3f}", ha="center", fontsize=11.5, color=TEXT)
        ax.set_xticks(range(len(wins)), wl, fontsize=11.5, rotation=0)
        ax.set_title(lab, loc="left", color=c, fontsize=16, fontweight="bold")
        ax.set_xlabel("训练步数区间", fontsize=12.5)
    axes[0].set_ylabel("单步训练 loss（log₁₀）")
    yt = [-3, -2, -1, 0]
    axes[0].set_yticks(yt, ["0.001", "0.01", "0.1", "1"])
    fig.tight_layout(w_pad=1.0)
    save(fig, "qi21_b_train_dist.png")


def fig_train_by_sigma_norm(train):
    """Same split, all bands on one axis, each normalised to its first 500 steps = 100."""
    st = np.array([r["step"] for r in train]); lo = np.array([r["loss"] for r in train])
    sg = np.array([r["sigma"][0] for r in train])
    bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.8), (0.8, 0.9), (0.9, 1.0001)]
    labs = ["σ 0–0.1", "σ 0.1–0.3", "σ 0.3–0.8", "σ 0.8–0.9", "σ 0.9–1"]
    cols = ["#eb6834", "#86b6ef", "#5598e7", "#104281", "#b0126a"]
    edges = np.arange(0, 3001, 500)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(13, 6.6))
    style_ax(ax)
    for (a, b), lab, c in zip(bands, labs, cols):
        k = (sg >= a) & (sg < b)
        med = np.array([np.median(lo[k & (st > e0) & (st <= e1)]) for e0, e1 in zip(edges[:-1], edges[1:])])
        y = 100 * med / med[0]
        ax.plot(mid, y, "-o", color=c, lw=3, ms=7)
        ax.text(mid[-1] + 60, y[-1], f"{lab}  {y[-1]:.0f}", va="center", fontsize=14, color=c, fontweight="bold")
    ax.axhline(100, color=LIGHT, lw=1)
    ax.set_xlim(0, 3500); ax.set_xticks([250, 750, 1250, 1750, 2250, 2750], ["前 500", "~1000", "~1500", "~2000", "~2500", "~3000"])
    ax.set_xlabel("训练步数（每 500 步一段，取中位数）"); ax.set_ylabel("相对前 500 步（= 100）")
    save(fig, "qi21_b_train_by_sigma_norm.png")


def main():
    set_style()
    has = lambda n: os.path.exists(os.path.join(SRC, n))  # noqa: E731
    probe, val = jl("probe.jsonl"), jl("val.jsonl")
    if has("train.jsonl"):
        train = jl("train.jsonl")
        fig_loss(train, probe, os.environ.get("LOSS_TITLE", "loss：尖刺来自两端的 σ，趋势在测试集 loss 上看得最清楚"))
        drop = fig_sigma(probe, train)
        print("sigma drop %", {k: round(v, 1) for k, v in drop.items()})
    fig_probe_by_task(probe)
    fig_success(val, probe)
    first = fig_task_order(probe, val)
    print("first >=90%", first)
    if has("meta.json") and has("val"):
        with open(os.path.join(SRC, "meta.json")) as f:
            meta = json.load(f)
        vs = [r["step"] for r in val]
        fig_grid(val, meta, [vs[0], vs[len(vs) // 4], vs[len(vs) // 2], vs[-1]])
    else:
        print("skip qi21_b_samples.png (page 14): needs meta.json + val/ images from your own run; see deck/media/")


if __name__ == "__main__":
    main()
