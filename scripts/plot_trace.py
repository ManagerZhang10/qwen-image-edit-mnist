#!/usr/bin/env python3
"""第 5 步 · 画图（实验 A：一次编辑前向的张量追踪）：从 scripts/trace_forward.py 的输出画图。CPU。

讲义页 → 图（位置参数选图，默认全部）：
  第 4 页 张量怎么走          讲义用的是 draw.io 静态图（不提供生成脚本）；flow → qi21_a_flow.png 是同内容的 matplotlib 版
  第 5 页 参考图的两条预处理  prep → qi21_a_prep.png
  第 6 页 块因果注意力        mask_simple → qi21_a_mask_simple.png（示意图，不需要追踪数据）
讲义未用：latent、strip、mask、x0（README 的 qi21_a_x0hat.png）、shapes、rope。

用法：
  python scripts/plot_trace.py mask_simple                               # 不需要任何数据
  python scripts/plot_trace.py --src outputs/trace prep flow x0          # 需要 trace.json + trace_arrays.npz
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402,F401

from qie_mnist.plotting import cjk_fonts, save as _save  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default="outputs/trace", help="trace_forward.py output dir")
ap.add_argument("--out", default="outputs/figures")
ap.add_argument("figs", nargs="*", help="latent strip mask x0 shapes rope flow prep mask_simple（默认全部）")
ARGS = ap.parse_args()
OUT = Path(ARGS.src)
MEDIA = Path(ARGS.out)

TEXT, MUTED, ACCENT, GRAY, LIGHT, PANEL = "#1D1D1F", "#6E6E73", "#0071E3", "#86868B", "#D2D2D7", "#F5F5F7"
ACC_L = "#9CC8F5"   # condition image
TXT_C = "#C7C7CC"   # text
plt.rcParams.update({"font.family": cjk_fonts() + ["DejaVu Sans"],
                     "figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
                     "text.color": TEXT, "axes.edgecolor": GRAY, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.unicode_minus": False, "font.size": 16,
                     "axes.spines.top": False, "axes.spines.right": False})

T = A = S = SEGS = None  # 追踪数据：_load() 按需读取（mask_simple 不需要）


def _load():
    global T, A, S, SEGS
    T = json.load(open(OUT / "trace.json"))
    A = np.load(OUT / "trace_arrays.npz")
    S = T["stages"]
    SEGS = S["6_joint"]["segments"]
KIND_ZH = {"text": "文本", "cond0": "参考图", "target": "目标图（噪声）"}
KIND_C = {"text": TXT_C, "cond0": ACC_L, "target": ACCENT}


def save(fig, name):
    _save(fig, MEDIA, name)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def shp(l):
    return "[" + ",".join(str(v) for v in l) + "]"


# ---------------------------------------------------------------- 1. latent visual
def fig_latent():
    z = A["z_norm"].astype(np.float32)          # [64,32,32]
    C, H, W = z.shape
    X = z.reshape(C, -1).T
    X = X - X.mean(0)
    U, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    pc = X @ Vt[:3].T
    pc = (pc - np.percentile(pc, 1, 0)) / (np.percentile(pc, 99, 0) - np.percentile(pc, 1, 0) + 1e-6)
    pc = np.clip(pc, 0, 1).reshape(H, W, 3)
    var = (Sv[:3] ** 2).sum() / (Sv ** 2).sum()
    stds = z.reshape(C, -1).std(1)
    chans = list(np.argsort(-stds)[:3])

    fig = plt.figure(figsize=(19.2, 7.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1], wspace=0.12, left=0.02, right=0.98, top=0.8, bottom=0.12)
    ax = fig.add_subplot(gs[0]); ax.imshow(A["ref_rgba"][..., :3]); off(ax)
    ax.set_title("参考图\n512×512 像素", fontsize=18)
    ax = fig.add_subplot(gs[1]); ax.imshow(pc, interpolation="nearest"); off(ax)
    ax.set_title(f"潜变量 32×32\n64 通道 → PCA 3 维上色", fontsize=18, color=ACCENT)
    ax.set_xlabel(f"前 3 个主成分占方差 {var:.0%}", fontsize=14, color=MUTED)
    for j, c in enumerate(chans):
        ax = fig.add_subplot(gs[2 + j]); ax.imshow(z[c], cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest"); off(ax)
        ax.set_title(f"第 {c} 通道\n（归一化后）", fontsize=18)
        ax.set_xlabel(f"std {stds[c]:.2f}", fontsize=14, color=MUTED)
    ax = fig.add_subplot(gs[5]); ax.imshow(A["vae_recon_rgba"][..., :3]); off(ax)
    ax.set_title("VAE 解码还原\n512×512", fontsize=18)
    ax.set_xlabel(f"与原图平均差 {S['2_vae_encode']['vae_roundtrip_rgb_mae_uint8']:.1f}/255", fontsize=14, color=MUTED)
    fig.suptitle("一个格子 = 16×16 像素：512² 的图变成 32×32×64 的潜变量", fontsize=24, y=0.97)
    zr, zn = S["2_vae_encode"]["latent_raw"], S["2_vae_encode"]["latent_norm"]
    fig.text(0.5, 0.03, f"编码原始值 均值 {zr['mean']:.2f} / 标准差 {zr['std']:.2f}   →   (x − mean) / std 之后 均值 {zn['mean']:.2f} / 标准差 {zn['std']:.2f}",
             ha="center", fontsize=16, color=MUTED)
    save(fig, "qi21_a_latent.png")


# ---------------------------------------------------------------- 2. sequence strip
def fig_strip():
    L = S["6_joint"]["joint_len"]
    fig, ax = plt.subplots(figsize=(19.2, 7.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.05)
    ax.set_xlim(-L * 0.01, L * 1.01); ax.set_ylim(-3.0, 3.0); off(ax)
    y0, h = 0.6, 1.2
    for sg in SEGS:
        a, b, k = sg["start"], sg["end"], sg["kind"]
        ax.add_patch(Rectangle((a, y0), b - a, h, fc=KIND_C[k], ec="white", lw=1.5))
        mid = (a + b) / 2
        if sg["len"] > 150:
            ax.text(mid, y0 + h / 2, f"{KIND_ZH[k]}\n{sg['len']} 个 token", ha="center", va="center", fontsize=19,
                    color="white" if k == "target" else TEXT, fontweight="bold")
    for v in (0, S["6_joint"]["prefix_len"], L):
        ax.text(v, y0 + h + 0.12, str(v), ha="center", va="bottom", fontsize=13, color=MUTED)
    # labels for short text segments
    texts = [sg for sg in SEGS if sg["kind"] == "text"]
    for n, sg in enumerate(texts):
        mid = (sg["start"] + sg["end"]) / 2
        # per-token decode splits the UTF-8 bytes of 「度」 into two pieces; show the joined text
        label = sg.get("text", "").replace(" ��", " 度").replace("\n", "⏎")
        if len(label) > 34:  # break after the instruction instead of truncating
            k = label.find("<|im_end|>")
            label = label[:k] + "\n" + label[k:] if k > 0 else label
        ax.annotate(f"文本 {sg['len']} 个 token（下标 {sg['start']}–{sg['end']-1}）\n{label}", xy=(mid, y0), xytext=(mid + (190 if n == 0 else 0), -0.9),
                    ha="center", va="top", fontsize=14, color=TEXT,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    # modulation + cache brackets
    pre = S["6_joint"]["prefix_len"]
    yb = 2.45
    ax.plot([0, pre], [yb, yb], color=GRAY, lw=2); ax.plot([0, 0], [yb - .12, yb], color=GRAY, lw=2); ax.plot([pre, pre], [yb - .12, yb], color=GRAY, lw=2)
    ax.text(pre / 2, yb + 0.1, f"前缀 {pre} 个：调制用 t = 0 那一行 · 第 1 步算完存进 KV 缓存", ha="center", va="bottom", fontsize=16, color=MUTED)
    ax.plot([pre, L], [yb, yb], color=ACCENT, lw=2); ax.plot([L, L], [yb - .12, yb], color=ACCENT, lw=2)
    ax.text((pre + L) / 2, yb + 0.1, "调制用真实 t · 每步重算", ha="center", va="bottom", fontsize=16, color=ACCENT)
    s4 = S["4_prompt"]
    ax.text(L * 0.5, -2.2, f"文本编码器那一侧只有 {s4['kept_len']} 个位置（{s4['kept_len'] - s4['n_image_pad']} 个文本 + {s4['n_image_pad']} 个 image_pad）；"
            f"进 Transformer 时每个 image_pad 展开成 4 个，填进参考图的 1024 个潜变量 token", ha="center", fontsize=15, color=TEXT)
    shapes = S["6_joint"]["img_shapes"][0]
    ax.text(L * 0.5, -2.75, f"img_shapes = {[tuple(s) for s in shapes]}   ·   输出只取最后 {shapes[-1][1]*shapes[-1][2]} 个 = 速度 v",
            ha="center", fontsize=15, color=MUTED)
    fig.suptitle(f"联合序列一共 {L} 个 token：文本在前，参考图嵌在文本中间，目标图接在最后", fontsize=24, y=0.97)
    save(fig, "qi21_a_sequence_strip.png")


# ---------------------------------------------------------------- 3. mask heatmap
def dense_mask(ii, lo, hi):
    q = np.arange(lo, hi)
    same = (ii[q][:, None] == ii[q][None, :]) & (ii[q][:, None] >= 0)
    return (q[:, None] >= q[None, :]) | same


def fig_mask():
    ii = A["image_ids"]; L = len(ii)
    md = A["mask_ds"].astype(np.float32); ds = S["7_mask"]["downsample"]
    fig = plt.figure(figsize=(19.2, 9.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28, left=0.08, right=0.97, top=0.80, bottom=0.14)
    # 左：示意图，不按真实长度，每段只画几个 token
    segs = [("文本", 3, -1), ("参考图", 4, 0), ("指令", 4, -1), ("目标图", 4, 1)]
    ids = sum([[k] * n for _, n, k in segs], []); n = len(ids); ids = np.array(ids)
    q = np.arange(n)
    causal = q[:, None] >= q[None, :]
    same = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
    img = np.ones((n, n, 3))
    c_causal = np.array([0.03, 0.32, 0.61]); c_same = np.array([0.93, 0.47, 0.13]); c_no = np.array([0.95, 0.96, 0.98])
    img[:] = c_no; img[causal] = c_causal; img[same & ~causal] = c_same
    ax = fig.add_subplot(gs[0])
    ax.imshow(img, extent=[0, n, n, 0], interpolation="nearest")
    for i in range(n + 1):
        ax.axhline(i, color="white", lw=1.2); ax.axvline(i, color="white", lw=1.2)
    pos = 0
    for name, k, _ in segs:
        if pos: ax.axhline(pos, color=TEXT, lw=1.6); ax.axvline(pos, color=TEXT, lw=1.6)
        ax.text(pos + k / 2, -0.35, name, ha="center", va="bottom", fontsize=16)
        ax.text(-0.35, pos + k / 2, name, ha="right", va="center", fontsize=16)
        pos += k
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(n / 2, n + 0.9, "被看的 token（key）→", ha="center", va="top", fontsize=15, color=MUTED)
    ax.text(-2.6, n / 2, "正在算的 token（query）→", ha="center", va="center", fontsize=15, color=MUTED, rotation=90)
    ax.text(9.0, 5.0, "参考图看不到\n后面的指令", ha="center", va="center", fontsize=14, color=MUTED)
    ax.set_title("示意：每段只画几个 token（不按真实长度）", fontsize=17, color=MUTED, pad=40)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c_causal, label="因果：只看自己和前面"), Patch(color=c_same, label="同一张图内部：也能看后面"),
                       Patch(color=c_no, label="看不到", ec=LIGHT)], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=13, frameon=False)
    # zoom: last cond rows -> text -> first target rows
    cond = [s for s in SEGS if s["kind"] == "cond0"][0]; tg = [s for s in SEGS if s["kind"] == "target"][0]
    lo, hi = cond["end"] - 16, tg["start"] + 16
    Z = dense_mask(ii, lo, hi)
    qq = np.arange(lo, hi); cz = qq[:, None] >= qq[None, :]
    Zi = np.ones((hi - lo, hi - lo, 3)); Zi[:] = c_no; Zi[Z & cz] = c_causal; Zi[Z & ~cz] = c_same
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(Zi, extent=[lo, hi, hi, lo], interpolation="nearest")
    for sg in SEGS:
        for v in (sg["start"],):
            if lo < v < hi:
                ax2.axhline(v, color=GRAY, lw=1); ax2.axvline(v, color=GRAY, lw=1)
    txt = [s for s in SEGS if s["kind"] == "text" and s["start"] >= cond["end"]][0]
    ax2.text((txt["start"] + txt["end"]) / 2, lo - 0.8, "指令", ha="center", va="bottom", fontsize=15)
    ax2.text((lo + cond["end"]) / 2, lo - 0.8, "参考图末尾", ha="center", va="bottom", fontsize=15)
    ax2.text((tg["start"] + hi) / 2, lo - 0.8, "目标图开头", ha="center", va="bottom", fontsize=15)
    ax2.annotate("指令：下三角\n只看自己和前面", xy=(txt["start"] + 4, txt["start"] + 11),
                 xytext=(lo + 3, hi - 6), fontsize=15, color=TEXT, arrowprops=dict(arrowstyle="->", color=TEXT))
    ax2.annotate("目标图：看见全部前文\n块内互相都看得见", xy=(tg["start"] + 8, tg["start"] + 8), xytext=(txt["start"] + 2, tg["start"] + 13),
                 fontsize=15, color="white", arrowprops=dict(arrowstyle="->", color="white"))
    ax2.set_xlabel(f"key 下标 {lo}–{hi}", fontsize=15); ax2.set_ylabel(f"query 下标", fontsize=15)
    ax2.set_title("真实 mask 放大：参考图 → 指令 → 目标图 的交界", fontsize=17, color=MUTED, pad=34)
    ax2.tick_params(labelsize=12)
    fig.suptitle("块因果注意力：token 只看前面，但同一张图内部全连通", fontsize=24, y=0.975)
    fig.text(0.5, 0.905, "规则：允许看 = (在我前面) 或 (和我在同一张图里)；右图是本次前向真实 mask 的一段", ha="center", fontsize=15, color=MUTED)
    save(fig, "qi21_a_mask.png")


# ---------------------------------------------------------------- 4. x0_hat strip
def fig_x0():
    steps = list(A["x0_steps"]); imgs = A["x0_imgs"]
    sig = A["sigmas"]
    fig = plt.figure(figsize=(19.2, 9.0))
    n = len(steps) + 1
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.62], hspace=0.42, wspace=0.08, left=0.05, right=0.98, top=0.86, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(A["ref_rgba"][..., :3]); off(ax)
    ax.set_title("参考图", fontsize=18); ax.set_xlabel("指令：顺时针旋转 90°", fontsize=14, color=MUTED)
    for j, (k, im) in enumerate(zip(steps, imgs)):
        ax = fig.add_subplot(gs[0, j + 1]); ax.imshow(im[..., :3]); off(ax)
        s = sig[k - 1]
        ax.set_title(f"第 {k} 步", fontsize=18, color=ACCENT if k == steps[-1] else TEXT)
        ax.set_xlabel(f"σ = {s:.3f}", fontsize=15, color=MUTED)
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(1, len(sig))
    ax.plot(x, sig[:-1], "-o", color=GRAY, ms=4, lw=1.5, label="本次实际用的 σ（40 步）")
    lin = np.linspace(1.0, 1 / 40, 40)
    ax.plot(x, lin, "--", color=LIGHT, lw=1.5, label="未平移的等间距 σ")
    ax.plot(steps, [sig[k - 1] for k in steps], "o", color=ACCENT, ms=10, zorder=3)
    ax.set_xlabel("采样步", fontsize=15); ax.set_ylabel("σ（噪声占比）", fontsize=15)
    ax.set_xlim(0.5, 40.5); ax.set_ylim(0, 1.05); ax.legend(fontsize=13, loc="upper right", frameon=False)
    mu = S["5_noise"]["mu"]
    ax.text(1.5, 0.12, f"动态平移 μ = {mu:.3f}（按 {S['5_noise']['image_seq_len']} 个目标 token 算），末端拉伸到 σ = {sig[-2]:.3f}", fontsize=13, color=MUTED)
    fig.suptitle("每步的预测干净图 x₀′ = x_t − σ·v：第 5 步（σ = 0.94）就已转好，后面只修细节", fontsize=24, y=0.97)
    save(fig, "qi21_a_x0hat.png")


# ---------------------------------------------------------------- 5. shapes along the pipeline
def fig_shapes():
    P = T["params"]; s4 = S["4_prompt"]; L = S["6_joint"]["joint_len"]
    N = S["6_joint"]["n_target"]
    kept = s4["prompt_embeds"]["shape"]
    fig, ax = plt.subplots(figsize=(19.2, 10.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); off(ax)

    def box(x, y, w, h, title, sub, c=PANEL, tc=TEXT, ec=LIGHT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=c, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=15, color=tc, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=13, color=tc if tc != TEXT else MUTED)

    def arr(x0, y0, x1, y1, c=GRAY):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.6))

    w, h = 13, 7.5
    X = [1, 17, 33, 49, 65]
    HL = "#E6F0FB"
    # lane 1: image -> VAE -> token -> img_in
    y = 48
    box(X[0], y, w, h, "参考图", "PIL 512×512 RGB")
    box(X[1], y, w, h, "RGBA 张量", shp(S["1_image"]["vae_image_tensor"]["shape"]))
    box(X[2], y, w, h, "VAE 编码 → 归一化", shp(S["2_vae_encode"]["latent_norm"]["shape"]))
    box(X[3], y, w, h, "展平成 token", shp(S["3_flatten"]["cond_tokens"]["shape"]))
    box(X[4], y, w, h, "img_in 线性投影", "64 → 4096\n[1,1024,4096]", c=HL, tc=ACCENT, ec=ACCENT)
    # lane 2: text -> txt_in
    y = 34
    box(X[0], y, w, h, "指令 + 模板", f"{s4['input_ids_len']} 个 token\n（含 {s4['n_image_pad']} 个 image_pad）")
    box(X[1], y, w, h, "Qwen3-VL", f"{P['text_encoder']/1e9:.1f}B 参数 · 36 层")
    box(X[2], y, w, h, "末层隐状态", shp(s4["hidden_last_prenorm"]["shape"]) + "\n（未过 final norm）")
    box(X[3], y, w, h, f"去掉 system {s4['drop_idx']} 个", shp(kept))
    box(X[4], y, w, h, "txt_in 投影", "RMSNorm + MLP\n[1,281,4096]", c=HL, tc=ACCENT, ec=ACCENT)
    for yy in (48, 34):
        for i in range(4):
            arr(X[i] + w + 0.6, yy + h / 2, X[i + 1] - 0.6, yy + h / 2)
    # lane 3: noise -> img_in (same layer)
    y = 20
    box(X[0], y, w, h, "高斯噪声 x_T", shp(S["5_noise"]["target_latents"]["shape"]), c=ACCENT, tc="white", ec=ACCENT)
    box(X[4], y, w, h, "同一个 img_in", "64 → 4096\n[1,1024,4096]", c=HL, tc=ACCENT, ec=ACCENT)
    arr(X[0] + w + 0.6, y + h / 2, X[4] - 0.6, y + h / 2, c=ACCENT)
    # joint
    box(81, 20, 17.5, 35.5, "拼联合序列", f"文本 + 参考图 token\n嵌在 image_pad 处\n+ 目标 token 接末尾\n\n[1, {L}, 4096]", c=HL)
    for yy in (48, 34, 20):
        arr(X[4] + w + 0.6, yy + h / 2, 80.6, yy + h / 2)
    # lane 4: transformer -> output
    y = 3
    box(81, y, 17.5, 12, "Transformer · 宽 4096", f"{P['transformer']/1e9:.1f}B 参数 · 32 层\nnorm_out + proj_out：4096 → 64\n输出 [1, {L}, 64]")
    arr(89.7, 19.6, 89.7, 15.4)
    box(61, y, w + 2, 12, "只取最后 N 个", f"v  [1, {N}, 64]", c=ACCENT, tc="white", ec=ACCENT)
    box(41, y, w + 2, 12, "Euler 一步", "x ← x + (σₙ₊₁ − σₙ)·v\n× 40 步")
    box(21, y, w + 2, 12, "反归一化 → VAE 解码", "[1,64,1,32,32]\n→ [1,4,1,512,512]")
    box(1, y, w + 2, 12, "输出图", "RGBA → RGB 512×512")
    for x in (81, 61, 41, 21):
        arr(x - 0.4, y + 6, x - 4.4, y + 6)
    ax.text(48.5, 16.2, "循环 40 次（x 回到序列末尾）", ha="center", fontsize=13, color=MUTED)
    fig.suptitle("一次编辑里张量怎么走：三路输入各自投到 4096 维，拼成一条序列，最后投回 64 维只读目标段", fontsize=24, y=0.975)
    save(fig, "qi21_a_pipeline_shapes.png")


# ---------------------------------------------------------------- 6. RoPE ids
def fig_rope():
    f, h, w = A["rope_f"], A["rope_h"], A["rope_w"]
    L = len(f)
    fig, axs = plt.subplots(1, 2, figsize=(19.2, 7.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.18})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.12)
    ax = axs[0]
    x = np.arange(L)
    for sg in SEGS:
        ax.axvspan(sg["start"], sg["end"], color=KIND_C[sg["kind"]], alpha=0.25, lw=0)
    ax.plot(x, f, color=TEXT, lw=2, label="帧轴 frame")
    ax.plot(x, h, color=ACCENT, lw=1, alpha=0.9, label="高轴 h")
    ax.plot(x, w, color="#FF9F0A", lw=0.6, alpha=0.7, label="宽轴 w")
    ax.set_xlabel("联合序列下标", fontsize=15); ax.set_ylabel("位置编号", fontsize=15)
    ax.legend(fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.8))
    ax.set_xlim(0, L); ax.set_ylim(-20, 66)
    for sg in SEGS:
        if sg["len"] > 150:
            ax.text((sg["start"] + sg["end"]) / 2, 61, KIND_ZH[sg["kind"]].replace("（噪声）", ""), ha="center", fontsize=15, color=MUTED)
    ax2 = axs[1]; off(ax2)
    rows = [["段", "下标", "frame", "h", "w"]]
    for r in S["9_rope"]["per_segment"]:
        rows.append([KIND_ZH[r["kind"]].replace("（噪声）", ""), f"{r['start']}–{r['end']-1}",
                     f"{r['frame'][0]}" + ("" if r["frame"][0] == r["frame"][1] else f"–{r['frame'][1]}"),
                     f"{r['h'][0]}–{r['h'][1]}", f"{r['w'][0]}–{r['w'][1]}"])
    tb = ax2.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(15); tb.scale(1, 2.6)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor(LIGHT)
        if r_ == 0:
            cell.set_facecolor(PANEL); cell.set_text_props(fontweight="bold")
    ax2.set_title("每段的位置编号范围", fontsize=17, color=MUTED)
    fig.suptitle("位置编码：文本三轴一起 +1；每张图帧轴固定，h / w 以 0 为中心铺成网格", fontsize=24, y=0.97)
    save(fig, "qi21_a_rope.png")

# ---------------------------------------------------------------- 7. whole flow with latent visuals (relative sizes)
def fig_flow():
    """Pipeline drawn with the real tensors. Sizes are relative, not to scale:
    width grows with token count, height grows with feature dim, images with pixel side."""
    P = T["params"]; s4 = S["4_prompt"]; L = S["6_joint"]["joint_len"]
    n_in, n_kept = s4["input_ids_len"], s4["prompt_embeds"]["shape"][1]
    z = A["z_norm"].astype(np.float32)
    fig, ax = plt.subplots(figsize=(16.5, 9.11))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    Wn = lambda n: 13.5 * (n / 1024) ** 0.55          # token count -> width
    H64, H4096 = 2.2, 6.6                             # feature dim -> height
    HL = "#E6F0FB"

    def pca(zz):
        C = zz.shape[0]; X = zz.reshape(C, -1).T; X = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc = X @ Vt[:3].T
        pc = (pc - np.percentile(pc, 1, 0)) / (np.percentile(pc, 99, 0) - np.percentile(pc, 1, 0) + 1e-6)
        return np.clip(pc, 0, 1).reshape(zz.shape[1], zz.shape[2], 3)

    def img(a, x, y, w, h, **kw):
        ax.imshow(a, extent=(x, x + w, y, y + h), aspect="auto", interpolation=kw.pop("interp", "nearest"), zorder=kw.pop("zorder", 3), **kw)
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=kw.get("ec", GRAY), lw=1, zorder=4))

    def mat(x, y, w, h, c, alpha=1.0, z_=3):
        ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="white" if alpha == 1 else c, lw=1, alpha=alpha, zorder=z_))
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=GRAY, lw=1, zorder=z_ + 1))

    def cube(zz, x, y, s, d=1.6, n=6):
        for k in range(n, 0, -1):
            o = d * k / n
            ax.add_patch(Rectangle((x + o, y + o), s, s, fc="#EDEDF0", ec=GRAY, lw=0.8, zorder=2))
        img(pca(zz), x, y, s, s)

    def lab(x, y, t, sub=None, c=TEXT, va="top"):
        ax.text(x, y, t, ha="center", va=va, fontsize=14, color=c, fontweight="bold", zorder=6)
        if sub:
            ax.text(x, y - 1.35 if va == "top" else y + 1.35, sub, ha="center", va=va, fontsize=12.5, color=MUTED, zorder=6)

    def arr(x0, y0, x1, y1, t=None, c=GRAY, tc=None, ls="-", dy=0.5, fs=12.5, rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, ls=ls, mutation_scale=14, shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))
        if t:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, t, ha="center", va="bottom", fontsize=fs, color=tc or c, zorder=6, linespacing=1.25)

    # ---- lane A: reference image -> RGBA -> latent -> tokens -> img_in
    yA = 43.0; S_IMG = 9.0
    img(A["ref_rgba"][..., :3], 1, yA, S_IMG, S_IMG, interp="lanczos")
    lab(1 + S_IMG / 2, yA - 0.5, "参考图", "512×512 像素")
    xr = 14
    ch = A["ref_rgba"].astype(np.float32) / 255
    for k, (c, nm) in enumerate(zip([3, 2, 1, 0], "ABGR")):
        o = 0.75 * (3 - k)
        img(ch[..., c], xr + o, yA - 0.2 + o, S_IMG - 1.6, S_IMG - 1.6, cmap="gray", vmin=0, vmax=1, interp="lanczos")
    lab(xr + 4.6, yA - 1.1, "RGBA 张量", "[1,4,1,512,512]")
    arr(1 + S_IMG + 0.4, yA + 4.5, xr - 0.4, yA + 4.5, "[−1,1]", dy=0.4, fs=11.5)
    xz, sz = 31.5, 3.4
    cube(z, xz, yA + 2.4, sz)
    lab(xz + 2.3, yA + 1.6, "潜变量", "[1,64,1,32,32]")
    ax.text(xz + sz + 1.9, yA + 5.1, "64\n通道", fontsize=11, color=MUTED, ha="left", va="center")
    arr(xr + 2.25 + S_IMG - 1.6 + 0.4, yA + 4.5, xz - 0.4, yA + 4.5, "VAE 编码\n边长 ÷16", dy=0.4)
    xt = 42.0; wt = Wn(1024)
    img(z.reshape(64, -1), xt, yA + 3.4, wt, H64, cmap="RdBu_r", vmin=-3, vmax=3)
    lab(xt + wt / 2, yA + 2.9, "1024 个 token × 64 维", "[1,1024,64]")
    arr(xz + sz + 1.6 + 2.6, yA + 4.5, xt - 0.4, yA + 4.5, "展平", dy=0.4)
    xp = 60.5
    mat(xp, yA + 1.2, wt, H4096, ACC_L)
    lab(xp + wt / 2, yA + 0.7, "[1,1024,4096]")
    arr(xt + wt + 0.4, yA + 4.5, xp - 0.4, yA + 4.5, "img_in\n64→4096", c=ACCENT, dy=0.4, fs=11.5)

    # ---- lane B: instruction -> Qwen3-VL -> hidden -> drop -> txt_in
    yB = 27.5
    wi = Wn(n_in)
    for k in range(18):
        ax.add_patch(Rectangle((1 + k * wi / 18, yB + 3.1), wi / 18 * 0.8, 1.0, fc=TXT_C if not (5 <= k <= 13) else ACC_L, ec="none", zorder=3))
    lab(1 + wi / 2, yB + 2.6, f"指令 + 模板", f"{n_in} 个 token id")
    ax.add_patch(FancyBboxPatch((13.8, yB + 1.3), 8.2, 4.6, boxstyle="round,pad=0.2,rounding_size=0.8", fc=PANEL, ec=LIGHT, lw=1.2, zorder=3))
    ax.text(17.9, yB + 4.2, "Qwen3-VL", ha="center", va="center", fontsize=15, fontweight="bold", zorder=6)
    ax.text(17.9, yB + 2.6, f"{P['text_encoder']/1e9:.1f}B · 36 层", ha="center", va="center", fontsize=12.5, color=MUTED, zorder=6)
    arr(1 + wi + 0.4, yB + 3.6, 13.6, yB + 3.6)
    arr(1 + S_IMG / 2, yA - 3.2, 16.0, yB + 6.1, rad=0.15)
    ax.text(11.8, yA - 5.2, "参考图也进 Qwen3-VL\n→ 256 个 image_pad", fontsize=12, color=MUTED, ha="left", va="top", zorder=6)
    xh = 26.0; wh = Wn(n_in)
    mat(xh, yB + 0.3, wh, H4096, TXT_C)
    lab(xh + wh / 2, yB - 0.2, "末层隐状态", f"[1,{n_in},4096]")
    arr(22.4, yB + 3.6, xh - 0.4, yB + 3.6)
    xk = 42.0; wk = Wn(n_kept)
    mat(xk, yB + 0.3, wk, H4096, TXT_C)
    lab(xk + wk / 2, yB - 0.2, f"去掉 system {s4['drop_idx']} 个", f"[1,{n_kept},4096]")
    arr(xh + wh + 0.4, yB + 3.6, xk - 0.4, yB + 3.6)
    mat(xp, yB + 0.3, wk, H4096, TXT_C)
    lab(xp + wk / 2, yB - 0.2, "[1,281,4096]")
    arr(xk + wk + 0.4, yB + 3.6, xp - 0.4, yB + 3.6, "txt_in\nRMSNorm+MLP", c=ACCENT, dy=0.4)

    # ---- lane C: noise -> img_in
    yC = 14.5
    rng = np.random.default_rng(0)
    img(rng.standard_normal((64, 1024)), xt, yC + 3.4, wt, H64, cmap="RdBu_r", vmin=-3, vmax=3)
    lab(xt + wt / 2, yC + 3.4 + H64 + 1.9, "目标：纯噪声 x_T", None, c=ACCENT, va="bottom")
    ax.text(xt + wt / 2, yC + 3.4 + H64 + 0.5, "[1,1024,64]", ha="center", va="bottom", fontsize=12.5, color=MUTED)
    mat(xp, yC + 1.2, wt, H4096, ACCENT, alpha=0.85)
    lab(xp + wt / 2, yC + 0.7, "[1,1024,4096]")
    arr(xt + wt + 0.4, yC + 4.5, xp - 0.4, yC + 4.5, "同一个\nimg_in", c=ACCENT, dy=0.4)

    # ---- joint sequence
    xj = 79.0; wj = Wn(L); yj = yB - 0.6
    segs = [(sg["len"], KIND_C[sg["kind"]]) for sg in SEGS]
    tot = sum(max(n, 110) for n, _ in segs)
    x = xj
    for (n, c), sg in zip(segs, SEGS):
        w = wj * max(n, 110) / tot
        mat(x, yj, w, H4096 + 1.8, c, alpha=0.85 if c == ACCENT else 1.0)
        if n > 500:
            ax.text(x + w / 2, yj + (H4096 + 1.8) / 2, f"{KIND_ZH[sg['kind']].replace('（噪声）', '')}\n{n}", ha="center", va="center",
                    fontsize=13, color="white" if c == ACCENT else TEXT, fontweight="bold", zorder=6)
        x += w
    lab(xj + wj / 2, yj - 0.5, "联合序列", f"[1,{L},4096]")
    ax.text(xj + wj / 2, yj + H4096 + 2.4, "灰条 = 文本 8 个、指令 17 个", ha="center", va="bottom", fontsize=12.5, color=MUTED)
    for yy in (yA + 4.5, yB + 3.6, yC + 4.5):
        x0_ = xp + (wk if yy == yB + 3.6 else wt) + 0.4
        arr(x0_, yy, xj - 0.4, min(max(yy, yj + 0.8), yj + H4096 + 1.0))

    # ---- lane D (right -> left): transformer out -> last N -> Euler -> decode
    yD = 2.0
    wo = wj
    x = xj
    for n, c in segs:
        w = wj * max(n, 110) / tot
        mat(x, yD + 3.0, w, H64, c, alpha=0.35 if c != ACCENT else 0.85)
        x += w
    lab(xj + wj / 2, yD + 2.5, "输出", f"[1,{L},64]")
    arr(xj + wj / 2, yj - 3.1, xj + wj / 2, yD + 3.0 + H64 + 0.4,
        f"Transformer {P['transformer']/1e9:.1f}B · 32 层\nnorm_out + proj_out\n4096 → 64", c=TEXT, dy=-1.8, fs=12.5)
    ax.texts[-1].set_position((xj + wj / 2 - 0.8, (yj - 3.1 + yD + 5.6) / 2 - 2.0)); ax.texts[-1].set_ha("right")
    xv = 50.0
    mat(xv, yD + 3.0, wt, H64, ACCENT, alpha=0.85)
    lab(xv + wt / 2, yD + 2.5, "速度 v", "[1,1024,64]", c=ACCENT)
    arr(xj - 0.4, yD + 4.1, xv + wt + 0.4, yD + 4.1, "只取最后 1024 个", dy=0.4)
    x0l = A["x0_lat"][-1].astype(np.float32)
    xo = 33.0
    cube(x0l, xo, yD + 2.4, sz)
    lab(xo + 2.3, yD + 1.6, "x₀ 潜变量", "[1,64,1,32,32]")
    arr(xv - 0.4, yD + 4.1, xo + sz + 1.6 + 0.4, yD + 4.1, "40 步后\n反展平", dy=0.4)
    arr(xv + wt * 0.35, yD + 3.0 + H64 + 0.4, xt + wt * 0.5, yC + 3.4 - 0.4, c=ACCENT, ls="--")
    ax.text(xt + wt * 0.5 - 1.2, yC + 0.6, "x ← x + (σₙ₊₁ − σₙ)·v\n放回序列末尾，循环 40 次", fontsize=12.5, color=ACCENT, ha="right", va="top", zorder=6)
    xf = 14.0
    img(A["final_rgb"], xf, yD - 0.2, S_IMG, S_IMG, interp="lanczos")
    lab(xf + S_IMG / 2, yD - 0.7, "输出图", "512×512 像素")
    arr(xo - 0.4, yD + 4.1, xf + S_IMG + 0.4, yD + 4.1, "反归一化\nVAE 解码 ×16", dy=0.4)

    ax.text(1, yC + 7.2, f"[1,4,1,512,512] = [批, 通道 RGBA, 帧, 高, 宽]\n[1,1024,64] = [批, token 数, 每个 token 的维度]\n"
            f"{n_in} = system 14 + 图前文本 8 + image_pad 256 + 指令 17\n（system 每个样本都一样，编码完丢掉）",
            fontsize=12.5, color=TEXT, ha="left", va="top", linespacing=1.55)
    ax.text(1, yC - 0.2, "尺寸只表示相对大小：宽 ∝ token 数，\n高 ∝ 每个 token 的维度，图片边长 ∝ 像素边长",
            fontsize=12, color=MUTED, ha="left", va="top", linespacing=1.4)
    ax.set_xlim(0, 100); ax.set_ylim(-2.2, 53.0); ax.set_aspect("equal"); off(ax)
    save(fig, "qi21_a_flow.png")


def fig_mask_simple():
    """Schematic block-causal mask only (each segment drawn with a few tokens)."""
    segs = [("文本", 3, -1), ("参考图", 4, 0), ("指令", 4, -1), ("目标图", 4, 1)]
    ids = np.array(sum([[k] * n for _, n, k in segs], [])); n = len(ids)
    q = np.arange(n)
    causal = q[:, None] >= q[None, :]
    same = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
    c_causal = np.array([0.03, 0.32, 0.61]); c_same = np.array([0.93, 0.47, 0.13]); c_no = np.array([0.95, 0.96, 0.98])
    img = np.ones((n, n, 3)); img[:] = c_no; img[causal] = c_causal; img[same & ~causal] = c_same
    fig, ax = plt.subplots(figsize=(10.5, 10))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12)
    ax.imshow(img, extent=[0, n, n, 0], interpolation="nearest")
    for i in range(n + 1):
        ax.axhline(i, color="white", lw=1.2); ax.axvline(i, color="white", lw=1.2)
    pos = 0
    for name, k, _ in segs:
        if pos: ax.axhline(pos, color=TEXT, lw=1.8); ax.axvline(pos, color=TEXT, lw=1.8)
        ax.text(pos + k / 2, -0.3, name, ha="center", va="bottom", fontsize=19)
        ax.text(-0.3, pos + k / 2, name, ha="right", va="center", fontsize=19)
        pos += k
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(n / 2, n + 0.5, "被看的 token（Key）→", ha="center", va="top", fontsize=16, color=MUTED)
    ax.text(-2.3, n / 2, "正在算的 token（Query）→", ha="center", va="center", fontsize=16, color=MUTED, rotation=90)
    ax.text(9.0, 5.0, "参考图看不到\n后面的指令", ha="center", va="center", fontsize=17, color=TEXT)
    ax.text(13.0, 5.0, "前面的 token\n都看不到目标图", ha="center", va="center", fontsize=17, color=TEXT)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c_causal, label="只看自己和前面"), Patch(color=c_same, label="同一张图内部：也能看后面"),
                       Patch(color=c_no, label="看不到", ec=LIGHT)], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=15, frameon=False)
    save(fig, "qi21_a_mask_simple.png")


C_VAE, C_VLM, C_TXT, C_REF, C_TGT, C_ATT = "#C8E6C4", "#C8E6C4", "#D9D9D9", "#CFE0F7", "#E2D3F5", "#FDE6C6"
BLUE_T = "#1F5FAF"


def _draw_prep(TH):
    import os
    from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: F811
    from PIL import Image
    fig = plt.figure(figsize=(16, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1600); ax.set_ylim(740, 0); ax.axis("off")

    def box(x, y, w, h, t, fc, sub=None, ec="none", fs=15, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12", fc=fc, ec=ec, lw=2))
        cy = y + h / 2 - (9 if sub else 0)
        ax.text(x + w / 2, cy, t, ha="center", va="center", fontsize=fs, color=TEXT, fontweight=600 if bold else None)
        if sub:
            ax.text(x + w / 2, cy + 22, sub, ha="center", va="center", fontsize=11.5, color=MUTED)

    def arrow(x0, y0, x1, y1, color="#555", label=None, lx=None, ly=None, ha="center"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=0, shrinkB=0))
        if label:
            ax.text(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2 - 10,
                    label, ha=ha, va="bottom", fontsize=12, color=color if color != "#555" else MUTED)

    def img(x, y, w, h, path, border=None):
        im = Image.open(path).convert("RGBA")
        ax.imshow(np.asarray(im), extent=(x, x + w, y + h, y), zorder=2, interpolation="bilinear")
        if border:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=border, lw=1.2, zorder=3))

    # ---------------- 左：参考图 + 两层网格
    X, Y, S = 24, 190, 300
    ax.text(X, Y - 64, "参考图 512 × 512", fontsize=17, color=TEXT, fontweight=600, va="bottom")
    ax.text(X, Y - 40, "细格 = 16 px（VAE 一个 token，ViT 一块）", fontsize=12, color=MUTED, va="bottom")
    ax.text(X, Y - 18, "粗格 = 32 px（2×2 合并后 = 1 个 slot）", fontsize=12, color=BLUE_T, va="bottom")
    ref = Image.open(os.path.join(str(TH), "ref.png")).convert("L").resize((512, 512), Image.BICUBIC)
    ax.imshow(np.asarray(ref), cmap="gray", extent=(X, X + S, Y + S, Y), zorder=1, vmin=0, vmax=255)
    for k in range(33):
        t = X + k * S / 32
        lw, c, a = (1.1, "#5B9BE8", .9) if k % 2 == 0 else (.5, "#9A9A9A", .55)
        ax.plot([t, t], [Y, Y + S], color=c, lw=lw, alpha=a, zorder=3)
        t = Y + k * S / 32
        ax.plot([X, X + S], [t, t], color=c, lw=lw, alpha=a, zorder=3)
    bx, by = X + 12 * S / 16, Y + 5 * S / 16
    ax.add_patch(Rectangle((bx, by), S / 16, S / 16, fill=False, ec="#FFB020", lw=2.6, zorder=4))
    ax.text(X + S / 2, Y + S + 24, "黄框：1 个 slot = 2×2 个 16px 块", ha="center", fontsize=12, color="#B77800")

    # ---------------- 上路：VAE
    yA = 110
    ax.text(380, yA - 58, "VAE 这一路：给 DiT 看像素细节", fontsize=15, color=TEXT, fontweight=600)
    arrow(X + S + 6, Y + 40, 380, yA, color="#555")
    box(380, yA - 30, 150, 60, "VAE 编码", C_VAE, sub="÷16 · 逐通道归一化")
    arrow(530, yA, 570, yA)
    img(572, yA - 58, 116, 116, os.path.join(str(TH), "lat.png"))
    ax.text(630, yA + 72, "32 × 32 × 64", ha="center", fontsize=12, color=MUTED)
    arrow(700, yA, 740, yA, label="展平", ly=yA - 12)
    img(742, yA - 14, 230, 28, os.path.join(str(TH), "tok.png"), border="#8E8E93")
    ax.text(857, yA + 34, "1024 个 token × 64 维", ha="center", fontsize=12.5, color=TEXT)
    arrow(972, yA, 1010, yA)
    box(1012, yA - 30, 120, 60, "img_in", C_ATT, sub="64 → 4096")
    arrow(1132, yA, 1170, yA)
    box(1172, yA - 30, 170, 60, "参考图 1024 个", C_REF, sub="× 4096")

    # ---------------- 下路：Qwen3-VL
    yB = 400
    ax.text(380, yB - 70, "Qwen3-VL 这一路：让指令「看懂」参考图", fontsize=15, color=TEXT, fontweight=600)
    arrow(X + S + 6, Y + S - 40, 380, yB, color="#555")
    box(380, yB - 36, 170, 72, "16px 切块", C_VLM, sub="32×32 = 1024 块")
    ax.text(465, yB + 48, "每块 16×16×3 × 2 帧\n= 1536 个数", ha="center", va="top", fontsize=11.5, color=MUTED)
    arrow(550, yB, 585, yB)
    box(587, yB - 36, 150, 72, "ViT 27 层", C_VLM, sub="1024 × 1152")
    arrow(737, yB, 772, yB)
    box(774, yB - 36, 150, 72, "2×2 合并", C_VLM, sub="→ 256 × 4096")
    # 2×2 -> 1 小示意
    gx, gy, c = 800, yB + 50, 13
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((gx + j * c, gy + i * c), c - 1, c - 1, fc="#9A9A9A", ec="none"))
    ax.annotate("", xy=(gx + 58, gy + c), xytext=(gx + 30, gy + c), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.add_patch(Rectangle((gx + 62, gy), 2 * c - 1, 2 * c - 1, fc="#5B9BE8", ec="none"))
    arrow(924, yB, 960, yB)
    # VLM 序列条
    sx, sw = 962, 380
    ax.text(sx + sw / 2, yB - 46, "Qwen3-VL 的输入序列（去掉 system 后 281 个）", ha="center", fontsize=12, color=MUTED)
    segs = [("文本 8", C_TXT, .7), ("256 个 image_pad（slot）", C_REF, 3.2), ("指令等 17", C_TXT, 1.1)]
    tot = sum(s[2] for s in segs); cx = sx
    for lab, col, wgt in segs:
        w = sw * wgt / tot - 4
        ax.add_patch(FancyBboxPatch((cx, yB - 24), w, 48, boxstyle="round,pad=0,rounding_size=8", fc=col, ec="none"))
        ax.text(cx + w / 2, yB, lab, ha="center", va="center", fontsize=12, color=TEXT)
        cx += w + 4
    arrow(sx + sw + 4, yB, sx + sw + 40, yB)
    box(sx + sw + 42, yB - 36, 196, 72, "Qwen3-VL 语言模型", C_VLM, sub="36 层 · 图文一条序列", fs=14)
    ax.text(sx + sw + 140, yB + 56, "image_pad 处的输出丢掉，\n只留文本 25 个", ha="center", fontsize=11.5, color=MUTED, va="top")

    # ---------------- 底部：DiT 联合序列
    yC = 640
    ax.text(380, yC - 60, "拼进 DiT：每个 slot 展开成 4 格，按行顺序填 VAE token（只对数量，不对 2×2 位置）", fontsize=14,
            color=TEXT, fontweight=600)
    jx, jw = 380, 1196
    js = [("文本 8", C_TXT, .5), ("参考图 1024 = 256 slot × 4", C_REF, 4), ("指令等 17", C_TXT, .9),
          ("目标 1024 = 末尾补 256 个空 slot × 4", C_TGT, 4)]
    tot = sum(s[2] for s in js); cx = jx; pos = {}
    for lab, col, wgt in js:
        w = jw * wgt / tot - 5
        ax.add_patch(FancyBboxPatch((cx, yC - 26), w, 52, boxstyle="round,pad=0,rounding_size=8", fc=col, ec="none"))
        ax.text(cx + w / 2, yC, lab, ha="center", va="center", fontsize=12.5, color=TEXT)
        pos[lab[:2]] = (cx, w); cx += w + 5
    ax.text(jx + jw, yC - 60, "联合序列 2073 × 4096", ha="right", fontsize=12.5, color=MUTED)
    # 来源用颜色和小字标，不画长箭头（会穿过中间一路）
    rx, rw = pos["参考"]
    ax.text(rx + rw / 2, yC + 34, "上路：VAE 的 1024 个 token", ha="center", va="top", fontsize=12, color=BLUE_T)
    ax.text(pos["文本"][0], yC + 62, "灰色两段：下路 Qwen3-VL 输出的文本 8 + 17 个", ha="left", va="top", fontsize=12, color=MUTED)
    gx_, gw_ = pos["目标"]
    ax.text(gx_ + gw_ / 2, yC + 34, "加噪目标 x_t 的 1024 个 token", ha="center", va="top", fontsize=12, color="#7A4FB5")
    _save(fig, MEDIA, "qi21_a_prep.png", facecolor="white")




def _prep_thumbs(th):
    """参考图、潜变量立方体、token 条三张缩略图（与讲义所用图完全一致的生成方式），写到临时目录 th。"""
    from PIL import Image

    def pca(zz):
        C = zz.shape[0]; X = zz.reshape(C, -1).T; X = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc = X @ Vt[:3].T
        pc = (pc - np.percentile(pc, 1, 0)) / (np.percentile(pc, 99, 0) - np.percentile(pc, 1, 0) + 1e-6)
        return (np.clip(pc, 0, 1).reshape(zz.shape[1], zz.shape[2], 3) * 255).astype(np.uint8)

    Image.fromarray(A["ref_rgba"][..., :3]).resize((256, 256), Image.LANCZOS).save(th / "ref.png")
    zz = A["z_norm"].astype(np.float32)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    for k in range(6, 0, -1):
        o = 0.035 * k
        ax = fig.add_axes([0.04 + o, 0.04 + o, 0.7, 0.7]); ax.set_facecolor("#EDEDF0")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#8E8E93")
    ax = fig.add_axes([0.04, 0.04, 0.7, 0.7]); ax.imshow(pca(zz), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#8E8E93")
    fig.savefig(th / "lat.png", transparent=True); plt.close(fig)
    fig = plt.figure(figsize=(8, 0.9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(zz.reshape(64, -1), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(th / "tok.png"); plt.close(fig)


def fig_prep():
    """第 5 页：参考图的两条预处理路线。VAE（16px 一个 latent token）和 Qwen3-VL（16px 切块、2×2 合并成一个 image_pad
    slot），以及 slot 在 DiT 联合序列里展开成 4 个 latent token。图中数字来自实验 A 那次真实前向
    （pixel_values [1024,1536]、ViT 输出 [1024,1152]、image_pad 256 个、prompt_embeds [1,281,4096]、联合序列 2073）。"""
    import os
    import tempfile
    from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: F811
    from PIL import Image
    from qie_mnist.plotting import set_style
    with tempfile.TemporaryDirectory() as td, plt.rc_context():
        matplotlib.rcdefaults()   # 缩略图按 matplotlib 默认样式生成
        _prep_thumbs(Path(td))
        set_style()
        _draw_prep(Path(td))


if __name__ == "__main__":
    import traceback
    todo = ARGS.figs or ["latent", "strip", "mask", "x0", "shapes", "rope", "flow", "prep", "mask_simple"]
    if any(n != "mask_simple" for n in todo):
        _load()
    for n in todo:
        try:
            globals()["fig_" + n]()
        except Exception:
            traceback.print_exc()
