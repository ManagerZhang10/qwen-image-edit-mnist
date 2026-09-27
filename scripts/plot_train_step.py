#!/usr/bin/env python3
"""第 5 步 · 画图（训练一步怎么走）：讲义第 3 页动画里的潜变量缩略图 tf_*.png。CPU，但需要 VAE 权重。

样本：make_pairs('test', 16) 的第 11 张（rot90，数字 5），σ = 0.5。噪声加在 VAE 潜变量上，不是像素上：
  x0   = 目标图的归一化 VAE mode 潜变量，与训练 / probe 的编码方式完全相同            [64, 32, 32]
  ε    = probe 的固定噪声（torch.randn，CPU，种子 1234 + 11）
  x_t  = (1 − σ)·x0 + σ·ε
  v    = ε − x0（训练目标）；v′ = (x_t − x̂0)/σ，x̂0 是第 3000 步 LoRA 那张一步猜测图
         （probe_loss.py vis 输出的 rot90_11_x0hat_0.5.png）重新过 VAE 编码的结果：接近、但不等于模型自己的潜变量。
显示：64 个通道取 x0 的前 3 个主成分当 RGB（x0 / ε / x_t 共用同一组基和色阶）；速度取第 1 主成分上的投影，红正蓝负。
每个潜变量像素 = 16×16 个图像像素。参考图和目标图本身按像素显示（tf_ref.png、tf_tgt.png）。

需要：本地 VAE 目录（权重快照里的 vae/，含 config.json + safetensors）、diffusers @ e0abab8（AutoencoderKLQwenImage21）、
     probe_loss.py vis 的输出目录。
用法：python scripts/plot_train_step.py --vae models/Qwen-Image-2.1/vae --vis outputs/probe/vis [--out outputs/figures]
"""
import argparse
import os

import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from qie_mnist import data as D
from qie_mnist.probe import to_pixels

IDX, S, PX = 11, 0.5, 200


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae", required=True, help="VAE 目录（Qwen-Image-2.1 快照下的 vae/）")
    ap.add_argument("--vis", required=True, help="probe_loss.py vis 的输出目录（需要 rot90_11_x0hat_0.5.png）")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--mnist-root", default=None)
    a = ap.parse_args()
    from diffusers import AutoencoderKLQwenImage21

    os.makedirs(a.out, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(a.vae).eval()  # fp32，CPU
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1)
    enc = lambda pil: ((vae.encode(to_pixels(pil)).latent_dist.mode() - mean) / std)  # noqa: E731

    def save(rgb, name):
        Image.fromarray(rgb).resize((PX, PX), Image.NEAREST).save(os.path.join(a.out, f"tf_{name}.png"))

    s = D.make_pairs("test", 16, root=a.mnist_root)[IDX]
    assert s["task"] == "rot90", s["task"]
    with torch.no_grad():
        x0 = enc(s["target"])
        hat = Image.open(os.path.join(a.vis, f"rot90_{IDX:02d}_x0hat_{S}.png")).convert("RGB").resize((D.RES, D.RES), Image.BICUBIC)
        x0h = enc(hat)
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1234 + IDX))
    xt = (1 - S) * x0 + S * eps
    v, vp = eps - x0, (xt - x0h) / S
    f = lambda t: t[0, :, 0].numpy().reshape(z, -1).T  # noqa: E731  [1024, 64]

    X0 = f(x0); mu = X0.mean(0)
    _, _, Vt = np.linalg.svd(X0 - mu, full_matrices=False)
    B = Vt[:3].T
    P0 = (X0 - mu) @ B
    lo, hi = np.percentile(P0, 1, 0), np.percentile(P0, 99, 0)
    # ε 没有均值偏移：加上 μ，让三张图落在同一色阶上
    rgb = lambda X: (np.clip(((X - mu) @ B - lo) / (hi - lo), 0, 1).reshape(32, 32, 3) * 255).astype(np.uint8)  # noqa: E731
    save(rgb(X0), "x0lat"); save(rgb(f(eps) + mu), "eps"); save(rgb(f(xt)), "xt")

    pv, pvp = f(v) @ B[:, 0], f(vp) @ B[:, 0]
    lim = np.percentile(np.abs(np.concatenate([pv, pvp])), 98)
    red = lambda p: (cm.RdBu_r(np.clip(p / lim, -1, 1) * .5 + .5)[..., :3].reshape(32, 32, 3) * 255).astype(np.uint8)  # noqa: E731
    save(red(pv), "v"); save(red(pvp), "vp")
    for k, name in (("ref", "ref"), ("target", "tgt")):
        s[k].resize((PX, PX), Image.LANCZOS).save(os.path.join(a.out, f"tf_{name}.png"))
    print("latent", tuple(x0.shape), "x0 std", float(x0.std()), "| mse(v′, ε−x0):", float(((vp - v) ** 2).mean()),
          "| mse(x̂0, x0):", float(((x0h - x0) ** 2).mean()), "->", a.out)


if __name__ == "__main__":
    main()
