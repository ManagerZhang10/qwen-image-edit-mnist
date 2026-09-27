#!/usr/bin/env python3
"""Step 5 - figure "one training step": the latent thumbnails tf_*.png in the deck p.3 (One training step) animation.
CPU, but needs the VAE weights.

Sample: make_pairs('test', 16)[11] (rot90, digit 5), sigma = 0.5. Noise is added to VAE latents, not pixels:
  x0   = normalized VAE mode latent of the target, encoded exactly as in training / the probe       [64, 32, 32]
  eps  = the probe's fixed noise (torch.randn on CPU, seed 1234 + 11)
  x_t  = (1 - sigma) * x0 + sigma * eps
  v    = eps - x0 (training target); v' = (x_t - x0_hat) / sigma, where x0_hat is the step-3000 LoRA's one-step guess
         (rot90_11_x0hat_0.5.png from probe_loss.py vis) encoded again by the VAE: close to, not exactly, the model's
         own latent.
Display: the 64 channels -> first 3 principal components of x0 as RGB (one shared basis and scale for x0 / eps / x_t);
velocities -> projection on the first component, red + / blue -. One latent pixel = 16x16 image pixels. The pair
itself is shown in pixels (tf_ref.png, tf_tgt.png).

Needs: a local VAE dir (vae/ of the weight snapshot, config.json + safetensors), diffusers @ e0abab8
(AutoencoderKLQwenImage21), and the probe_loss.py vis output dir.
Usage: python scripts/plot_train_step.py --vae models/Qwen-Image-2.1/vae --vis outputs/probe/vis [--out outputs/figures]
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
    ap.add_argument("--vae", required=True, help="VAE dir (vae/ of the Qwen-Image-2.1 snapshot)")
    ap.add_argument("--vis", required=True, help="output dir of probe_loss.py vis (needs rot90_11_x0hat_0.5.png)")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--mnist-root", default=None)
    a = ap.parse_args()
    from diffusers import AutoencoderKLQwenImage21

    os.makedirs(a.out, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(a.vae).eval()  # fp32, CPU
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
    # eps has no mean offset: add mu so all three share one colour scale
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
