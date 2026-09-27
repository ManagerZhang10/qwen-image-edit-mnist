"""Held-out flow-matching loss at fixed sigmas ("probe loss"), shared by the training hooks and probe_loss.py.

Convention (diffusers FlowMatchEulerDiscreteScheduler / Qwen-Image):
    x_t = (1 - sigma) * x0 + sigma * noise,   v = noise - x0,   loss = mean((v_pred - v)^2)

Preprocessing matches the diffusers img2img LoRA trainer at 512x512: RGBA, [0,1] -> [-1,1],
vae.encode().latent_dist.mode(), then per-channel (x - mean) / std.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import TASK_LIST


def to_pixels(pil):
    """PIL -> (1, 4, 1, H, W) float tensor in [-1, 1], RGBA, with the frame axis the VAE expects."""
    a = np.asarray(pil.convert("RGBA"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1)
    return (t * 2 - 1).unsqueeze(0).unsqueeze(2)


class LatentEncoder:
    """Normalized VAE mode latents, identical to the trainer's latent space."""

    def __init__(self, vae, device="cuda"):
        self.vae, self.device = vae, device
        z = vae.config.z_dim
        self.mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device)
        self.inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device)

    @torch.no_grad()
    def __call__(self, pils, bs=16):
        """Returns (latents float32 CPU [N,C,1,h,w], normalized posterior std)."""
        lats, stds = [], []
        for i in range(0, len(pils), bs):
            x = torch.cat([to_pixels(p) for p in pils[i:i + bs]]).to(self.device, self.vae.dtype)
            post = self.vae.encode(x).latent_dist
            lats.append(((post.mode() - self.mean) * self.inv_std).float().cpu())
            stds.append((post.std * self.inv_std).float().cpu())
        return torch.cat(lats), torch.cat(stds)

    @torch.no_grad()
    def decode(self, lat_norm):
        """Normalized latent (1,C,1,h,w) -> uint8 RGB numpy [H,W,3]."""
        x = self.vae.decode((lat_norm.float().to(self.device) / self.inv_std + self.mean).to(self.vae.dtype),
                            return_dict=False)[0]
        return ((x[0, :3, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def fixed_noise(shapes, seed_base=1234, device="cuda", dtype=torch.bfloat16):
    """One noise tensor per sample: torch.randn on CPU with seed seed_base + i (same as the training probe)."""
    g = torch.Generator(device="cpu")
    return [torch.randn(s, generator=g.manual_seed(seed_base + i)).to(device, dtype) for i, s in enumerate(shapes)]


@torch.no_grad()
def forward_v(transformer, x0, cond, noise, emb, pad, sigma, device="cuda"):
    """One transformer call at a fixed sigma for a batch sharing the same prompt layout.

    x0, cond, noise: (B, C, 1, h, w) normalized latents; emb: (B, L, D) prompt embeddings; pad: (1, L) image-pad
    mask. Returns (v_pred, x_t), both (B, C, 1, h, w).
    """
    from diffusers import QwenImage21Pipeline as P
    bsz, c, _, h, w = x0.shape
    xt = (1 - sigma) * x0 + sigma * noise
    packed = torch.cat([P._pack_latents(cond, bsz, c, h, w), P._pack_latents(xt, bsz, c, h, w)], 1)
    img_mask = torch.cat([pad.repeat(bsz, 1), torch.ones(bsz, h * w // 4, dtype=pad.dtype, device=device)], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = transformer(hidden_states=packed, encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                           timestep=torch.full((bsz,), sigma, device=device, dtype=torch.float32),
                           img_shapes=[[(1, h, w), (1, h, w)]] * bsz, img_mask=img_mask, return_dict=False)[0]
    return P._unpack_latents(pred[:, -(h * w):], h * 16, w * 16, 16), xt


def task_groups(samples, embeds, task):
    """Indices of one task, grouped by prompt layout (same pad mask and length -> legal batch)."""
    groups = {}
    for i, s in enumerate(samples):
        if s["task"] == task:
            groups.setdefault(tuple(embeds[i][1].shape) + (embeds[i][0].shape[1],), []).append(i)
    return list(groups.values())


@torch.no_grad()
def probe_loss(transformer, samples, embeds, tgt, cond, noise_sets, sigmas, device="cuda"):
    """Per-sample v-MSE at each sigma for each noise set.

    embeds: list of (prompt_embeds (1,L,D), image_pad_mask (1,L)); tgt / cond: lists of (1,C,1,h,w) latents;
    noise_sets: list of lists of noise tensors (one list per noise draw).
    Returns per[task][str(sigma)][noise_idx] = list of per-sample losses.
    """
    per = {t: {str(s): [[] for _ in noise_sets] for s in sigmas} for t in TASK_LIST}
    for task in TASK_LIST:
        for gi in task_groups(samples, embeds, task):
            x0 = torch.cat([tgt[i] for i in gi])
            cnd = torch.cat([cond[i] for i in gi])
            emb = torch.cat([embeds[i][0] for i in gi])
            pad = embeds[gi[0]][1]
            if not all(torch.equal(embeds[i][1], pad) for i in gi):
                raise RuntimeError("pad layout differs inside a task group")
            for j, ns in enumerate(noise_sets):
                noise = torch.cat([ns[i] for i in gi])
                for sg in sigmas:
                    pred, _ = forward_v(transformer, x0, cnd, noise, emb, pad, sg, device)
                    loss = ((pred.float() - (noise - x0).float()) ** 2).reshape(len(gi), -1).mean(1)
                    per[task][str(sg)][j] += loss.tolist()
    return per
