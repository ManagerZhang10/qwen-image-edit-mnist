#!/usr/bin/env python3
"""Analysis D: held-out loss vs sigma, and the sigma = 1 floor D = E_c[Var(x0 | c)].

Deck: p.10 (Sigma sweep of the x0 guess) <- vis, p.11 (loss = x0 error x 1/sigma^2) <- measure.
Needs one CUDA GPU with >= 40 GB; figures with scripts/plot_probe.py (CPU).

measure
  Part 1 (VAE only): D per task in the exact training latent space.
    - next: pooled within-class variance of class-k MNIST latents (k = 0..9), plus per-class values
    - deterministic tasks: D = 0 (target is a function of the reference; VAE mode encoding is deterministic), checked
    - reference: unconditional total variance of target latents across all classes
    - sanity: std of the normalized latents, VAE posterior variance, bf16 vs fp32 VAE
  Part 2 (transformer): the training probe (same 16 test pairs per task, same fixed noise seed 1234+i, plus
    extra noise draws) at sigma in SIGMAS, for the base model and optionally a LoRA.
vis
  For one rot90 and one next sample: x_t decoded through the VAE, the one-step guess x0' = x_t - sigma * v'
  decoded, and the per-sample v-loss / x0-error (x0-error = sigma^2 * v-loss) at --sigmas (the deck uses
  0.1, 0.5, 0.95, 0.99, 1.0); at the last sigma, --extra more noise draws (x0' only) show how the guess varies.

Usage:
  python scripts/probe_loss.py measure --model MODEL_DIR --lora LORA_DIR --out outputs/probe
  python scripts/probe_loss.py vis --model MODEL_DIR --lora LORA_DIR --out outputs/probe/vis
Then: python scripts/plot_probe.py --src outputs/probe --vis outputs/probe/vis
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from qie_mnist import data as D
from qie_mnist.probe import LatentEncoder, fixed_noise, forward_v, probe_loss

SIGMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
N_VAL = 16
DEV = "cuda"
BF16 = torch.bfloat16


def log(msg):
    print(f"[probe_loss {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def part1(vae, n_train_per_class, n_test_per_class):
    t0 = time.time()
    enc = LatentEncoder(vae, DEV)
    xtr, ytr, xte, yte = D.load_mnist()
    rng = np.random.default_rng(0)
    res = dict(n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    lat_tr, var_tr, post_var = {}, {}, []
    for k in range(10):
        idx = rng.choice(np.flatnonzero(ytr == k), n_train_per_class, replace=False)
        L, S = enc([D.to_pil(xtr[i]) for i in idx])
        lat_tr[k] = L
        var_tr[k] = L.var(0, unbiased=True)  # per-element within-class variance
        post_var.append((S ** 2).mean().item())
        log(f"class {k}: within-class var {var_tr[k].mean().item():.4f}")
    res["latent_shape"] = list(lat_tr[0].shape[1:])
    res["D_next_per_class_train"] = {k: var_tr[k].mean().item() for k in range(10)}
    res["D_next_train"] = float(np.mean(list(res["D_next_per_class_train"].values())))
    allL = torch.cat([lat_tr[k] for k in range(10)])
    res["D_uncond_train"] = allL.var(0, unbiased=True).mean().item()
    res["vae_posterior_var_mean"] = float(np.mean(post_var))
    res["std_per_image_mean"] = allL.flatten(1).std(1).mean().item()
    res["std_global"] = allL.std().item()
    res["mean_global"] = allL.mean().item()
    res["std_per_channel"] = allL.transpose(0, 1).flatten(1).std(1).tolist()
    # Gaussian approximation of the MMSE for next: pooled scalar and per-element
    res["gauss_pooled"] = {str(s): res["D_next_train"] / ((1 - s) ** 2 * res["D_next_train"] + s ** 2) for s in SIGMAS}
    res["gauss_per_element"] = {
        str(s): float(np.mean([(var_tr[k] / ((1 - s) ** 2 * var_tr[k] + s ** 2)).mean().item() for k in range(10)]))
        for s in SIGMAS}
    vu = allL.var(0, unbiased=True)
    res["gauss_uncond_per_element"] = {str(s): (vu / ((1 - s) ** 2 * vu + s ** 2)).mean().item() for s in SIGMAS}

    # test split within-class variance (the probe's next targets come from the test split)
    var_te = {}
    for k in range(10):
        idx = rng.choice(np.flatnonzero(yte == k), n_test_per_class, replace=False)
        L, _ = enc([D.to_pil(xte[i]) for i in idx])
        var_te[k] = L.var(0, unbiased=True).mean().item()
    res["D_next_per_class_test"] = var_te
    res["D_next_test"] = float(np.mean(list(var_te.values())))

    # fp32 VAE cross-check on one class (training used the VAE in bf16)
    vae32 = copy.deepcopy(vae).float()
    idx = np.flatnonzero(ytr == 1)[:64]
    L16, _ = enc([D.to_pil(xtr[i]) for i in idx])
    L32, _ = LatentEncoder(vae32, DEV)([D.to_pil(xtr[i]) for i in idx])
    res["fp32_check_class1_64"] = dict(var_bf16=L16.var(0).mean().item(), var_fp32=L32.var(0).mean().item(),
                                       mse_bf16_vs_fp32=((L16 - L32) ** 2).mean().item())
    del vae32
    torch.cuda.empty_cache()

    # determinism: data generation and VAE mode encoding
    p1 = D.make_pairs("train", 500)
    p2 = D.make_pairs("train", 500)
    same = all(np.array_equal(a["tgt28"], b["tgt28"]) and np.array_equal(a["ref28"], b["ref28"]) and a["task"] == b["task"]
               for a, b in zip(p1, p2))
    det = [p for p in p1 if p["task"] != "next"]
    func_ok = True
    for p in det:
        t = p["task"]
        want = np.rot90(p["ref28"], -1) if t == "rot90" else np.rot90(p["ref28"], 2) if t == "rot180" else 255 - p["ref28"]
        func_ok &= bool(np.array_equal(want, p["tgt28"]))
    two = [next(p for p in p1 if p["task"] == "rot90"), next(p for p in p1 if p["task"] == "invert")]
    a, _ = enc([p["target"] for p in two])
    b, _ = enc([p["target"] for p in two])
    c, _ = enc([two[0]["target"]], bs=1)  # batch-size independence
    res["determinism"] = dict(make_pairs_reproducible=bool(same), deterministic_targets_are_function_of_ref=bool(func_ok),
                              n_deterministic_pairs=len(det), encode_twice_max_abs_diff=(a - b).abs().max().item(),
                              encode_bs1_vs_bs2_max_abs_diff=(a[:1] - c).abs().max().item())

    # oracle sigma=1 loss on the probe's 16 next pairs: predict the train class mean of class n+1
    samples = D.make_pairs("test", N_VAL)
    nxt = [s for s in samples if s["task"] == "next"]
    X, _ = enc([s["target"] for s in nxt])
    mu = {k: lat_tr[k].mean(0) for k in range(10)}
    per = [((X[i] - mu[s["want_label"]]) ** 2).mean().item() for i, s in enumerate(nxt)]
    res["oracle_next_probe16"] = dict(mean=float(np.mean(per)), per_sample=per,
                                      want_labels=[s["want_label"] for s in nxt])
    gm = allL.mean(0)
    res["oracle_uncond_next_probe16"] = float(np.mean([((X[i] - gm) ** 2).mean().item() for i in range(len(nxt))]))
    res["sec"] = round(time.time() - t0, 1)
    log(f"part1 done in {res['sec']}s: D_next {res['D_next_train']:.4f} (test {res['D_next_test']:.4f}), "
        f"D_uncond {res['D_uncond_train']:.4f}, oracle probe16 {res['oracle_next_probe16']['mean']:.4f}, "
        f"std/img {res['std_per_image_mean']:.3f}, posterior var {res['vae_posterior_var_mean']:.2e}")
    return res


def prepare_probe(pipe, n_seeds):
    enc = LatentEncoder(pipe.vae, DEV)
    samples = D.make_pairs("test", N_VAL)
    embeds = []
    with torch.no_grad():
        for s in samples:
            e, _, pad = pipe.encode_prompt(prompt=s["prompt"], image=[s["ref"].convert("RGBA")])
            embeds.append((e.to(DEV), pad.to(DEV)))
    tgt, _ = enc([s["target"] for s in samples])
    cond, _ = enc([s["ref"] for s in samples])
    tgt = [t.unsqueeze(0).to(DEV, BF16) for t in tgt]
    cond = [c.unsqueeze(0).to(DEV, BF16) for c in cond]
    # Noise draw j uses seeds 1234 + 100000*j + i; j = 0 is exactly the training probe's fixed noise.
    noise = [fixed_noise([x.shape for x in tgt], 1234 + 100_000 * j, DEV, BF16) for j in range(n_seeds)]
    return dict(samples=samples, embeds=embeds, tgt=tgt, cond=cond, noise=noise, enc=enc)


def summarize(per):
    out = {}
    for t, d in per.items():
        out[t] = {}
        for s, seeds in d.items():
            a = np.array(seeds)  # (n_seeds, n)
            out[t][s] = dict(seed0=float(a[0].mean()), mean=float(a.mean()),
                             sem=float(a.mean(0).std(ddof=1) / np.sqrt(a.shape[1])),
                             per_seed=[float(x) for x in a.mean(1)])
    return out


def cmd_measure(a):
    from diffusers import QwenImage21Pipeline
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(a.model, torch_dtype=BF16).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    log(f"pipeline loaded in {time.time() - t0:.0f}s; vae dtype {pipe.vae.dtype}, z_dim {pipe.vae.config.z_dim}")
    R = dict(sigmas=SIGMAS, n_seeds=a.n_seeds, n_val_per_task=N_VAL)
    R["part1"] = part1(pipe.vae, a.n_train_per_class, a.n_test_per_class)
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    P = prepare_probe(pipe, a.n_seeds)
    args = (P["samples"], P["embeds"], P["tgt"], P["cond"], P["noise"], SIGMAS)
    t1 = time.time()
    per_base = probe_loss(pipe.transformer.eval(), *args, device=DEV)
    R["probe_base"] = summarize(per_base)
    log(f"base probe {time.time() - t1:.0f}s: " + " ".join(
        f"{t}@0.9={R['probe_base'][t]['0.9']['seed0']:.4f}/@1={R['probe_base'][t]['1.0']['mean']:.4f}" for t in D.TASK_LIST))
    per_lora = None
    if a.lora:
        pipe.load_lora_weights(a.lora)
        t1 = time.time()
        per_lora = probe_loss(pipe.transformer.eval(), *args, device=DEV)
        R["probe_lora"] = summarize(per_lora)
        log(f"lora probe {time.time() - t1:.0f}s: " + " ".join(
            f"{t}@0.9={R['probe_lora'][t]['0.9']['seed0']:.4f}/@1={R['probe_lora'][t]['1.0']['mean']:.4f}" for t in D.TASK_LIST))
    R["sec_total"] = round(time.time() - t0, 1)
    R["versions"] = dict(torch=torch.__version__, diffusers=__import__("diffusers").__version__,
                         gpu=torch.cuda.get_device_name(0))
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    json.dump(dict(base=per_base, lora=per_lora), open(os.path.join(a.out, "per_sample.json"), "w"))
    log(f"wrote {a.out}/results.json")


@torch.no_grad()
def cmd_vis(a):
    from diffusers import QwenImage21Pipeline
    os.makedirs(a.out, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(a.model, torch_dtype=BF16).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    if a.lora:
        pipe.load_lora_weights(a.lora)
    sigmas = [float(x) for x in a.sigmas.split(",")]
    P = prepare_probe(pipe, 1 + a.extra)  # noise draw 0 = the training probe's noise; draws 1.. only for the extras
    enc, tr = P["enc"], pipe.transformer.eval()

    def save(name, lat):
        Image.fromarray(enc.decode(lat), "RGB").resize((a.px, a.px), Image.BICUBIC).save(
            os.path.join(a.out, name + ".png"), optimize=True)

    rec = {}
    for idx in [int(x) for x in a.ids.split(",")]:
        s = P["samples"][idx]
        task = s["task"]
        # Batch the whole task group exactly as the probe does, so the numbers match the probe's per-sample values.
        gi = [i for i, x in enumerate(P["samples"]) if x["task"] == task]
        j = gi.index(idx)
        x0 = torch.cat([P["tgt"][i] for i in gi])
        cond = torch.cat([P["cond"][i] for i in gi])
        emb = torch.cat([P["embeds"][i][0] for i in gi])
        pad = P["embeds"][gi[0]][1]
        noise = torch.cat([P["noise"][0][i] for i in gi])
        tag = f"{task}_{idx:02d}"
        save(f"{tag}_ref", P["cond"][idx])
        save(f"{tag}_x0", P["tgt"][idx])
        r = dict(idx=idx, task=task, prompt=s["prompt"], src_label=s["src_label"], want_label=s["want_label"], sigma={})
        for sg in sigmas:
            pred, xt = forward_v(tr, x0, cond, noise, emb, pad, sg, DEV)
            pred = pred.float()
            vl = ((pred - (noise - x0).float()) ** 2).reshape(len(gi), -1).mean(1)
            x0p = xt.float() - sg * pred
            x0e = ((x0p - x0.float()) ** 2).reshape(len(gi), -1).mean(1)
            r["sigma"][str(sg)] = dict(v_loss=vl[j].item(), x0_err=x0e[j].item(), sigma2_vloss=sg ** 2 * vl[j].item(),
                                       inv_sigma2=1 / sg ** 2, task_group_v_loss_seed0=vl.mean().item())
            save(f"{tag}_xt_{sg}", xt[j:j + 1])
            save(f"{tag}_x0hat_{sg}", x0p[j:j + 1])
            log(f"{tag} sigma {sg}: v-loss {vl[j].item():.4f} x0-err {x0e[j].item():.5f}")
        sg = sigmas[-1]
        r["extra"] = {}
        for k in range(1, 1 + a.extra):
            nz = torch.cat([P["noise"][k][i] for i in gi])
            pred, xt = forward_v(tr, x0, cond, nz, emb, pad, sg, DEV)
            x0p = xt.float() - sg * pred.float()
            x0e = ((x0p - x0.float()) ** 2).reshape(len(gi), -1).mean(1)
            r["extra"][f"{sg}_s{k}"] = dict(x0_err=x0e[j].item())
            save(f"{tag}_x0hat_{sg}_s{k}", x0p[j:j + 1])
            log(f"{tag} sigma {sg} draw {k}: x0-err {x0e[j].item():.5f}")
        rec[tag] = r
    json.dump(rec, open(os.path.join(a.out, "vis_numbers.json"), "w"), indent=1, ensure_ascii=False)
    log(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="Part 1 (D) + Part 2 (probe loss vs sigma)")
    m.add_argument("--model", required=True)
    m.add_argument("--lora", default=None, help="LoRA dir or .safetensors (optional)")
    m.add_argument("--out", default="outputs/probe")
    m.add_argument("--n-seeds", type=int, default=4)
    m.add_argument("--n-train-per-class", type=int, default=200)
    m.add_argument("--n-test-per-class", type=int, default=100)
    v = sub.add_parser("vis", help="x_t / x0' thumbnails across sigma for two samples")
    v.add_argument("--model", required=True)
    v.add_argument("--lora", default=None)
    v.add_argument("--out", default="outputs/probe/vis")
    v.add_argument("--ids", default="11,7", help="indices into make_pairs('test', 16): a rot90 and a next sample")
    v.add_argument("--px", type=int, default=160)
    v.add_argument("--sigmas", default="0.1,0.5,0.95,0.99,1.0", help="comma-separated sigmas (the deck uses the default)")
    v.add_argument("--extra", type=int, default=3, help="extra noise draws at the last sigma (x0' only)")
    a = ap.parse_args()
    cmd_measure(a) if a.cmd == "measure" else cmd_vis(a)


if __name__ == "__main__":
    main()
