"""MNIST editing tasks: task definitions and (reference, target) pair synthesis.

Four edits. The target is fully determined by code (for `next`, by the seed):

  rot90   rotate 90 degrees clockwise
  rot180  rotate 180 degrees
  next    replace the digit with the next one (9 -> 0); the target is a *different* writer's n+1
  invert  black/white inversion

Every task uses ONE fixed Chinese instruction (see TASKS). There is no paraphrasing or prompt
augmentation; that is enough for a toy task but not for real editing data.

Images: 28x28 grayscale -> bicubic upsample to RES (default 512, must be a multiple of 32) -> RGB.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

RES = int(os.environ.get("QIE_RES", 512))
TASKS = {
    "rot90": "把图片顺时针旋转 90 度",   # "Rotate the image 90 degrees clockwise"
    "rot180": "把图片旋转 180 度",       # "Rotate the image 180 degrees"
    "next": "把数字换成下一个数字，9 变成 0",  # "Replace the digit with the next digit; 9 becomes 0"
    "invert": "把图片黑白反色",          # "Invert black and white"
}
TASK_LIST = list(TASKS)

_MNIST_CACHE: dict[str, tuple] = {}


def default_data_root() -> str:
    """MNIST download/cache dir: $MNIST_ROOT, else ./data under the current working directory."""
    return os.environ.get("MNIST_ROOT", str(Path.cwd() / "data"))


def load_mnist(root: str | None = None):
    """Return (x_train, y_train, x_test, y_test) as numpy arrays (uint8 images, int labels)."""
    root = root or default_data_root()
    if root not in _MNIST_CACHE:
        import torchvision
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        _MNIST_CACHE[root] = (tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy())
    return _MNIST_CACHE[root]


def apply_edit(task, img28, label, pool_imgs, pool_labels, rng):
    """img28: uint8 [28,28] (raw MNIST: white digit on black). Returns (target uint8 [28,28], expected label)."""
    if task == "rot90":
        return np.rot90(img28, k=-1).copy(), label
    if task == "rot180":
        return np.rot90(img28, k=2).copy(), label
    if task == "invert":
        return (255 - img28).copy(), label
    if task == "next":
        want = (label + 1) % 10
        idx = np.flatnonzero(pool_labels == want)
        return pool_imgs[rng.choice(idx)].copy(), want
    raise KeyError(task)


def to_pil(img28, res=RES):
    return Image.fromarray(img28, "L").resize((res, res), Image.BICUBIC).convert("RGB")


def from_pil(im):
    return np.asarray(im.convert("L").resize((28, 28), Image.BILINEAR))


def make_pairs(split="train", n_per_task=500, seed=0, root: str | None = None):
    """Return list[dict] with keys: task, prompt, ref (PIL), target (PIL), ref28, tgt28, src_label, want_label.

    Deterministic for a given (split, n_per_task, seed). The train and test splits use disjoint MNIST splits.
    """
    xtr, ytr, xte, yte = load_mnist(root)
    imgs, labels = (xtr, ytr) if split == "train" else (xte, yte)
    rng = np.random.default_rng(seed + (0 if split == "train" else 10_000))
    out = []
    for task in TASK_LIST:
        for i in rng.choice(len(imgs), n_per_task, replace=False):
            tgt, want = apply_edit(task, imgs[i], int(labels[i]), imgs, labels, rng)
            out.append(dict(task=task, prompt=TASKS[task], ref28=imgs[i], tgt28=tgt,
                            src_label=int(labels[i]), want_label=int(want),
                            ref=to_pil(imgs[i]), target=to_pil(tgt)))
    random.Random(seed).shuffle(out)
    return out
