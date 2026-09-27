"""MNIST editing tasks: task definitions, fixed instructions and (reference, target) pair synthesis.

Deck p.2 (Four edit tasks). Every experiment (A-F) uses this module for both the train and the test pairs.

Four edits (the target is fully determined by code; for `next`, by the seed):
  rot90   rotate 90 degrees clockwise
  rot180  rotate 180 degrees
  next    replace the digit with the next one (9 -> 0). By default the target is a *different* writer's n+1
          from the same split (no unique answer); with next_target="proto" it is one fixed prototype per digit
          (experiment F, see next_prototypes)
  invert  black/white inversion

Every task uses ONE fixed Chinese instruction (TASKS), with no paraphrasing or prompt augmentation: enough for a
toy task, not for real editing data.
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
NEXT_TARGETS = ("random", "proto")

_MNIST_CACHE: dict[str, tuple] = {}
_PROTO_CACHE: dict[tuple, list] = {}


def default_data_root() -> str:
    """MNIST download/cache dir: $MNIST_ROOT, else ./data under the current working directory."""
    return os.environ.get("MNIST_ROOT", str(Path.cwd() / "data"))


def load_mnist(root: str | None = None):
    """Return (x_train, y_train, x_test, y_test): uint8 images [N,28,28] and int labels."""
    root = root or default_data_root()
    if root not in _MNIST_CACHE:
        import torchvision
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        _MNIST_CACHE[root] = (tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy())
    return _MNIST_CACHE[root]


def next_prototypes(root: str | None = None, n_per_class: int = 2000):
    """Fixed `next` targets for experiment F: per digit, the medoid of the first n_per_class train images of that class
    (the image with the smallest mean distance to the others).

    The target is then fully determined by the condition, so the variance D from "another writer's n+1" becomes 0.
    The train and test splits share these 10 images.
    """
    key = (root or default_data_root(), n_per_class)
    if key not in _PROTO_CACHE:
        xtr, ytr, _, _ = load_mnist(root)
        protos = []
        for c in range(10):
            X = xtr[ytr == c][:n_per_class].reshape(-1, 784).astype(np.float32) / 255
            sq = (X ** 2).sum(1)
            d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
            protos.append(xtr[ytr == c][int(d.mean(1).argmin())])
        _PROTO_CACHE[key] = protos
    return _PROTO_CACHE[key]


def apply_edit(task, img28, label, pool_imgs, pool_labels, rng, next_target="random", root=None):
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
        pick = rng.choice(idx)  # consumed in both modes, so every other sample is identical to the default data
        if next_target == "proto":
            return next_prototypes(root)[want].copy(), want
        return pool_imgs[pick].copy(), want
    raise KeyError(task)


def to_pil(img28, res=RES):
    return Image.fromarray(img28, "L").resize((res, res), Image.BICUBIC).convert("RGB")


def from_pil(im):
    return np.asarray(im.convert("L").resize((28, 28), Image.BILINEAR))


def parse_tasks(spec: str | None):
    """CLI task list: "rot90,next" -> ["rot90", "next"] (in the given order); "all" or empty -> all four tasks."""
    if not spec or spec == "all":
        return list(TASK_LIST)
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        raise ValueError(f"unknown task(s) {bad}; choose from {TASK_LIST}")
    return tasks


def make_pairs(split="train", n_per_task=500, seed=0, root: str | None = None, tasks=None, next_target="random"):
    """Return list[dict] with keys: task, prompt, ref (PIL), target (PIL), ref28, tgt28, src_label, want_label.

    Deterministic for a given (split, n_per_task, seed, tasks, next_target). The train and test splits use disjoint
    MNIST splits.
    tasks: only generate these edits (default: all four; for training on a task subset). The random draws are
           consumed task by task, so a subset gets different reference images than the same task in the four-task
           data. Always keep the default for the test set.
    next_target: "random" (default, another writer's n+1) or "proto" (fixed prototype, experiment F).
    """
    if next_target not in NEXT_TARGETS:
        raise ValueError(f"next_target must be one of {NEXT_TARGETS}")
    xtr, ytr, xte, yte = load_mnist(root)
    imgs, labels = (xtr, ytr) if split == "train" else (xte, yte)
    rng = np.random.default_rng(seed + (0 if split == "train" else 10_000))
    out = []
    for task in (tasks or TASK_LIST):
        for i in rng.choice(len(imgs), n_per_task, replace=False):
            tgt, want = apply_edit(task, imgs[i], int(labels[i]), imgs, labels, rng, next_target, root)
            out.append(dict(task=task, prompt=TASKS[task], ref28=imgs[i], tgt28=tgt,
                            src_label=int(labels[i]), want_label=int(want),
                            ref=to_pil(imgs[i]), target=to_pil(tgt)))
    random.Random(seed).shuffle(out)
    return out
