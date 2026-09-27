"""MNIST 编辑任务的数据定义：四种编辑、固定指令，以及（参考图, 目标图）样本对的生成。

对应讲义第 2 页「四种编辑任务」。所有实验（A–F）共用这一份定义，训练集和测试集都从这里来。

四种编辑（目标图完全由代码决定；next 由种子决定）：
  rot90   顺时针旋转 90°
  rot180  旋转 180°
  next    换成下一个数字（9 → 0）。默认目标是同一 split 里「另一个人写的」n+1（答案不唯一）；
          next_target="proto" 时改为每个数字一张固定原型（实验 F，见 next_prototypes）
  invert  黑白反色

每个任务只有一条固定中文指令（TASKS），没有改写或同义扩充：够做玩具实验，不够做真实编辑数据。
图像：28×28 灰度 → 双三次放大到 RES（默认 512，需为 32 的倍数）→ RGB。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

RES = int(os.environ.get("QIE_RES", 512))
TASKS = {
    "rot90": "把图片顺时针旋转 90 度",
    "rot180": "把图片旋转 180 度",
    "next": "把数字换成下一个数字，9 变成 0",
    "invert": "把图片黑白反色",
}
TASK_LIST = list(TASKS)
NEXT_TARGETS = ("random", "proto")

_MNIST_CACHE: dict[str, tuple] = {}
_PROTO_CACHE: dict[tuple, list] = {}


def default_data_root() -> str:
    """MNIST 下载 / 缓存目录：$MNIST_ROOT，否则是当前工作目录下的 ./data。"""
    return os.environ.get("MNIST_ROOT", str(Path.cwd() / "data"))


def load_mnist(root: str | None = None):
    """返回 (x_train, y_train, x_test, y_test)，图像为 uint8 [N,28,28]，标签为 int。"""
    root = root or default_data_root()
    if root not in _MNIST_CACHE:
        import torchvision
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        _MNIST_CACHE[root] = (tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy())
    return _MNIST_CACHE[root]


def next_prototypes(root: str | None = None, n_per_class: int = 2000):
    """实验 F 的 next 固定目标：每个数字取训练集前 n_per_class 张里的 medoid（到同类其余样本平均距离最小的那张）。

    目标从此由条件唯一决定，「另一个人写的 n+1」带来的方差 D 变成 0。训练集和测试集共用这 10 张。
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
    """img28：uint8 [28,28]（MNIST 原图，黑底白字）。返回 (目标图 uint8 [28,28], 期望标签)。"""
    if task == "rot90":
        return np.rot90(img28, k=-1).copy(), label
    if task == "rot180":
        return np.rot90(img28, k=2).copy(), label
    if task == "invert":
        return (255 - img28).copy(), label
    if task == "next":
        want = (label + 1) % 10
        idx = np.flatnonzero(pool_labels == want)
        pick = rng.choice(idx)  # 两种模式都消耗一次随机数，保证其余任务抽到的参考图和默认模式完全相同
        if next_target == "proto":
            return next_prototypes(root)[want].copy(), want
        return pool_imgs[pick].copy(), want
    raise KeyError(task)


def to_pil(img28, res=RES):
    return Image.fromarray(img28, "L").resize((res, res), Image.BICUBIC).convert("RGB")


def from_pil(im):
    return np.asarray(im.convert("L").resize((28, 28), Image.BILINEAR))


def parse_tasks(spec: str | None):
    """命令行的任务列表："rot90,next" → ["rot90", "next"]（按给定顺序）；"all" 或空 → 四个任务。"""
    if not spec or spec == "all":
        return list(TASK_LIST)
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        raise ValueError(f"unknown task(s) {bad}; choose from {TASK_LIST}")
    return tasks


def make_pairs(split="train", n_per_task=500, seed=0, root: str | None = None, tasks=None, next_target="random"):
    """返回 list[dict]，键为 task, prompt, ref (PIL), target (PIL), ref28, tgt28, src_label, want_label。

    给定 (split, n_per_task, seed, tasks, next_target) 时结果确定；train / test 用 MNIST 各自的 split，互不重叠。
    tasks：只生成这几种编辑（默认四种，实验 G 式的「只训部分任务」用）。注意随机数按任务顺序依次消耗，
           所以只选部分任务时，抽到的参考图不同于四任务版本里的同名任务。测试集一律用默认值。
    next_target："random"（默认，另一个人写的 n+1）或 "proto"（固定原型，实验 F）。
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
