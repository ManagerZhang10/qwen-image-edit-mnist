"""评测：编辑成功率和像素 IoU（讲义第 12 页「成功率怎么算」）。训练验证、推理扫描、画图都用这一份。

Scoring for the MNIST editing tasks.

The edited output is resized back to 28x28 grayscale, the edit is undone (rotate back / invert back) and a
small CNN classifies the result. Success = predicted label equals the expected label. Deterministic tasks
(rot90, rot180, invert) also get the pixel IoU against the ground-truth target; `next` reports how often the
output was classified as the *source* digit (copied_source), i.e. the model returned the input unchanged.

The classifier weights ship with the package (assets/mnist_cls.pt, ~1.6 MB, trained from scratch on the MNIST
train split by scripts/train_classifier.py, ~99% test accuracy). Override the path with $QIE_CLS.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data import TASK_LIST, from_pil

DEFAULT_CLS = Path(__file__).resolve().parent / "assets" / "mnist_cls.pt"
_CLS = None


def classifier_path() -> str:
    return os.environ.get("QIE_CLS", str(DEFAULT_CLS))


def build_net():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))


def _classifier():
    global _CLS
    if _CLS is None:
        import torch
        m = build_net()
        m.load_state_dict(torch.load(classifier_path(), map_location="cpu"))
        _CLS = m.eval()
    return _CLS


def undo(task, out28):
    """Undo the edit so the digit is upright and white-on-black again before classification."""
    if task == "rot90":
        return np.rot90(out28, k=1).copy()
    if task == "rot180":
        return np.rot90(out28, k=2).copy()
    if task == "invert":
        return (255 - out28).copy()
    return out28


def classify(imgs28):
    import torch
    x = torch.tensor(np.stack(imgs28), dtype=torch.float32).div(255).unsqueeze(1)
    with torch.no_grad():
        return _classifier()(x).argmax(1).numpy()


def iou(a28, b28, thr=128):
    a, b = a28 >= thr, b28 >= thr
    return float((a & b).sum() / max((a | b).sum(), 1))


def evaluate(samples, outputs):
    """samples: elements of make_pairs(); outputs: matching PIL outputs. Returns a per-task summary dict."""
    out28 = [from_pil(o) for o in outputs]
    pred = classify([undo(s["task"], o) for s, o in zip(samples, out28)])
    res = {}
    for task in TASK_LIST:
        ids = [i for i, s in enumerate(samples) if s["task"] == task]
        if not ids:
            continue
        ok = [pred[i] == samples[i]["want_label"] for i in ids]
        r = {"n": len(ids), "success": float(np.mean(ok))}
        if task != "next":
            r["iou"] = float(np.mean([iou(out28[i], samples[i]["tgt28"]) for i in ids]))
        else:
            r["copied_source"] = float(np.mean([pred[i] == samples[i]["src_label"] for i in ids]))
        res[task] = r
    return res
