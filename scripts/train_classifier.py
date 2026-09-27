#!/usr/bin/env python3
"""评测用的小 MNIST 分类器（讲义第 12 页「成功率怎么算」里的分类器，测试集准确率 98.8%）。CPU 约 1–2 分钟。

仓库已附带训练好的 src/qie_mnist/assets/mnist_cls.pt，只有想重新生成时才需要跑。

Usage: python scripts/train_classifier.py [--out src/qie_mnist/assets/mnist_cls.pt] [--mnist-root data]
"""
import argparse
import os

import torch
import torch.nn.functional as F

from qie_mnist import data as D
from qie_mnist import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(E.DEFAULT_CLS))
    ap.add_argument("--mnist-root", default=None)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()

    torch.manual_seed(0)
    xtr, ytr, xte, yte = D.load_mnist(a.mnist_root)
    X = torch.tensor(xtr, dtype=torch.float32).div(255).unsqueeze(1)
    Y = torch.tensor(ytr)
    m = E.build_net()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(a.steps):
        i = torch.randint(0, len(X), (128,))
        loss = F.cross_entropy(m(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        Xt = torch.tensor(xte, dtype=torch.float32).div(255).unsqueeze(1)
        acc = (m(Xt).argmax(1).numpy() == yte).mean()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(m.state_dict(), a.out)
    print(f"classifier test acc {acc:.4f} -> {a.out}")
    # Self-check: the ground-truth targets themselves should score close to 100%.
    os.environ["QIE_CLS"] = a.out
    E._CLS = None
    s = D.make_pairs("test", 50, root=a.mnist_root)
    print(E.evaluate(s, [x["target"] for x in s]))


if __name__ == "__main__":
    main()
