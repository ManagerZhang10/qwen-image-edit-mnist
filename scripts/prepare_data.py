#!/usr/bin/env python3
"""第 1 步 · 准备训练数据：把 make_pairs('train', N) 写成 diffusers img2img LoRA 训练脚本能读的本地数据集。

讲义：第 2 页「四种编辑任务」（数据长什么样），第 3 页「训练一步怎么走」第 1 步（取一对样本）。
只用 CPU；MNIST 首次运行时自动下载到 --mnist-root（默认 $MNIST_ROOT 或 ./data）。

训练脚本调用 datasets.load_dataset(--dataset_name)，需要三列：
  --cond_image_column cond_image   参考图（被编辑的那张）
  --image_column      image        目标图
  --caption_column    caption      固定的中文编辑指令
一个装着 data/train-00000-of-00001.parquet（带 Image feature 元数据）的目录就够了。
另写 summary.json（样本数、任务、next 目标模式），scripts/train_lora.sh 会读其中的 next_target。

用法：
  python scripts/prepare_data.py --out data/mnist_edit_train                       # 实验 B：4 × 500 = 2000 对
  python scripts/prepare_data.py --out data/mnist_edit_proto --next-target proto   # 实验 F：next 用固定原型
  python scripts/prepare_data.py --out data/mnist_edit_next --tasks next           # 只训 next（实验 G 式）
  python scripts/prepare_data.py --out /tmp/qie_smoke --n-per-task 4               # 冒烟测试
"""
import argparse
import io
import json
import os

from qie_mnist import data as D


def png(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return {"bytes": b.getvalue(), "path": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--n-per-task", type=int, default=500, help="每个任务的样本对数（实验 B 用 500 → 共 2000 对）")
    ap.add_argument("--tasks", default="all",
                    help="逗号分隔的任务子集，如 next 或 rot90,invert；默认 all = rot90,rot180,next,invert")
    ap.add_argument("--next-target", default="random", choices=D.NEXT_TARGETS,
                    help="next 的目标：random = 另一个人写的 n+1（实验 B）；proto = 每个数字一张固定原型（实验 F）")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist-root", default=None, help="MNIST 下载 / 缓存目录（默认 $MNIST_ROOT 或 ./data）")
    a = ap.parse_args()
    tasks = D.parse_tasks(a.tasks)

    # 直接用 pyarrow 写（外加标记两列图像的 `huggingface` schema 元数据）：Dataset.from_dict 会用 dill 给内存表算指纹，
    # 在部分 datasets / pyarrow / python 组合下报错。load_dataset() 读回来时两列就是 Image feature。
    import pyarrow as pa
    import pyarrow.parquet as pq
    pairs = D.make_pairs(a.split, a.n_per_task, seed=a.seed, root=a.mnist_root, tasks=tasks,
                         next_target=a.next_target)
    img_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    feats = {"cond_image": {"_type": "Image"}, "image": {"_type": "Image"},
             "caption": {"dtype": "string", "_type": "Value"}, "task": {"dtype": "string", "_type": "Value"},
             "src_label": {"dtype": "int32", "_type": "Value"}, "want_label": {"dtype": "int32", "_type": "Value"}}
    schema = pa.schema([("cond_image", img_t), ("image", img_t), ("caption", pa.string()), ("task", pa.string()),
                        ("src_label", pa.int32()), ("want_label", pa.int32())],
                       metadata={"huggingface": json.dumps({"info": {"features": feats}})})
    table = pa.table({
        "cond_image": [png(p["ref"]) for p in pairs],
        "image": [png(p["target"]) for p in pairs],
        "caption": [p["prompt"] for p in pairs],
        "task": [p["task"] for p in pairs],
        "src_label": [p["src_label"] for p in pairs],
        "want_label": [p["want_label"] for p in pairs],
    }, schema=schema)
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    pq.write_table(table, os.path.join(a.out, "data", "train-00000-of-00001.parquet"))
    counts = {t: sum(p["task"] == t for p in pairs) for t in tasks}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"n": len(pairs), "split": a.split, "seed": a.seed, "n_per_task": a.n_per_task, "tasks": tasks,
                   "next_target": a.next_target, "per_task": counts, "res": D.RES, "prompts": D.TASKS},
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {len(pairs)} pairs {counts} (next_target={a.next_target}) -> {a.out}")


if __name__ == "__main__":
    main()
