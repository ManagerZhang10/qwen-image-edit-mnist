#!/usr/bin/env python3
"""Step 1 - training data: write make_pairs('train', N) as the local dataset the diffusers img2img LoRA trainer reads.

Deck: p.2 (Four edit tasks) for what the data looks like, p.3 (One training step) step 1 (take a pair).
CPU only; MNIST is downloaded on first use to --mnist-root (default $MNIST_ROOT or ./data).

The trainer calls datasets.load_dataset(--dataset_name) and needs three columns:
  --cond_image_column cond_image   reference image (the one being edited)
  --image_column      image        target image
  --caption_column    caption      the fixed Chinese edit instruction
A folder holding data/train-00000-of-00001.parquet (with Image feature metadata) is enough.
summary.json records the counts, tasks and next-target mode; scripts/train_lora.sh reads its next_target.

Usage:
  python scripts/prepare_data.py --out data/mnist_edit_train                       # experiment B: 4 x 500 = 2000 pairs
  python scripts/prepare_data.py --out data/mnist_edit_proto --next-target proto   # experiment F: fixed next prototypes
  python scripts/prepare_data.py --out data/mnist_edit_next --tasks next           # train on next only (task subset)
  python scripts/prepare_data.py --out /tmp/qie_smoke --n-per-task 4               # smoke test
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
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n-per-task", type=int, default=500, help="pairs per task (experiment B used 500 -> 2000 pairs)")
    ap.add_argument("--tasks", default="all",
                    help="comma-separated task subset, e.g. next or rot90,invert; default all = rot90,rot180,next,invert")
    ap.add_argument("--next-target", default="random", choices=D.NEXT_TARGETS,
                    help="next target: random = another writer's n+1 (experiment B); proto = one fixed prototype per digit (experiment F)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist-root", default=None, help="MNIST download/cache dir (default: $MNIST_ROOT or ./data)")
    a = ap.parse_args()
    tasks = D.parse_tasks(a.tasks)

    # Written with pyarrow directly (plus the `huggingface` schema metadata that marks the two image columns),
    # because Dataset.from_dict fingerprints the in-memory table with dill, which fails on some
    # datasets/pyarrow/python combinations. load_dataset() reads the file back with Image features.
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
