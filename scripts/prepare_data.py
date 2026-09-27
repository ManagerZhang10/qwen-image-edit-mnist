#!/usr/bin/env python3
"""Write make_pairs('train', N) as the local dataset the diffusers img2img LoRA trainer reads.

The trainer calls datasets.load_dataset(--dataset_name) and needs three columns:
  --cond_image_column cond_image   (reference image, the one being edited)
  --image_column      image        (target image)
  --caption_column    caption      (the fixed Chinese edit instruction)
A folder holding data/train-00000-of-00001.parquet with Image features is enough.

Usage:
  python scripts/prepare_data.py --out data/mnist_edit_train --n-per-task 500
  python scripts/prepare_data.py --out /tmp/smoke --n-per-task 4 --mnist-root data
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
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist-root", default=None, help="MNIST download/cache dir (default: $MNIST_ROOT or ./data)")
    a = ap.parse_args()

    # Written with pyarrow directly (plus the `huggingface` schema metadata that marks the two image columns),
    # because Dataset.from_dict fingerprints the in-memory table with dill, which fails on some
    # datasets/pyarrow/python combinations. load_dataset() reads the file back with Image features.
    import pyarrow as pa
    import pyarrow.parquet as pq
    pairs = D.make_pairs(a.split, a.n_per_task, seed=a.seed, root=a.mnist_root)
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
    counts = {t: sum(p["task"] == t for p in pairs) for t in D.TASK_LIST}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"n": len(pairs), "split": a.split, "seed": a.seed, "per_task": counts, "res": D.RES,
                   "prompts": D.TASKS}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(pairs)} pairs {counts} -> {a.out}")


if __name__ == "__main__":
    main()
