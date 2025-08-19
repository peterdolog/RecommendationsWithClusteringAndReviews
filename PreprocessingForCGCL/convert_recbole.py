#!/usr/bin/env python3
"""
Write RecBole train.inter / test.inter exactly like legacy.

Behavior:
- Parse each line as: user item1 item2 ... and emit (user,item) pairs in order.
- Keep tokens as strings (no remap).
- Write TSVs with headers exactly: user_id:token, item_id:token.
- Filenames exactly: train.inter and test.inter.
- No dedup, preserve row order.
- Output dir: if CGCL-Pytorch-master/dataset/<dataset>/ exists, write there; otherwise fallback to data/processed/recbole/<dataset>/.
- Prints the same messages as legacy for processing and saved files.
"""

import os
import pandas as pd
from typing import List

# -------------------- Config (edit if needed) --------------------
DATASETS = [
    "yelp2018",
    "amazon-book",
    "mindreader_fold_0",
    "mindreader_fold_1",
    "mindreader_fold_2",
    "mindreader_fold_3",
    "mindreader_fold_4",
]
DATA_ROOT = "data"
CGCL_DATASET_ROOT = "CGCL-Pytorch-master/dataset"
# ---------------------------------------------------------------


def process_file(file_path: str) -> List[List[str]]:
    """Parse KGCL-format split file into list of [user_id, item_id] pairs as strings."""
    interactions: List[List[str]] = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_ids = parts[1:]
            for item_id in item_ids:
                interactions.append([user_id, item_id])
    return interactions


def write_split_inter(train_pairs, test_pairs, out_dir: str) -> None:
    """Write train/test .inter files with exact headers and format used by RecBole."""
    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.DataFrame(train_pairs, columns=["user_id:token", "item_id:token"])
    test_df = pd.DataFrame(test_pairs, columns=["user_id:token", "item_id:token"])
    train_df.to_csv(os.path.join(out_dir, "train.inter"), sep="\t", index=False)
    test_df.to_csv(os.path.join(out_dir, "test.inter"), sep="\t", index=False)


def _dataset_io_paths(dataset_name: str):
    train_path = os.path.join(DATA_ROOT, dataset_name, "train.txt")
    test_path = os.path.join(DATA_ROOT, dataset_name, "test.txt")
    if os.path.isdir(os.path.join(CGCL_DATASET_ROOT, dataset_name)):
        out_dir = os.path.join(CGCL_DATASET_ROOT, dataset_name)
    else:
        out_dir = os.path.join(DATA_ROOT, "processed", "recbole", dataset_name)
    return train_path, test_path, out_dir


if __name__ == "__main__":
    any_processed = False
    for ds in DATASETS:
        train_fp, test_fp, out_dir = _dataset_io_paths(ds)
        if not os.path.exists(train_fp) or not os.path.exists(test_fp):
            print(f"🚨 Skipping {ds}: train or test file not found.")
            continue
        print(f"🔄 Processing {ds}...")
        train_pairs = process_file(train_fp)
        test_pairs = process_file(test_fp)
        write_split_inter(train_pairs, test_pairs, out_dir)
        print(f"✅ Saved: {out_dir}train.inter and test.inter")
        any_processed = True
    if any_processed:
        print("🎉 All datasets processed successfully!")
    else:
        print("⚠️ No datasets were processed.") 