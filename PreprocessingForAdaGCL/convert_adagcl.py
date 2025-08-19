#!/usr/bin/env python3
"""
Write AdaGCL PKLs exactly like legacy.

Behavior:
- Build sparse matrices with same axis convention (users=rows, items=cols).
- Use same ID mapping policy as legacy (integers from raw tokens).
- Use same dtype for data (np.float64 as in legacy convert_kgcl_datasets_to_pkl.py).
- Filenames exactly: trnMat.pkl, tstMat.pkl (or follow legacy naming per dataset).
- Output dir mirrors legacy (default AdaGCL/Datasets/<dataset>/; fallback to data/processed/adagcl/<dataset>/ if desired).
- Print minimal progress.
"""

import os
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Any

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
DATA_ROOT = "../data"
OUT_ROOT = "AdaGCL/Datasets"
# ---------------------------------------------------------------


def load_kgcl_data(path: str) -> Tuple[List[Tuple[int, int]], int, int]:
    """Load KGCL-format file into list of (user,item) interactions and max IDs as integers."""
    interactions: List[Tuple[int, int]] = []
    max_user = 0
    max_item = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            items = list(map(int, parts[1:]))
            interactions.extend((user, it) for it in items)
            if user > max_user:
                max_user = user
            if items:
                mi = max(items)
                if mi > max_item:
                    max_item = mi
    return interactions, max_user, max_item


def build_sparse_matrix(train_path: str, test_path: str) -> Dict[str, Any]:
    """Build legacy-compatible sparse matrices and shapes."""
    trn, u_trn, i_trn = load_kgcl_data(train_path)
    tst, u_tst, i_tst = load_kgcl_data(test_path)
    n_users = max(u_trn, u_tst) + 1
    n_items = max(i_trn, i_tst) + 1
    
    def _to_coo(pairs: List[Tuple[int, int]]):
        row = [u for u, _ in pairs]
        col = [i for _, i in pairs]
        data = np.ones(len(pairs), dtype=np.float64)
        return coo_matrix((data, (row, col)), shape=(n_users, n_items))
    
    trn_mat = _to_coo(trn)
    tst_mat = _to_coo(tst)
    return {
        "trn": trn_mat,
        "tst": tst_mat,
        "n_users": n_users,
        "n_items": n_items,
    }


def save_pkls_like_legacy(out_dir: str, trn_mat, tst_mat) -> None:
    """Save PKLs exactly as legacy did."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trnMat.pkl"), "wb") as f:
        pickle.dump(trn_mat, f)
    with open(os.path.join(out_dir, "tstMat.pkl"), "wb") as f:
        pickle.dump(tst_mat, f)


def _dataset_io_paths(dataset_name: str):
    train_path = os.path.join(DATA_ROOT, dataset_name, "train.txt")
    test_path = os.path.join(DATA_ROOT, dataset_name, "test.txt")
    out_dir = os.path.join(OUT_ROOT, dataset_name)
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(DATA_ROOT, "processed", "adagcl", dataset_name)
    return train_path, test_path, out_dir


if __name__ == "__main__":
    for ds in DATASETS:
        trn_fp, tst_fp, out_dir = _dataset_io_paths(ds)
        if not os.path.exists(trn_fp) or not os.path.exists(tst_fp):
            print(f"🚨 Skipping {ds}: train or test file not found.")
            continue
        print(f"🔄 Processing {ds}...")
        built = build_sparse_matrix(trn_fp, tst_fp)
        save_pkls_like_legacy(out_dir, built["trn"], built["tst"])
        print(f"✅ Saved: {out_dir}/trnMat.pkl and {out_dir}/tstMat.pkl") 