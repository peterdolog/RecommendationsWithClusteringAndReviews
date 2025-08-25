#!/usr/bin/env python3
"""
Attach IDs to embeddings exactly like legacy add_ids_to_embeddings.py.

Behavior:
- Read user/item embeddings CSVs (with RecBole extra row at [0]) and drop first row.
- Read user_ids.txt and item_ids.txt.
- Assert equal lengths, then write CSVs with first column as id and no header, no index.
- Keep delimiter and numeric formatting as pandas default (legacy used default).
"""

import os
import numpy as np
import pandas as pd

# -------------------- Config (edit to match your setup) --------------------
EMBEDDINGS_DIR = "CGCL-Pytorch-master/embeddings" # <--- ATTENTION: replace with your embeddings directory
IDS_DIR = "data"
DATASET = "yelp2018"  # example default matching legacy usage
# ---------------------------------------------------------------


def attach_ids_to_embeddings(emb: np.ndarray, ids, out_csv: str, id_name: str) -> None:
    df = pd.DataFrame(emb)
    df.insert(0, id_name, ids)
    df.to_csv(out_csv, index=False, header=False)


if __name__ == "__main__":
    dataset = DATASET
    emb_folder = os.path.join(EMBEDDINGS_DIR, dataset)
    ids_folder = os.path.join(IDS_DIR, dataset)

    user_emb_file = os.path.join(emb_folder, "user_embeddings.csv")
    item_emb_file = os.path.join(emb_folder, "item_embeddings.csv")
    user_ids_file = os.path.join(ids_folder, "user_ids.txt")
    item_ids_file = os.path.join(ids_folder, "item_ids.txt")

    if not (os.path.exists(user_emb_file) and os.path.exists(item_emb_file)
            and os.path.exists(user_ids_file) and os.path.exists(item_ids_file)):
        print(f"⚠️ Skipping {dataset}: Missing files")
    else:
        print(f"🔄 Processing {dataset}...")
        with open(user_ids_file, "r") as f:
            user_ids = [int(line.strip()) for line in f]
        with open(item_ids_file, "r") as f:
            item_ids = [int(line.strip()) for line in f]

        user_emb = np.loadtxt(user_emb_file, delimiter=",")
        item_emb = np.loadtxt(item_emb_file, delimiter=",")

        user_emb = user_emb[1:]
        item_emb = item_emb[1:]

        print("len(user_embeddings) == len(user_ids): ", len(user_emb), "==", len(user_ids))
        print("len(item_embeddings) == len(item_ids): ", len(item_emb), "==", len(item_ids))

        assert len(user_emb) == len(user_ids), f"🚨 Mismatch in user embeddings for {dataset}!"
        assert len(item_emb) == len(item_ids), f"🚨 Mismatch in item embeddings for {dataset}!"

        user_out = os.path.join(emb_folder, "user_embeddings_with_ids.csv")
        item_out = os.path.join(emb_folder, "item_embeddings_with_ids.csv")

        attach_ids_to_embeddings(user_emb, user_ids, user_out, "user_id")
        attach_ids_to_embeddings(item_emb, item_ids, item_out, "item_id")

        print(f"✅ Saved corrected embeddings for {dataset}")
        print("🎉 All datasets processed!") 