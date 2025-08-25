#!/usr/bin/env python3
"""
Methods to Compute review sentence embeddings.

Behavior:
- Default model: sentence-transformers/all-MiniLM-L6-v2
- Batch by step_size and optionally save intermediate joblib chunks.
- Return stacked numpy array; assert row count equals input length.
"""

import os
from typing import List

import torch
import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


def texts_to_sentence_rep(
        texts: List[str], save_dir: str = None, step_size: int = 1000,
        sentence_bert_version: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> np.ndarray:
    model = SentenceTransformer(sentence_bert_version)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    full_encoded = []
    with torch.no_grad():
        for i in tqdm(range(0, int(len(texts) / step_size) + 1)):
            start_id = i * step_size
            end_id = start_id + step_size
            inp = texts[start_id: end_id]
            encoded = model.encode(inp)
            if save_dir is not None:
                joblib.dump(encoded, os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            elif encoded.shape[0] > 0:
                full_encoded.append(encoded)

    if save_dir is not None:
        for i in tqdm(range(0, int(len(texts) / step_size) + 1)):
            start_id = i * step_size
            encoded = joblib.load(os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            if encoded.shape[0] > 0:
                full_encoded.append(encoded)

    full_encoded = np.vstack(full_encoded)
    assert full_encoded.shape[0] == len(texts)
    return full_encoded


def df_to_sentence_rep(
        df: pd.DataFrame, column: str = "review", save_dir: str = None, step_size: int = 1000,
        sentence_bert_version: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> np.ndarray:
    model = SentenceTransformer(sentence_bert_version)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    full_encoded = []
    with torch.no_grad():
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):
            start_id = i * step_size
            end_id = start_id + step_size
            inp = df[start_id: end_id][column].tolist()
            encoded = model.encode(inp)
            if save_dir is not None:
                joblib.dump(encoded, os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            elif encoded.shape[0] > 0:
                full_encoded.append(encoded)

    if save_dir is not None:
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):
            start_id = i * step_size
            encoded = joblib.load(os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            if encoded.shape[0] > 0:
                full_encoded.append(encoded)

    full_encoded = np.vstack(full_encoded)
    return full_encoded


if __name__ == "__main__":
    print("This script provides functions to compute sentence embeddings, which were applied to the reviews of each dataset.\n"
          "Import and call texts_to_sentence_rep or df_to_sentence_rep to use.") 