import json, numpy as np, pandas as pd
from pathlib import Path
import torch

ARTIFACT_DIR = Path("artifacts") / "model_training"
PREPROCESS_DIR    = Path("artifacts") / "data_preprocessing"
EMBED_DIR    = Path("artifacts") / "data_embedding"

with open(PREPROCESS_DIR / "vocabs.json") as f:
    VOCABS = json.load(f)
with open(EMBED_DIR / "embed_schema.json") as f:
    SCHEMA = json.load(f)
with open(ARTIFACT_DIR / "numeric_stats.json") as f:
    NUM_STATS = json.load(f)
with open(ARTIFACT_DIR / "dummy_cols.json") as f:
    DUMMY_COLS = json.load(f)['dummies']

EMBED_TABLES = {
    name: np.load(EMBED_DIR / "embed_matrices" / f"{name}.npy")
    for name in VOCABS.keys()
}

def z_score(col, val):           
    mu, sd = NUM_STATS[col]["mean"], NUM_STATS[col]["std"]
    return (val - mu) / (sd or 1.0)

def embed_special(col_name: str, raw_val: str) -> np.ndarray:
    idx = VOCABS[col_name].get(str(raw_val), 0)
    return EMBED_TABLES[col_name][idx]        

def make_feature_vector(raw: dict) -> np.ndarray:
    """
    input : dict from API caller .
    returns 1-D float32 vector that matches training order.
    """
    num_parts = [
        z_score("apr", float(raw.get("apr", 0))),
        z_score("nPaidOff", float(raw.get("nPaidOff", 0))),
        z_score("loanAmount", float(raw.get("loanAmount", 0))),
        z_score("originallyScheduledPaymentAmount",
                float(raw.get("originallyScheduledPaymentAmount", 0))),
    ]

    dummy_vec = [0.0] * len(DUMMY_COLS)
    cat_prefix_pairs = {
        "payFrequency_": raw.get("payFrequency", ""),
        "state_":        raw.get("state",  ""),
        "leadType_":     raw.get("leadType",  ""),
        "fpStatus_":     raw.get("fpStatus",  ""),
    }
    for prefix, value in cat_prefix_pairs.items():
        key = prefix + str(value).strip()
        if key in DUMMY_COLS:
            dummy_vec[DUMMY_COLS.index(key)] = 1.0

    emb_vecs = [
        embed_special(col, raw.get(col, "NA"))
        for col in VOCABS.keys()
    ]

    return np.hstack([*emb_vecs, num_parts, dummy_vec]).astype("float32")
