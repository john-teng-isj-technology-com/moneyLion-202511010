import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", str(REPO_ROOT / "artifacts")))
GCS_BUCKET = os.getenv("GCS_ARTIFACT_BUCKET", "moneylion-202511010")
GCS_PREFIX = os.getenv("GCS_ARTIFACT_PREFIX", "artifacts")
ARTIFACT_DIR   = ARTIFACTS_ROOT / "model_training"
PREPROCESS_DIR = ARTIFACTS_ROOT / "data_preprocessing"
EMBED_DIR      = ARTIFACTS_ROOT / "data_embedding"


def _download_prefix(client: storage.Client, bucket_name: str, prefix: str, dst_root: Path) -> None:
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        name = blob.name
        if name.endswith("/"):
            continue
        rel = Path(name).relative_to(prefix)  
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))


def _ensure_artifacts() -> None:
    if (ARTIFACT_DIR / "xgb_model.json").exists():
        return
    client = storage.Client()
    for folder in ("model_training", "data_preprocessing", "data_embedding"):
        _download_prefix(
            client,
            GCS_BUCKET,
            f"{GCS_PREFIX}/{folder}/",
            ARTIFACTS_ROOT / folder,
        )


_ensure_artifacts()

with open(PREPROCESS_DIR / "vocabs.json") as f:
    VOCABS = json.load(f)

with open(EMBED_DIR / "embed_schema.json") as f:
    SCHEMA = json.load(f)

with open(ARTIFACT_DIR / "numeric_stats.json") as f:
    NUM_STATS = json.load(f)

with open(ARTIFACT_DIR / "dummy_cols.json") as f:
    DUMMY_COLS = json.load(f)["dummies"]

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
        "state_": raw.get("state", ""),
        "leadType_": raw.get("leadType", ""),
        "fpStatus_": raw.get("fpStatus", ""),
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
