# src/moneylion/components/data_preprocessing.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.moneylion import logger
from src.moneylion.entity.config_entity import DataPreprocessingConfig
from src.moneylion.utils.common import create_directories


_SPECIAL_COLS: List[str] = [
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncode",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncodedescription",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncodedescription",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.phonetype",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.ssndobreasoncode",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncode",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncode",
]

_ID_COLS = ["loanId", "underwritingid", "clarityFraudId"]
_LABEL_COL = "isBadDebt"


class DataPreprocessor:
    def __init__(self, config: DataPreprocessingConfig) -> None:
        self.cfg = config
        create_directories([self.cfg.root_dir])

    @staticmethod
    def _build_vocab(series: pd.Series) -> Dict[str, int]:
        uniq = sorted(series.unique().tolist())
        return {token: idx for idx, token in enumerate(uniq)}

    def _split_and_save(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y: np.ndarray,
    ) -> None:
        train_idx, temp_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.cfg.test_size + self.cfg.val_size,
            stratify=y,
            random_state=self.cfg.random_state,
        )
        rel_test = self.cfg.test_size / (self.cfg.test_size + self.cfg.val_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            stratify=y[temp_idx],
            random_state=self.cfg.random_state,
        )

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        for split_name, idx in splits.items():
            np.save(self.cfg.root_dir / f"{split_name}_num.npy", X_num[idx].astype("float32"))
            np.save(self.cfg.root_dir / f"{split_name}_cat.npy", X_cat[idx].astype("int64"))
            np.save(self.cfg.root_dir / f"{split_name}_y.npy",   y[idx].astype("int64"))
            logger.info("Saved %s split – %d rows", split_name, len(idx))

    def run(self) -> None:
        logger.info("Loading joined_df from %s", self.cfg.joined_csv)
        df = pd.read_csv(self.cfg.joined_csv, low_memory=False)

        cat_idx_arrays: List[np.ndarray] = []
        vocabs_json: Dict[str, Dict[str, int]] = {}

        for col in _SPECIAL_COLS:
            if col not in df.columns:
                logger.warning("Special column %s not found – filling 'NA'", col)
                df[col] = "NA"
            df[col] = df[col].fillna("NA").astype(str)

            vocab = self._build_vocab(df[col])
            vocabs_json[col] = vocab
            cat_idx_arrays.append(df[col].map(vocab).to_numpy(np.int64))

        X_cat = np.stack(cat_idx_arrays, axis=1)  # shape [N, C]
        logger.info("Built categorical matrix shape %s", X_cat.shape)

        numeric_cols = [
            c
            for c in df.columns
            if c not in (_SPECIAL_COLS + _ID_COLS + [_LABEL_COL])
        ]
        X_num = df[numeric_cols].to_numpy(np.float32)
        logger.info("Numeric matrix shape %s", X_num.shape)

        y = df[_LABEL_COL].to_numpy(np.int64)

        self._split_and_save(X_num, X_cat, y)

        with open(self.cfg.root_dir / "vocabs.json", "w") as f:
            json.dump(vocabs_json, f, indent=2)
        with open(self.cfg.root_dir / "columns.json", "w") as f:
            json.dump(
                {
                    "categorical": _SPECIAL_COLS,
                    "numeric": numeric_cols,
                    "label": _LABEL_COL,
                },
                f,
                indent=2,
            )

        logger.info("Data-preprocessing completed. Artifacts written to %s", self.cfg.root_dir)
