
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score
from itertools import product
import mlflow

from src.moneylion import logger
from src.moneylion.entity.config_entity import ModelTrainingConfig
from src.moneylion.utils.common import create_directories


class ModelTrainerBase:
    def __init__(self, cfg: ModelTrainingConfig) -> None:
        self.cfg = cfg
        create_directories([self.cfg.root_dir])

    def run(self) -> None:
        raise NotImplementedError


class XGBTrainer(ModelTrainerBase):
    STAGE_NAME = "Model Training (XGBoost)"

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.load(self.cfg.embed_dir / f"X_{split}.npy")
        y = np.load(self.cfg.embed_dir / f"y_{split}.npy")
        return X, y

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        # 1. Load data
        X_train, y_train = self._load_split("train")
        X_val,   y_val   = self._load_split("val")
        X_test,  y_test  = self._load_split("test")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        # 2. Grid search (tiny PoV grid)
        best_auc = -1.0
        best_params: Dict[str, Any] = {}

        param_grid = self.cfg.param_grid
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update(
                objective="binary:logistic",
                eval_metric=self.cfg.metric,
            )
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params.get("n_estimators", 200),
                    evals=[(dval, "val")],
                    early_stopping_rounds=self.cfg.early_stopping_rounds,
                    verbose_eval=False,
                )

                # validation AUC
                val_pred = booster.predict(dval)
                auc = roc_auc_score(y_val, val_pred)
                mlflow.log_metric("val_auc", auc)

                logger.info("Params %s → val_auc=%.4f", params, auc)
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
                    best_booster = booster

        # 3. Final evaluation on test split
        test_pred = best_booster.predict(dtest)
        test_auc  = roc_auc_score(y_test, test_pred)
        test_f1   = f1_score(y_test, (test_pred > 0.5).astype(int))
        logger.info("Best params %s", best_params)
        logger.info("Test AUC=%.4f  F1=%.4f", test_auc, test_f1)

        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1",  test_f1)

        # 4. Persist model & metrics
        model_path = self.cfg.root_dir / "xgb_model.json"
        best_booster.save_model(model_path)
        with open(self.cfg.root_dir / "metrics.json", "w") as f:
            json.dump({"test_auc": test_auc, "test_f1": test_f1}, f, indent=2)

        logger.info("Model saved to %s", model_path)
