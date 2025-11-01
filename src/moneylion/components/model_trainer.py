from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from itertools import product

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix
)
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

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.load(self.cfg.embed_dir / f"X_{split}.npy")
        y = np.load(self.cfg.embed_dir / f"y_{split}.npy")
        return X.astype("float32"), y.astype("int64")

    @staticmethod
    def _choose_threshold_on_val(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            y_hat = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        y_hat = (y_prob >= best_t).astype(int)
        metrics = {
            "val_f1": f1_score(y_true, y_hat, zero_division=0),
            "val_precision": precision_score(y_true, y_hat, zero_division=0),
            "val_recall": recall_score(y_true, y_hat, zero_division=0),
            "val_accuracy": accuracy_score(y_true, y_hat),
            "val_mcc": matthews_corrcoef(y_true, y_hat) if len(np.unique(y_true)) == 2 else 0.0,
            "val_balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
        }
        return best_t, metrics

    @staticmethod
    def _compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "") -> Dict[str, float]:
        """Compute all classification metrics at a given threshold"""
        y_hat = (y_prob >= threshold).astype(int)
        metrics = {
            f"{prefix}f1": f1_score(y_true, y_hat, zero_division=0),
            f"{prefix}precision": precision_score(y_true, y_hat, zero_division=0),
            f"{prefix}recall": recall_score(y_true, y_hat, zero_division=0),
            f"{prefix}accuracy": accuracy_score(y_true, y_hat),
            f"{prefix}mcc": matthews_corrcoef(y_true, y_hat) if len(np.unique(y_true)) == 2 else 0.0,
            f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
        }
        return metrics

    @staticmethod
    def _compute_test_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[Dict[str, float], np.ndarray]:
        y_hat = (y_prob >= threshold).astype(int)
        metrics = {
            "test_auc": roc_auc_score(y_true, y_prob),
            "test_f1": f1_score(y_true, y_hat, zero_division=0),
            "test_precision": precision_score(y_true, y_hat, zero_division=0),
            "test_recall": recall_score(y_true, y_hat, zero_division=0),
            "test_accuracy": accuracy_score(y_true, y_hat),
            "test_mcc": matthews_corrcoef(y_true, y_hat) if len(np.unique(y_true)) == 2 else 0.0,
            "test_balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
            "decision_threshold": float(threshold),
        }
        cm = confusion_matrix(y_true, y_hat)
        return metrics, cm

    def run(self) -> None:
        if self.cfg.mlflow_experiment:
            mlflow.set_experiment(self.cfg.mlflow_experiment)

        X_train, y_train = self._load_split("train")
        X_val,   y_val   = self._load_split("val")
        X_test,  y_test  = self._load_split("test")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        # Grid search on validation AUC
        best_auc = -1.0
        best_params: Dict[str, Any] = {}
        best_booster: xgb.Booster | None = None

        param_grid = self.cfg.param_grid or {}
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        for combo in product(*values) if values else [()]:
            params = dict(zip(keys, combo)) if combo else {}
            params.update(
                objective="binary:logistic",
                eval_metric=self.cfg.metric,
            )
            num_boost_round = int(params.pop("n_estimators", 200))

            with mlflow.start_run(nested=True):
                mlflow.log_params({**params, "num_boost_round": num_boost_round})

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, "val")],
                    early_stopping_rounds=self.cfg.early_stopping_rounds,
                    verbose_eval=False,
                )

                val_pred = booster.predict(dval)
                val_auc = roc_auc_score(y_val, val_pred)
                
                val_class_metrics = self._compute_classification_metrics(
                    y_val, val_pred, threshold=0.5, prefix="val_"
                )
                
                mlflow.log_metric("val_auc", val_auc)
                
                for metric_name, metric_value in val_class_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                logger.info(
                    "Params %s → val_auc=%.4f, val_f1=%.4f, val_acc=%.4f, val_mcc=%.4f",
                    {**params, "num_boost_round": num_boost_round},
                    val_auc,
                    val_class_metrics["val_f1"],
                    val_class_metrics["val_accuracy"],
                    val_class_metrics["val_mcc"]
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_params = {**params, "num_boost_round": num_boost_round}
                    best_booster = booster

        assert best_booster is not None, "No model was trained."

        val_pred_best = best_booster.predict(dval)
        if self.cfg.decision_threshold.lower() == "auto":
            threshold, val_class_metrics = self._choose_threshold_on_val(y_val, val_pred_best)
        else:
            try:
                threshold = float(self.cfg.decision_threshold)
            except ValueError:
                threshold = 0.5
            val_class_metrics = self._compute_classification_metrics(
                y_val, val_pred_best, threshold, prefix="val_"
            )

        # Final test metrics
        test_prob = best_booster.predict(dtest)
        test_metrics, cm = self._compute_test_metrics(y_test, test_prob, threshold)

        # Persist model and reports
        model_path = self.cfg.root_dir / "xgb_model.json"
        best_booster.save_model(model_path)

        metrics_all = {
            "best_params": best_params,
            "val_auc": best_auc,
            **val_class_metrics,
            **test_metrics,
        }
        with open(self.cfg.root_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

        # Save confusion matrix in JSON and CSV
        np.savetxt(self.cfg.root_dir / "confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")
        with open(self.cfg.root_dir / "confusion_matrix.json", "w") as f:
            json.dump({"labels": ["neg", "pos"], "matrix": cm.astype(int).tolist()}, f, indent=2)

        logger.info("Best params %s", best_params)
        logger.info(
            "Test AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f  Acc=%.4f  MCC=%.4f  Thr=%.3f",
            test_metrics["test_auc"], test_metrics["test_f1"], test_metrics["test_precision"],
            test_metrics["test_recall"], test_metrics["test_accuracy"], test_metrics["test_mcc"],
            test_metrics["decision_threshold"],
        )
        logger.info("Model saved to %s", model_path)

        # MLflow logging for the winning run
        with mlflow.start_run(nested=True):
            mlflow.log_params(best_params)
            for k, v in metrics_all.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(self.cfg.root_dir / "metrics.json", artifact_path="reports")
            mlflow.log_artifact(self.cfg.root_dir / "confusion_matrix.csv", artifact_path="reports")
            mlflow.log_artifact(self.cfg.root_dir / "confusion_matrix.json", artifact_path="reports")
