import numpy as np
from typing import Dict, Optional


class MetricsCalculator:
    """Калькулятор метрик"""

    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Метрики для классификации"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except Exception:
                metrics["auc_roc"] = None

        return metrics

    @staticmethod
    def regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Метрики для регрессии"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }