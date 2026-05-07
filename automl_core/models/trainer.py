import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

from .tuner import HyperparameterTuner


class ModelTrainer:
    """Обучение и оценка моделей"""

    def __init__(self, model, tune_hyperparams: bool = False, n_trials: int = 50):
        self.model = model
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials
        self.tuned_model = None
        self.metrics = {}
        self.feature_importance = None

    def fit(self, X, y, task_type: str = "classification") -> "ModelTrainer":
        """Обучение модели"""
        if self.tune_hyperparams:
            tuner = HyperparameterTuner(type(self.model), task_type, self.n_trials)
            tuner.tune(X, y)
            self.tuned_model = tuner.get_best_model()
            self.tuned_model.fit(X, y)
            self.model = self.tuned_model
        else:
            self.model.fit(X, y)

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_

        return self

    def evaluate(self, X, y, task_type: str = "classification") -> Dict[str, float]:
        """Оценка модели"""
        from automl_core.evaluation.metrics import MetricsCalculator

        if task_type == "classification":
            y_pred = self.model.predict(X)
            y_pred_proba = None
            if hasattr(self.model, "predict_proba"):
                y_pred_proba = self.model.predict_proba(X)
            self.metrics = MetricsCalculator.classification_metrics(y, y_pred, y_pred_proba)
        else:
            y_pred = self.model.predict(X)
            self.metrics = MetricsCalculator.regression_metrics(y, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        self.metrics["cv_mean"] = float(cv_scores.mean())
        self.metrics["cv_std"] = float(cv_scores.std())

        return self.metrics

    def save(self, filepath: str):
        """Сохранение модели"""
        joblib.dump(self.model, filepath)

    @staticmethod
    def load(filepath: str):
        """Загрузка модели"""
        return joblib.load(filepath)