from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Any, Dict


class ModelRegistry:
    """Реестр доступных моделей"""

    def __init__(self):
        self.classification_models = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "xgboost": self._get_xgboost_classifier,
            "lightgbm": self._get_lightgbm_classifier,
            "catboost": self._get_catboost_classifier,
        }

        self.regression_models = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "xgboost": self._get_xgboost_regressor,
            "lightgbm": self._get_lightgbm_regressor,
            "catboost": self._get_catboost_regressor,
        }

    def _get_xgboost_classifier(self, **kwargs):
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(**kwargs)
        except ImportError:
            raise ImportError("xgboost не установлен")

    def _get_xgboost_regressor(self, **kwargs):
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(**kwargs)
        except ImportError:
            raise ImportError("xgboost не установлен")

    def _get_lightgbm_classifier(self, **kwargs):
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(**kwargs)
        except ImportError:
            raise ImportError("lightgbm не установлен")

    def _get_lightgbm_regressor(self, **kwargs):
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**kwargs)
        except ImportError:
            raise ImportError("lightgbm не установлен")

    def _get_catboost_classifier(self, **kwargs):
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(verbose=0, **kwargs)
        except ImportError:
            raise ImportError("catboost не установлен")

    def _get_catboost_regressor(self, **kwargs):
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(verbose=0, **kwargs)
        except ImportError:
            raise ImportError("catboost не установлен")

    def get(self, model_name: str, task_type: str, **kwargs) -> Any:
        """Получение модели по имени"""
        if task_type == "classification":
            models = self.classification_models
        elif task_type == "regression":
            models = self.regression_models
        else:
            raise ValueError(f"Неизвестный тип задачи: {task_type}")

        if model_name not in models:
            raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {list(models.keys())}")

        return models[model_name](**kwargs)

    def get_available_models(self, task_type: str) -> list:
        """Список доступных моделей"""
        if task_type == "classification":
            return list(self.classification_models.keys())
        elif task_type == "regression":
            return list(self.regression_models.keys())
        return []