"Model Registry - Central registry for all ML models"

from typing import Any, Callable, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)


class ModelRegistry:
    "Registry of available ML models with lazy loading"

    def __init__(self):
        self.classification_models: Dict[str, Callable] = {
            "random_forest": lambda **kw: RandomForestClassifier(**kw, random_state=42, n_jobs=-1),
            "logistic_regression": lambda **kw: LogisticRegression(**kw, random_state=42, max_iter=1000, n_jobs=-1),
            "sgd_classifier": lambda **kw: SGDClassifier(**kw, random_state=42, max_iter=1000, tol=1e-3),
            "xgboost": self._get_xgb_clf,
            "lightgbm": self._get_lgbm_clf,
            "catboost": self._get_cat_clf,
        }

        self.regression_models: Dict[str, Callable] = {
            "random_forest": lambda **kw: RandomForestRegressor(**kw, random_state=42, n_jobs=-1),
            "linear_regression": lambda **kw: LinearRegression(),
            "ridge": lambda **kw: Ridge(**kw, random_state=42),
            "lasso": lambda **kw: Lasso(**kw, random_state=42, max_iter=1000),
            "elastic_net": lambda **kw: ElasticNet(**kw, random_state=42, max_iter=1000),
            "sgd_regressor": lambda **kw: SGDRegressor(**kw, random_state=42, max_iter=1000, tol=1e-3),
            "xgboost": self._get_xgb_reg,
            "lightgbm": self._get_lgbm_reg,
            "catboost": self._get_cat_reg,
        }

    def _get_xgb_clf(self, **kw):
        from xgboost import XGBClassifier
        return XGBClassifier(**kw, random_state=42, n_jobs=-1)

    def _get_xgb_reg(self, **kw):
        from xgboost import XGBRegressor
        return XGBRegressor(**kw, random_state=42, n_jobs=-1)

    def _get_lgbm_clf(self, **kw):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**kw, random_state=42, n_jobs=-1)

    def _get_lgbm_reg(self, **kw):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**kw, random_state=42, n_jobs=-1)

    def _get_cat_clf(self, **kw):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(verbose=0, **kw, random_state=42)

    def _get_cat_reg(self, **kw):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(verbose=0, **kw, random_state=42)

    def get(self, model_name: str, task_type: str, **kwargs) -> Any:
        """Get model instance by name"""
        models = self.classification_models if task_type == "classification" else self.regression_models
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")
        return models[model_name](**kwargs)

    def create_model(self, model_name: str, task_type: str, **kwargs) -> Any:
        """Create model instance by name and task type"""
        return self.get(model_name, task_type, **kwargs)

    def get_available_models(self, task_type: str) -> list:
        """Get list of available models for task type"""
        if task_type == "classification":
            return list(self.classification_models.keys())
        return list(self.regression_models.keys())

    def get_model_category(self, model_name: str) -> str:
        """Get model category for UI grouping"""
        linear = ["linear_regression", "ridge", "lasso", "elastic_net", "logistic_regression", "sgd_classifier", "sgd_regressor"]
        boosting = ["xgboost", "lightgbm", "catboost"]
        if model_name in linear:
            return "linear"
        elif model_name in boosting:
            return "boosting"
        return "tree"