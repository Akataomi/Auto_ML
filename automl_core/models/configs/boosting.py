from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from typing import Any, Dict
from automl_core.models.base import ModelConfig
from .linear import _BaseOptunaSpace


class CatBoostConfig(ModelConfig):
    def __init__(self, task_type: str = "classification"):
        self._task_type = task_type
    
    @property
    def name(self) -> str:
        return "catboost"
    
    @property
    def task_type(self) -> str:
        return self._task_type
    
    @property
    def category(self) -> str:
        return "boosting"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "iterations": trial.randint(100, 500),
            "depth": trial.randint(3, 10),
            "learning_rate": trial.loguniform(0.01, 0.3),
            "l2_leaf_reg": trial.loguniform(1, 10),
        }
    
    def create_model(self, **params) -> Any:
        ModelClass = CatBoostClassifier if self._task_type == "classification" else CatBoostRegressor
        return ModelClass(verbose=0, **params, random_state=42)


class LightGBMConfig(ModelConfig):
    def __init__(self, task_type: str = "classification"):
        self._task_type = task_type
    
    @property
    def name(self) -> str:
        return "lightgbm"
    
    @property
    def task_type(self) -> str:
        return self._task_type
    
    @property
    def category(self) -> str:
        return "boosting"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.randint(100, 500),
            "max_depth": trial.randint(3, 10),
            "learning_rate": trial.loguniform(0.01, 0.3),
            "num_leaves": trial.randint(20, 100),
            "subsample": trial.uniform(0.7, 1.0),
        }
    
    def create_model(self, **params) -> Any:
        ModelClass = LGBMClassifier if self._task_type == "classification" else LGBMRegressor
        return ModelClass(**params, random_state=42, n_jobs=-1)


class XGBoostConfig(ModelConfig):
    def __init__(self, task_type: str = "classification"):
        self._task_type = task_type
    
    @property
    def name(self) -> str:
        return "xgboost"
    
    @property
    def task_type(self) -> str:
        return self._task_type
    
    @property
    def category(self) -> str:
        return "boosting"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.randint(100, 500),
            "max_depth": trial.randint(3, 10),
            "learning_rate": trial.loguniform(0.01, 0.3),
            "subsample": trial.uniform(0.7, 1.0),
            "colsample_bytree": trial.uniform(0.7, 1.0),
        }
    
    def create_model(self, **params) -> Any:
        ModelClass = XGBClassifier if self._task_type == "classification" else XGBRegressor
        return ModelClass(**params, random_state=42, n_jobs=-1)