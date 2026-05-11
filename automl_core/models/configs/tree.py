from typing import Any, Dict
from automl_core.models.base import ModelConfig
from .linear import _BaseOptunaSpace
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestConfig(ModelConfig):
    def __init__(self, task_type: str = "classification"):
        self._task_type = task_type
    
    @property
    def name(self) -> str:
        return "random_forest"
    
    @property
    def task_type(self) -> str:
        return self._task_type
    
    @property
    def category(self) -> str:
        return "tree"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.randint(100, 500),
            "max_depth": trial.randint(3, 20),
            "min_samples_split": trial.randint(2, 20),
            "min_samples_leaf": trial.randint(1, 10),
        }
    
    def create_model(self, **params) -> Any:
        ModelClass = RandomForestClassifier if self._task_type == "classification" else RandomForestRegressor
        return ModelClass(**params, random_state=42, n_jobs=-1)