from typing import Any, Dict
from automl_core.models.base import ModelConfig
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge, LinearRegression


class _BaseOptunaSpace:
    
    @staticmethod
    def loguniform(trial, low: float = 0.001, high: float = 100) -> float:
        return trial.loguniform(low, high)
    
    @staticmethod
    def choice(trial, options: list) -> Any:
        return trial.choice(options)
    
    @staticmethod
    def uniform(trial, low: float = 0.0, high: float = 1.0) -> float:
        return trial.uniform(low, high)


class LogisticRegressionConfig(ModelConfig):
    @property
    def name(self) -> str:
        return "logistic_regression"
    
    @property
    def task_type(self) -> str:
        return "classification"
    
    @property
    def category(self) -> str:
        return "linear"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "C": _BaseOptunaSpace.loguniform(trial),
            "penalty": _BaseOptunaSpace.choice(trial, ["l2", "l1", "elasticnet", None]),
            "solver": _BaseOptunaSpace.choice(trial, ["liblinear", "saga"]),
        }
    
    def create_model(self, **params) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        return LogisticRegression(**params, random_state=42, max_iter=1000, n_jobs=-1)


class LinearRegressionConfig(ModelConfig):
    @property
    def name(self) -> str:
        return "linear_regression"
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    @property
    def category(self) -> str:
        return "linear"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {}
    
    def create_model(self, **params) -> Any:
        return LinearRegression()


class RidgeConfig(ModelConfig):
    @property
    def name(self) -> str:
        return "ridge"
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    @property
    def category(self) -> str:
        return "linear"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {"alpha": _BaseOptunaSpace.loguniform(trial)}
    
    def create_model(self, **params) -> Any:
        return Ridge(**params, random_state=42)


class LassoConfig(ModelConfig):
    @property
    def name(self) -> str:
        return "lasso"
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    @property
    def category(self) -> str:
        return "linear"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {"alpha": _BaseOptunaSpace.loguniform(trial)}
    
    def create_model(self, **params) -> Any:
        return Lasso(**params, random_state=42, max_iter=1000)


class ElasticNetConfig(ModelConfig):
    @property
    def name(self) -> str:
        return "elastic_net"
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    @property
    def category(self) -> str:
        return "linear"
    
    def get_search_space(self, trial) -> Dict[str, Any]:
        return {
            "alpha": _BaseOptunaSpace.loguniform(trial),
            "l1_ratio": _BaseOptunaSpace.uniform(trial),
        }
    
    def create_model(self, **params) -> Any:
        return ElasticNet(**params, random_state=42, max_iter=1000)