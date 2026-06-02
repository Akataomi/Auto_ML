"Model Registry - Central registry for all ML models"

from typing import Any, Callable, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from automl_core.models.mlp import MLPClassifier, MLPRegressor


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
            "mlp_classifier": lambda **kw: MLPClassifier(**kw),
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
            "mlp_regressor": lambda **kw: MLPRegressor(**kw),
        }
        
        # DeepMLP models are NOT in registry - they require special handling with config dict
        # They are created directly in trainer.py via _fit_deep_mlp_model()

        self.clustering_models: Dict[str, Callable] = {
            "kmeans": lambda **kw: KMeans(**kw, random_state=42, n_init=10),
            "dbscan": lambda **kw: DBSCAN(**kw),
            "agglomerative": lambda **kw: AgglomerativeClustering(**kw),
        }

        self.anomaly_models: Dict[str, Callable] = {
            "isolation_forest": lambda **kw: IsolationForest(**kw, random_state=42),
            "local_outlier_factor": lambda **kw: LocalOutlierFactor(**kw),
            "one_class_svm": lambda **kw: OneClassSVM(**kw),
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
        if 'iterations' not in kw:
            kw['iterations'] = 100
        if 'eval_metric' not in kw:
            kw['eval_metric'] = 'Accuracy'
        return CatBoostClassifier(verbose=0, **kw, random_state=42)

    def _get_cat_reg(self, **kw):
        from catboost import CatBoostRegressor
        if 'iterations' not in kw:
            kw['iterations'] = 100
        if 'eval_metric' not in kw:
            kw['eval_metric'] = 'R2'
        return CatBoostRegressor(verbose=0, **kw, random_state=42)

    def get(self, model_name: str, task_type: str, **kwargs) -> Any:
        """Get model instance by name"""
        models = self._get_models_for_task(task_type)
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found for task '{task_type}'. Available: {list(models.keys())}")
        return models[model_name](**kwargs)

    def create_model(self, model_name: str, task_type: str, **kwargs) -> Any:
        """Create model instance by name and task type"""
        return self.get(model_name, task_type, **kwargs)

    def _get_models_for_task(self, task_type: str) -> Dict:
        """Get models dict for task type"""
        mapping = {
            "classification": self.classification_models,
            "regression": self.regression_models,
            "clustering": self.clustering_models,
            "anomaly_detection": self.anomaly_models,
        }
        if task_type not in mapping:
            raise ValueError(f"Unknown task type: {task_type}")
        return mapping[task_type]

    def get_available_models(self, task_type: str) -> list:
        """Get list of available models for task type"""
        models = self._get_models_for_task(task_type)
        result = list(models.keys())
        # Add DeepMLP models for classification/regression (they're created specially in trainer)
        if task_type == "classification":
            result.append("deep_mlp_classifier")
        elif task_type == "regression":
            result.append("deep_mlp_regressor")
        return result

    def get_model_category(self, model_name: str) -> str:
        """Get model category for UI grouping"""
        linear = ["linear_regression", "ridge", "lasso", "elastic_net", "logistic_regression", "sgd_classifier", "sgd_regressor"]
        boosting = ["xgboost", "lightgbm", "catboost"]
        clustering = ["kmeans", "dbscan", "agglomerative"]
        anomaly = ["isolation_forest", "local_outlier_factor", "one_class_svm"]
        neural = ["mlp_classifier", "mlp_regressor", "deep_mlp_classifier", "deep_mlp_regressor"]

        if model_name in linear:
            return "linear"
        elif model_name in boosting:
            return "boosting"
        elif model_name in clustering:
            return "clustering"
        elif model_name in anomaly:
            return "anomaly"
        elif model_name in neural:
            return "neural"
        return "other"
