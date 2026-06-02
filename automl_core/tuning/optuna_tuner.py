"""Optuna Tuner Implementation"""

import optuna
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score
from typing import Any, Dict, List, Optional
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, f1_score


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner
    """
    
    def __init__(self, registry, n_trials: int = 50, timeout: Optional[int] = None):
        self.registry = registry
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params: Dict[str, Any] = {}
        self.history: List[Dict] = []
        
        self.sampler = TPESampler(seed=42, multivariate=True)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    def _get_objective(self, model_name: str, task_type: str, X, y, cv: int):
        """Create objective function for Optuna"""
        
        def objective(trial: optuna.Trial):
            params = self._get_search_space(trial, model_name)

            model = self.registry.create_model(model_name, task_type, **params)

            if task_type == "clustering":
                return self._clustering_objective(trial, model, X)
            elif task_type == "anomaly_detection":
                return self._anomaly_objective(trial, model, X, y)

            if model_name == "xgboost":
                return self._xgboost_objective(trial, model, X, y, cv)
            elif model_name == "lightgbm":
                return self._lightgbm_objective(trial, model, X, y, cv)
            elif model_name == "catboost":
                return self._catboost_objective(trial, model, X, y, cv)
            else:
                scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                score = scores.mean() if task_type == "classification" else -scores.mean()
                
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
        
        return objective
    
    def _clustering_objective(self, trial: optuna.Trial, model, X):
        """Objective for clustering - maximize silhouette score"""
        
        model.fit(X)
        labels = model.labels_

        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= len(X):
            return 0.0
        
        score = silhouette_score(X, labels)
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _anomaly_objective(self, trial: optuna.Trial, model, X, y=None):
        """Objective for anomaly detection"""
        
        model.fit(X)
        predictions = model.predict(X)

        if y is not None:
            y_binary = (y != y.mode().iloc[0]).astype(int) if hasattr(y, 'mode') else (y == -1).astype(int)
            pred_binary = (predictions == -1).astype(int)
            score = f1_score(y_binary, pred_binary, zero_division=0)
        else:
            anomaly_mask = predictions == -1
            if anomaly_mask.sum() > 1 and (~anomaly_mask).sum() > 1:
                try:
                    score = silhouette_score(X, predictions)
                except:
                    score = 0.5
            else:
                score = 0.5
        
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _xgboost_objective(self, trial: optuna.Trial, model, X, y, cv: int):
        scoring = "accuracy" if isinstance(model, XGBClassifier) else "neg_mean_squared_error"
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        score = scores.mean() if isinstance(model, XGBClassifier) else -scores.mean()
        
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _lightgbm_objective(self, trial: optuna.Trial, model, X, y, cv: int):
        scoring = "binary" if model._task_type == "classification" else "regression"
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy" if scoring == "binary" else "neg_mean_squared_error", n_jobs=-1)
        score = scores.mean() if scoring == "binary" else -scores.mean()
        
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _catboost_objective(self, trial: optuna.Trial, model, X, y, cv: int):    
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy" if isinstance(model, CatBoostClassifier) else "neg_mean_squared_error", n_jobs=-1)
        score = scores.mean() if isinstance(model, CatBoostClassifier) else -scores.mean()
        
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _get_search_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 0.4),
            }
        
        elif model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        
        elif model_name == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 100, 500, step=50),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
        
        elif model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        
        elif model_name == "sgd_regressor":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "optimal", "adaptive"]),
                "eta0": trial.suggest_float("eta0", 0.001, 0.1, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            }
        
        elif model_name == "sgd_classifier":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "optimal", "adaptive"]),
                "eta0": trial.suggest_float("eta0", 0.001, 0.1, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            }
        
        elif model_name == "ridge":
            return {
                "alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True),
            }
        
        elif model_name == "lasso":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            }
        
        elif model_name == "elastic_net":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
        
        elif model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", None]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            }

        elif model_name == "kmeans":
            return {
                "n_clusters": trial.suggest_int("n_clusters", 2, 10),
                "init": trial.suggest_categorical("init", ["k-means++", "random"]),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
            }
        
        elif model_name == "dbscan":
            return {
                "eps": trial.suggest_float("eps", 0.1, 10.0, log=True),
                "min_samples": trial.suggest_int("min_samples", 2, 20),
            }
        
        elif model_name == "agglomerative":
            return {
                "n_clusters": trial.suggest_int("n_clusters", 2, 10),
                "linkage": trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"]),
            }

        elif model_name == "isolation_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "contamination": trial.suggest_float("contamination", 0.01, 0.2),
                "max_samples": trial.suggest_categorical("max_samples", ["auto", 256, 512, 1024]),
            }
        
        elif model_name == "local_outlier_factor":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
                "contamination": trial.suggest_float("contamination", 0.01, 0.2),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
            }
        
        elif model_name == "one_class_svm":
            return {
                "nu": trial.suggest_float("nu", 0.01, 0.5),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                "gamma": trial.suggest_float("gamma", 0.001, 1.0, log=True),
            }

        elif model_name == "mlp_classifier":
            return {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
                "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
            }
        
        elif model_name == "mlp_regressor":
            return {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
                "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
            }

        elif model_name in ["deep_mlp_classifier", "deep_mlp_regressor"]:
            return {
                "hidden_layers": trial.suggest_categorical(
                    "hidden_layers",
                    [[64], [128], [64, 32], [128, 64], [256, 128], [128, 64, 32], [256, 128, 64]]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu", "tanh", "gelu"]),
                "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epochs": trial.suggest_int("epochs", 50, 150, step=25),
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 10, 30, step=5),
            }
        
        else:
            return {}
    
    def tune(self, model_name: str, task_type: str, X, y, cv: int = 5) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            model_name: Name of the model to tune
            task_type: 'classification' or 'regression'
            X: Features
            y: Target
            cv: Cross-validation folds
            
        Returns:
            Dict with best_params and best_score
        """
        print(f"\n [Optuna] Starting optimization for {model_name}...", file=__import__('sys').stderr)
        print(f" [Optuna] Trials: {self.n_trials}, Timeout: {self.timeout}", file=__import__('sys').stderr)
        
        direction = "maximize" if task_type == "classification" else "minimize"
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"{model_name}_{task_type}"
        )

        objective = self._get_objective(model_name, task_type, X, y, cv)

        try:
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
                gc_after_trial=True
            )
        except Exception as e:
            print(f" [Optuna] Optimization error: {e}", file=__import__('sys').stderr)
        
        self.best_params = self.study.best_params
        self.history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in self.study.trials
        ]
        
        print(" [Optuna] Optimization complete!", file=__import__('sys').stderr)
        print(f" [Optuna] Best score: {self.study.best_value:.4f}", file=__import__('sys').stderr)
        print(f" [Optuna] Best params: {self.study.best_params}", file=__import__('sys').stderr)
        
        return {
            "best_params": self.best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
            "n_complete_trials": len([t for t in self.study.trials if t.state.name == "COMPLETE"]),
        }
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        return self.best_params
    
    def get_history(self) -> List[Dict]:
        """Get optimization history"""
        return self.history
    
    def get_study(self):
        """Get Optuna study object for visualization"""
        return self.study