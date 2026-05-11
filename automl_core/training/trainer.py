"""
Model Trainer.

Training with history tracking for iterative models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, Any, List, Optional, Tuple
import joblib
import sys

from automl_core.models.registry import ModelRegistry
from automl_core.tuning.optuna_tuner import OptunaTuner
from .history import TrainingHistory
from sklearn.metrics import log_loss
from scipy.special import expit
from automl_core.evaluation.metrics import MetricsCalculator


class ModelTrainer:
    """Training and evaluation with history tracking"""
    
    def __init__(
        self,
        model_name: str,
        task_type: str,
        tune: bool = False,
        n_trials: int = 50
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.tune = tune
        self.n_trials = n_trials
        
        self.registry = ModelRegistry()
        self.model = None
        self.tuner = None
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        
        self.training_history: TrainingHistory = TrainingHistory()
        self.optimization_history: List[Dict] = []
        self.best_iteration: int = 0
    
    def _create_validation_split(self, X, y) -> Tuple:
        """Create train/validation split"""
        if len(y) > 10:
            stratify_y = y if self.task_type == "classification" else None
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_y)
        return X, y, None, None
    
    def fit(self, X, y):
        """Train the model with history tracking"""
        print(f"\n [ModelTrainer] Starting training: {self.model_name} ({self.task_type})", file=sys.stderr)
        print(f" [ModelTrainer] Data shape: X={X.shape}, y={y.shape}", file=sys.stderr)
        
        # Hyperparameter tuning
        if self.tune:
            print(f" [ModelTrainer] Running hyperparameter tuning with {self.n_trials} trials...", file=sys.stderr)
            from automl_core.tuning.optuna_tuner import OptunaTuner
            self.tuner = OptunaTuner(self.registry, self.n_trials)
            self.tuner.tune(self.model_name, self.task_type, X, y)
            best_params = self.tuner.get_best_params()
            self.optimization_history = self.tuner.get_history()
            print(f" [ModelTrainer] Tuning complete. Best params: {best_params}", file=sys.stderr)
        else:
            best_params = {}
            self.tuner = None
        
        # Create model
        print(f" [ModelTrainer] Creating model instance...", file=sys.stderr)
        self.model = self.registry.create_model(self.model_name, self.task_type, **best_params)
        
        if self.model_name in ["xgboost", "lightgbm", "catboost"]:
            print(f" [ModelTrainer] Training boosting model...", file=sys.stderr)
            self._fit_boosting_model(X, y)
        elif self.model_name in ["sgd_regressor", "sgd_classifier"]:
            print(f" [ModelTrainer] Training SGD model...", file=sys.stderr)
            self._fit_sgd_model(X, y)
        else:
            print(f" [ModelTrainer] Training analytical model (single step)...", file=sys.stderr)
            self.model.fit(X, y)
        
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            self.coefficients = self.model.coef_
            if len(self.coefficients.shape) == 1:
                self.coefficients = self.coefficients.reshape(1, -1)
        
        print(f" [ModelTrainer] Training complete!", file=sys.stderr)
        print(f" [ModelTrainer] History epochs: {len(self.training_history.epochs)}", file=sys.stderr)
        print(f" [ModelTrainer] Val loss points: {len(self.training_history.val_loss)}", file=sys.stderr)
        
        return self
    
    def _fit_boosting_model(self, X, y):
        """Train Boosting model with built-in eval logging"""
        print(f" [Boosting] Creating validation split...", file=sys.stderr)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y)
        self.training_history = TrainingHistory()
        
        if self.model_name == "xgboost":
            print(f"🔧 [XGBoost] Fitting model...", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=50 if X_val is not None else None,
                    verbose=False
                )
            except TypeError:
                print(f" [XGBoost] early_stopping_rounds not supported, fitting without...", file=sys.stderr)
                self.model.fit(X_train, y_train, verbose=False)
            
            if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                print(f" [XGBoost] Extracting evals_result...", file=sys.stderr)
                res = self.model.evals_result_
                print(f" [XGBoost] Keys: {res.keys()}", file=sys.stderr)
                for key, metrics in res.items():
                    if 'valid' in key.lower() or 'validation' in key.lower():
                        print(f" [XGBoost] Processing {key}: {list(metrics.keys())}", file=sys.stderr)
                        if 'rmse' in metrics:
                            self.training_history.val_loss.extend(metrics['rmse'])
                        if 'logloss' in metrics:
                            self.training_history.val_loss.extend(metrics['logloss'])
                        if 'auc' in metrics:
                            self.training_history.val_metric.extend(metrics['auc'])
                        if 'error' in metrics:
                            self.training_history.val_metric.extend([1 - e for e in metrics['error']])
            else:
                print(f" [XGBoost] No evals_result_ found!", file=sys.stderr)
            
            self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else 0

        elif self.model_name == "lightgbm":
            print(f" [LightGBM] Fitting model...", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            try:
                import lightgbm as lgb
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(50, verbose=False)] if X_val is not None else [],
                    verbose=-1
                )
            except Exception as e:
                print(f" [LightGBM] Error during fit: {e}", file=sys.stderr)
                self.model.fit(X_train, y_train, verbose=-1)
            
            if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                print(f" [LightGBM] Extracting evals_result...", file=sys.stderr)
                res = self.model.evals_result_
                for key, metrics in res.items():
                    if 'valid' in key.lower():
                        print(f" [LightGBM] Processing {key}: {list(metrics.keys())}", file=sys.stderr)
                        if 'rmse' in metrics:
                            self.training_history.val_loss.extend(metrics['rmse'])
                        if 'binary_logloss' in metrics:
                            self.training_history.val_loss.extend(metrics['binary_logloss'])
                        if 'auc' in metrics:
                            self.training_history.val_metric.extend(metrics['auc'])
            else:
                print(f" [LightGBM] No evals_result_ found!", file=sys.stderr)
            
            self.best_iteration = self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else 0

        elif self.model_name == "catboost":
            print(f" [CatBoost] Fitting model...", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False,
                    early_stopping_rounds=50
                )
            except TypeError:
                print(f"⚠️ [CatBoost] early_stopping_rounds not supported, fitting without...", file=sys.stderr)
                self.model.fit(X_train, y_train, verbose=False)
            
            if hasattr(self.model, 'get_evals_result'):
                print(f" [CatBoost] Extracting get_evals_result...", file=sys.stderr)
                res = self.model.get_evals_result()
                print(f" [CatBoost] Keys: {res.keys()}", file=sys.stderr)
                for key, metrics in res.items():
                    if 'valid' in key.lower():
                        print(f" [CatBoost] Processing {key}: {list(metrics.keys())}", file=sys.stderr)
                        for m_name, values in metrics.items():
                            if 'loss' in m_name.lower() or 'rmse' in m_name.lower():
                                self.training_history.val_loss.extend(values)
                            elif 'accuracy' in m_name.lower() or 'auc' in m_name.lower():
                                self.training_history.val_metric.extend(values)
                            elif 'r2' in m_name.lower():
                                self.training_history.val_metric.extend(values)
            else:
                print(f" [CatBoost] No get_evals_result found!", file=sys.stderr)
            
            self.best_iteration = self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else 0
        
        n_epochs = max(len(self.training_history.val_loss), len(self.training_history.val_metric), 1)
        self.training_history.epochs = list(range(1, n_epochs + 1))
        print(f" [Boosting] Training complete. Epochs: {n_epochs}, Loss points: {len(self.training_history.val_loss)}, Metric points: {len(self.training_history.val_metric)}", file=sys.stderr)
    
    def _fit_sgd_model(self, X, y):
        """Train SGD model iteratively to capture validation loss"""
        print(f" [SGD] Creating validation split...", file=sys.stderr)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y)
        self.training_history = TrainingHistory()
        
        n_epochs = 100
        print(f" [SGD] Training for {n_epochs} epochs...", file=sys.stderr)
        
        for i in range(1, n_epochs + 1):
            self.model.partial_fit(X_train, y_train)
            
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val) if X_val is not None else None
            
            if self.task_type == "regression":
                train_loss = np.mean((y_train - train_pred) ** 2)
                val_loss = np.mean((y_val - val_pred) ** 2) if val_pred is not None else None
            else:
                try:
                    train_probs = self.model.decision_function(X_train)
                    val_probs = self.model.decision_function(X_val) if val_pred is not None else None
                    train_loss = log_loss(y_train, expit(train_probs), labels=self.model.classes_)
                    val_loss = log_loss(y_val, expit(val_probs), labels=self.model.classes_) if val_probs is not None else None
                except:
                    train_loss = np.mean(y_train != train_pred)
                    val_loss = np.mean(y_val != val_pred) if val_pred is not None else None

            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val) if X_val is not None else None
            
            self.training_history.add(
                epoch=i,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metric=train_score,
                val_metric=val_score
            )
            
            if i % 20 == 0:
                print(f" [SGD] Epoch {i}/{n_epochs}, Val Loss: {val_loss:.4f}", file=sys.stderr)
        
        self.best_iteration = n_epochs
        print(f" [SGD] Training complete", file=sys.stderr)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model"""
        
        if self.task_type == "classification":
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
            self.metrics = MetricsCalculator.classification_metrics(y, y_pred, y_pred_proba)
        else:
            y_pred = self.model.predict(X)
            self.metrics = MetricsCalculator.regression_metrics(y, y_pred)
        
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        self.metrics["cv_mean"] = float(cv_scores.mean())
        self.metrics["cv_std"] = float(cv_scores.std())
        
        return self.metrics
    
    def save(self, filepath: str):
        """Save model to file"""
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath: str):
        """Load model from file"""
        return joblib.load(filepath)
    
    def get_training_history_df(self) -> Optional[pd.DataFrame]:
        """Get training history for plotting"""
        if self.training_history.is_empty():
            return None
        return self.training_history.to_dataframe()
    
    def get_optimization_history_df(self) -> Optional[pd.DataFrame]:
        """Get Optuna optimization history"""
        if not self.optimization_history:
            return None
        return pd.DataFrame(self.optimization_history)
    
    def has_learning_curves(self) -> bool:
        """Check if model has learning curve data"""
        return not self.training_history.is_empty()