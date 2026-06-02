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
import lightgbm as lgb
import traceback
from automl_core.models.deep_learning import DeepMLPTrainer, DeepMLP
from sklearn.preprocessing import StandardScaler

from automl_core.models.registry import ModelRegistry
from automl_core.tuning.optuna_tuner import OptunaTuner
from automl_core.utils.helpers import (
    ensure_numpy_array,
    validate_data_shape,
    is_unsupervised_task,
    get_default_metric_name,
    get_default_metric_display_name,
)
from automl_core.config.constants import (
    VALIDATION_SPLIT_RATIO,
    RANDOM_STATE,
    MIN_SAMPLES_FOR_STRATIFY,
    CV_FOLDS,
    DEFAULT_N_TRIALS,
    BOOSTING_EARLY_STOPPING_ROUNDS,
    MLP_EARLY_STOPPING_PATIENCE,
    MLP_DEFAULT_HIDDEN_LAYERS,
    MLP_DEFAULT_ACTIVATION,
    MLP_DEFAULT_BATCHNORM,
    MLP_DEFAULT_DROPOUT,
    MLP_DEFAULT_OPTIMIZER,
    MLP_DEFAULT_LEARNING_RATE,
    MLP_DEFAULT_BATCH_SIZE,
    MLP_DEFAULT_EPOCHS,
    MLP_DEFAULT_DEVICE,
    SGD_DEFAULT_EPOCHS,
    SGD_LOG_INTERVAL,
    ITERATIVE_MODELS,
)
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
        n_trials: int = 50,
        mlp_config: Optional[Dict] = None,
        use_cv: bool = True
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.tune = tune
        self.n_trials = n_trials
        self.mlp_config = mlp_config
        self.use_cv = use_cv
        
        self.registry = ModelRegistry()
        self.model = None
        self.tuner = None
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        
        self.training_history: TrainingHistory = TrainingHistory()
        self.optimization_history: List[Dict] = []
        self.best_iteration: int = 0
    
    def _create_validation_split(self, X, y=None) -> Tuple:
        """Create train/validation split"""
        if y is not None and len(y) > MIN_SAMPLES_FOR_STRATIFY:
            stratify_y = y if self.task_type == "classification" else None
            return train_test_split(
                X, y, 
                test_size=VALIDATION_SPLIT_RATIO, 
                random_state=RANDOM_STATE, 
                stratify=stratify_y
            )
        elif y is not None:
            return train_test_split(
                X, y, 
                test_size=VALIDATION_SPLIT_RATIO, 
                random_state=RANDOM_STATE
            )
        return train_test_split(
            X, 
            test_size=VALIDATION_SPLIT_RATIO, 
            random_state=RANDOM_STATE
        ) + (None, None)
    
    def fit(self, X, y=None):
        """Train the model with history tracking"""
        is_unsupervised = self.task_type in ["clustering", "anomaly_detection"]
        
        print(f"\n [ModelTrainer] Starting training: {self.model_name} ({self.task_type})", file=sys.stderr)
        print(f" [ModelTrainer] Data shape: X={X.shape}, y={'None' if y is None else y.shape}", file=sys.stderr)
        
        # Hyperparameter tuning
        if self.tune:
            print(f" [ModelTrainer] Running hyperparameter tuning with {self.n_trials or DEFAULT_N_TRIALS} trials...", file=sys.stderr)
            self.tuner = OptunaTuner(self.registry, self.n_trials or DEFAULT_N_TRIALS)
            self.tuner.tune(self.model_name, self.task_type, X, y)
            best_params = self.tuner.get_best_params()
            self.optimization_history = self.tuner.get_history()
            print(f" [ModelTrainer] Tuning complete. Best params: {best_params}", file=sys.stderr)
        else:
            best_params = {}
            self.tuner = None
        
        # Create model
        # DeepMLP models are created specially in _fit_deep_mlp_model()
        if self.model_name in ["deep_mlp_classifier", "deep_mlp_regressor"]:
            print(" [ModelTrainer] DeepMLP will be created in _fit_deep_mlp_model...", file=sys.stderr)
            self.model = None  # Will be created in _fit_deep_mlp_model
        else:
            print(" [ModelTrainer] Creating model instance...", file=sys.stderr)
            self.model = self.registry.create_model(self.model_name, self.task_type, **best_params)
        
        if is_unsupervised:
            if self.task_type == "clustering":
                print(" [ModelTrainer] Training clustering model...", file=sys.stderr)
                self.model.fit(X)
                self.labels = self.model.labels_
            elif self.task_type == "anomaly_detection":
                print(" [ModelTrainer] Training anomaly detection model...", file=sys.stderr)
                self.model.fit(X)
                self.predictions = self.model.predict(X)
        elif self.model_name in ["deep_mlp_classifier", "deep_mlp_regressor"]:
            print(" [ModelTrainer] Training Deep MLP model...", file=sys.stderr)
            self._fit_deep_mlp_model(X, y)
        elif self.model_name in ["xgboost", "lightgbm", "catboost"]:
            print(" [ModelTrainer] Training boosting model...", file=sys.stderr)
            self._fit_boosting_model(X, y)
        elif self.model_name in ["sgd_regressor", "sgd_classifier"]:
            print(" [ModelTrainer] Training SGD model...", file=sys.stderr)
            self._fit_sgd_model(X, y)
        else:
            print(" [ModelTrainer] Training analytical model (single step)...", file=sys.stderr)
            self.model.fit(X, y)
        
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            self.coefficients = self.model.coef_
            if len(self.coefficients.shape) == 1:
                self.coefficients = self.coefficients.reshape(1, -1)
        
        print(" [ModelTrainer] Training complete!", file=sys.stderr)
        print(f" [ModelTrainer] History epochs: {len(self.training_history.epochs)}", file=sys.stderr)
        print(f" [ModelTrainer] Val loss points: {len(self.training_history.val_loss)}", file=sys.stderr)
        
        return self
    
    def _fit_boosting_model(self, X, y):
        """Train Boosting model with built-in eval logging"""
        print("[Boosting] Creating validation split...", file=sys.stderr)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y)
        self.training_history = TrainingHistory()
        
        if self.model_name == "xgboost":
            print("[XGBoost] Fitting model...", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=BOOSTING_EARLY_STOPPING_ROUNDS if X_val is not None else None,
                    verbose=False
                )
            except TypeError:
                print("[XGBoost] early_stopping_rounds not supported, fitting without...", file=sys.stderr)
                self.model.fit(X_train, y_train, verbose=False)
            
            if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                print("[XGBoost] Extracting evals_result...", file=sys.stderr)
                res = self.model.evals_result_
                print(f"[XGBoost] Keys: {res.keys()}", file=sys.stderr)
                for key, metrics in res.items():
                    if 'valid' in key.lower() or 'validation' in key.lower():
                        print(f"[XGBoost] Processing {key}: {list(metrics.keys())}", file=sys.stderr)
                        if 'rmse' in metrics:
                            self.training_history.val_loss.extend(metrics['rmse'])
                        if 'logloss' in metrics:
                            self.training_history.val_loss.extend(metrics['logloss'])
                        if 'auc' in metrics:
                            self.training_history.val_metric.extend(metrics['auc'])
                        if 'error' in metrics:
                            self.training_history.val_metric.extend([1 - e for e in metrics['error']])
            else:
                print(" [XGBoost] No evals_result_ found!", file=sys.stderr)
            
            self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else 0

        elif self.model_name == "lightgbm":
            print("[LightGBM] Fitting model...", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(BOOSTING_EARLY_STOPPING_ROUNDS, verbose=False)] if X_val is not None else [],
                )
            except Exception as e:
                print(f"[LightGBM] Error during fit: {e}", file=sys.stderr)
                self.model.fit(X_train, y_train)
            
            if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                print("[LightGBM] Extracting evals_result...", file=sys.stderr)
                res = self.model.evals_result_
                for key, metrics in res.items():
                    if 'valid' in key.lower():
                        print(f"[LightGBM] Processing {key}: {list(metrics.keys())}", file=sys.stderr)
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
            print("[CatBoost] Fitting model...", file=sys.stderr)
            print(f" [CatBoost] X_val type: {type(X_val)}, shape: {X_val.shape if X_val is not None else 'None'}", file=sys.stderr)
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            print(f" [CatBoost] eval_set: {eval_set}", file=sys.stderr)
            
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False,
                    early_stopping_rounds=BOOSTING_EARLY_STOPPING_ROUNDS if eval_set else None
                )
            except TypeError:
                print("⚠️ [CatBoost] early_stopping_rounds not supported, fitting without...", file=sys.stderr)
                self.model.fit(X_train, y_train, verbose=False)
            
            if hasattr(self.model, 'get_evals_result'):
                print(" [CatBoost] Extracting get_evals_result...", file=sys.stderr)
                res = self.model.get_evals_result()
                print(f" [CatBoost] Keys: {res.keys()}", file=sys.stderr)
                
                # Debug: print full structure
                for key, metrics in res.items():
                    print(f" [CatBoost] Key '{key}': {list(metrics.keys())}", file=sys.stderr)
                    for m_name, values in metrics.items():
                        print(f" [CatBoost]   {m_name}: {len(values)} points, first 3: {values[:3] if len(values) > 0 else '[]'}", file=sys.stderr)
                
                # Extract both train and validation metrics
                train_key = None
                val_key = None
                for key in res.keys():
                    if 'valid' in key.lower() or 'validation' in key.lower():
                        val_key = key
                    elif 'learn' in key.lower() or 'train' in key.lower():
                        train_key = key
                
                print(f" [CatBoost] Train key: {train_key}, Val key: {val_key}", file=sys.stderr)
                
                # Process validation metrics
                if val_key:
                    print(f" [CatBoost] Processing validation: {list(res[val_key].keys())}", file=sys.stderr)
                    for m_name, values in res[val_key].items():
                        print(f" [CatBoost]   {m_name}: {len(values)} values", file=sys.stderr)
                        # Loss метрики
                        if 'loss' in m_name.lower() or 'rmse' in m_name.lower() or 'logloss' in m_name.lower():
                            self.training_history.val_loss.extend(values)
                        # Metric метрики (Accuracy, R2, AUC, etc.)
                        elif 'accuracy' in m_name.lower() or 'auc' in m_name.lower() or 'r2' in m_name.lower() or 'f1' in m_name.lower():
                            self.training_history.val_metric.extend(values)
                else:
                    print("[CatBoost] No validation key found in evals_result", file=sys.stderr)
                
                # Process train metrics
                if train_key:
                    print(f" [CatBoost] Processing training: {list(res[train_key].keys())}", file=sys.stderr)
                    for m_name, values in res[train_key].items():
                        print(f" [CatBoost]   {m_name}: {len(values)} values", file=sys.stderr)
                        # Loss метрики
                        if 'loss' in m_name.lower() or 'rmse' in m_name.lower() or 'logloss' in m_name.lower():
                            self.training_history.train_loss.extend(values)
                        # Metric метрики (Accuracy, R2, AUC, etc.)
                        elif 'accuracy' in m_name.lower() or 'auc' in m_name.lower() or 'r2' in m_name.lower() or 'f1' in m_name.lower():
                            self.training_history.train_metric.extend(values)
                else:
                    print("⚠️ [CatBoost] No train key found in evals_result!", file=sys.stderr)
            else:
                print(" [CatBoost] No get_evals_result found!", file=sys.stderr)
            
            self.best_iteration = self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else 0
        
        n_epochs = max(len(self.training_history.val_loss), len(self.training_history.val_metric), 1)
        self.training_history.epochs = list(range(1, n_epochs + 1))
        print(f" [Boosting] Training complete. Epochs: {n_epochs}, Loss points: {len(self.training_history.val_loss)}, Metric points: {len(self.training_history.val_metric)}", file=sys.stderr)
    
    def _fit_deep_mlp_model(self, X, y):
        """Train Deep MLP model with PyTorch"""
        
        print(" [DeepMLP] Creating validation split...", file=sys.stderr)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Получаем параметры из конфига модели
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train)) if self.task_type == "classification" else 1
        
        # Приоритет: mlp_config из конструктора > tuned параметры > дефолтные
        if self.mlp_config:
            params = self.mlp_config.copy()
            print(f" [DeepMLP] Using user config from UI: {params.keys()}", file=sys.stderr)
        elif self.tuner and self.tuner.best_params:
            params = self.tuner.best_params.copy()
            print(f" [DeepMLP] Using tuned params: {params.keys()}", file=sys.stderr)
        else:
            params = {
                "hidden_layers": MLP_DEFAULT_HIDDEN_LAYERS,
                "activation": MLP_DEFAULT_ACTIVATION,
                "use_batchnorm": MLP_DEFAULT_BATCHNORM,
                "dropout_rate": MLP_DEFAULT_DROPOUT,
                "optimizer": MLP_DEFAULT_OPTIMIZER,
                "learning_rate": MLP_DEFAULT_LEARNING_RATE,
                "batch_size": MLP_DEFAULT_BATCH_SIZE,
                "epochs": MLP_DEFAULT_EPOCHS,
                "early_stopping_patience": MLP_EARLY_STOPPING_PATIENCE,
            }

        model_config = {
            "input_dim": int(n_features),
            "output_dim": int(n_classes),
            "hidden_layers": params.get("hidden_layers", MLP_DEFAULT_HIDDEN_LAYERS),
            "activation": params.get("activation", MLP_DEFAULT_ACTIVATION),
            "use_batchnorm": params.get("use_batchnorm", MLP_DEFAULT_BATCHNORM),
            "dropout_rate": float(params.get("dropout_rate", MLP_DEFAULT_DROPOUT)),
            "task_type": self.task_type
        }

        if self.task_type == "classification":
            model_config["output_dim"] = int(n_classes)
        
        print(f" [DeepMLP] Creating model with config: {model_config}", file=sys.stderr)
        model = DeepMLP(model_config)
        self.model = model
        
        trainer = DeepMLPTrainer(
            model=model,
            optimizer=params.get("optimizer", MLP_DEFAULT_OPTIMIZER),
            learning_rate=float(params.get("learning_rate", MLP_DEFAULT_LEARNING_RATE)),
            batch_size=int(params.get("batch_size", MLP_DEFAULT_BATCH_SIZE)),
            epochs=int(params.get("epochs", MLP_DEFAULT_EPOCHS)),
            early_stopping_patience=int(params.get("early_stopping_patience", MLP_EARLY_STOPPING_PATIENCE)),
            device=MLP_DEFAULT_DEVICE
        )
        
        print(f" [DeepMLP] Starting training with config: hidden={model_config['hidden_layers']}, act={model_config['activation']}, opt={params.get('optimizer')}, lr={params.get('learning_rate')}", file=sys.stderr)
        history = trainer.fit(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            verbose=True
        )

        self.training_history = TrainingHistory()
        self.training_history.epochs = history['epochs']
        self.training_history.train_loss = history['train_loss']
        self.training_history.val_loss = history['val_loss']
        self.training_history.train_metric = history['train_metric']
        self.training_history.val_metric = history['val_metric']

        self._deep_trainer = trainer
        self._scaler = scaler
        
        print(f" [DeepMLP] Training complete. Best val_loss: {min(history['val_loss']):.4f}", file=sys.stderr)
    
    def _fit_sgd_model(self, X, y):
        """Train SGD model iteratively to capture validation loss"""
        print(" [SGD] Creating validation split...", file=sys.stderr)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y)
        self.training_history = TrainingHistory()
        
        print(f" [SGD] Training for {SGD_DEFAULT_EPOCHS} epochs...", file=sys.stderr)
        
        for i in range(1, SGD_DEFAULT_EPOCHS + 1):
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
            
            if i % SGD_LOG_INTERVAL == 0:
                print(f" [SGD] Epoch {i}/{SGD_DEFAULT_EPOCHS}, Val Loss: {val_loss:.4f}", file=sys.stderr)
        
        self.best_iteration = SGD_DEFAULT_EPOCHS
        print(" [SGD] Training complete", file=sys.stderr)
    
    def evaluate(self, X, y=None) -> Dict[str, float]:
        """Evaluate model"""
        
        is_unsupervised = self.task_type in ["clustering", "anomaly_detection"]
        
        # Для DeepMLP используем специальный трейнер
        is_deep_mlp = self.model_name in ["deep_mlp_classifier", "deep_mlp_regressor"]
        
        print(f"\n [Evaluate] Starting evaluation for {self.model_name}", file=sys.stderr)
        print(f" [Evaluate] X type: {type(X)}, shape: {X.shape if hasattr(X, 'shape') else 'N/A'}", file=sys.stderr)
        print(f" [Evaluate] y type: {type(y)}, shape: {y.shape if y is not None and hasattr(y, 'shape') else 'N/A'}", file=sys.stderr)
        
        if is_unsupervised:
            print(f" [Evaluate] Unsupervised task: {self.task_type}", file=sys.stderr)
            if self.task_type == "clustering":
                # Clustering metrics
                print(" [Evaluate] Computing clustering metrics...", file=sys.stderr)
                self.metrics = MetricsCalculator.clustering_metrics(X, self.labels)
            elif self.task_type == "anomaly_detection":
                # Anomaly detection metrics
                print(" [Evaluate] Computing anomaly metrics...", file=sys.stderr)
                self.metrics = MetricsCalculator.anomaly_metrics(X, self.predictions, y)
        elif is_deep_mlp and hasattr(self, '_deep_trainer'):
            # DeepMLP evaluation
            print(" [Evaluate] DeepMLP evaluation path...", file=sys.stderr)
            try:
                print(" [Evaluate] Converting X to array...", file=sys.stderr)
                X_array = X.values if hasattr(X, 'values') else np.array(X)
                print(" [Evaluate] X_array type: {type(X_array)}, shape: {X_array.shape}", file=sys.stderr)
                
                print(" [Evaluate] Scaling data...", file=sys.stderr)
                X_scaled = self._scaler.transform(X_array)
                print(f" [Evaluate] X_scaled shape: {X_scaled.shape}", file=sys.stderr)
                
                print(f" [Evaluate] Predicting...", file=sys.stderr)
                y_pred = self._deep_trainer.predict(X_scaled)
                print(f" [Evaluate] y_pred type: {type(y_pred)}, shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'N/A'}", file=sys.stderr)
                
                if self.task_type == "classification":
                    print(" [Evaluate] Getting probabilities...", file=sys.stderr)
                    y_pred_proba = self._deep_trainer.predict_proba(X_scaled)
                    print(f" [Evaluate] y_pred_proba shape: {y_pred_proba.shape}", file=sys.stderr)
                    
                    # Save for classification curves
                    self.y_true_val = y
                    self.y_pred_val = y_pred
                    self.y_pred_proba_val = y_pred_proba
                    
                    print(" [Evaluate] Computing classification metrics...", file=sys.stderr)
                    self.metrics = MetricsCalculator.classification_metrics(y, y_pred, y_pred_proba)
                else:
                    print(" [Evaluate] Computing regression metrics...", file=sys.stderr)
                    self.metrics = MetricsCalculator.regression_metrics(y, y_pred)
                    
                print(f" [Evaluate] Metrics computed: {list(self.metrics.keys())}", file=sys.stderr)
            except Exception as e:
                print(f" [Evaluate] Error during DeepMLP evaluation: {e}", file=sys.stderr)
                print(" [Evaluate] Traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise
        elif self.task_type == "classification":
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
            
            # Save for classification curves
            self.y_true_val = y
            self.y_pred_val = y_pred
            self.y_pred_proba_val = y_pred_proba
            
            self.metrics = MetricsCalculator.classification_metrics(y, y_pred, y_pred_proba)
        else:
            # Regression
            y_pred = self.model.predict(X)
            self.metrics = MetricsCalculator.regression_metrics(y, y_pred)
        
        # CV scores only for supervised and only if use_cv is True
        if not is_unsupervised and y is not None and self.use_cv:
            try:
                cv_scores = cross_val_score(self.model, X, y, cv=CV_FOLDS, n_jobs=-1)
                self.metrics["cv_mean"] = float(cv_scores.mean())
                self.metrics["cv_std"] = float(cv_scores.std())
                print(f" [Evaluate] CV completed: mean={self.metrics['cv_mean']:.4f}, std={self.metrics['cv_std']:.4f}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ [Evaluate] CV error: {e}", file=sys.stderr)
                self.metrics["cv_mean"] = 0.0
                self.metrics["cv_std"] = 0.0
        elif is_unsupervised:
            print(" [Evaluate] Skipping CV for unsupervised task", file=sys.stderr)
        elif not self.use_cv:
            print(" [Evaluate] CV disabled by user", file=sys.stderr)
        
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
    
    def get_classification_curves_data(self) -> Dict[str, Any]:
        """Get ROC, Precision-Recall curves and Confusion Matrix for classification"""
        if self.task_type != "classification":
            return {}
        
        curves_data = {
            'roc': None,
            'pr': None,
            'confusion_matrix': None
        }
        
        # Если есть сохраненные данные валидации
        if hasattr(self, 'y_true_val') and hasattr(self, 'y_pred_proba_val') and self.y_pred_proba_val is not None:
            try:
                curves_data['roc'] = MetricsCalculator.get_roc_curve_data(
                    self.y_true_val, 
                    self.y_pred_proba_val
                )
                curves_data['pr'] = MetricsCalculator.get_precision_recall_curve_data(
                    self.y_true_val, 
                    self.y_pred_proba_val
                )
            except Exception as e:
                print(f"[Trainer] Error computing curves: {e}", file=sys.stderr)
        
        # Confusion matrix
        if hasattr(self, 'y_true_val') and hasattr(self, 'y_pred_val') and self.y_pred_val is not None:
            try:
                curves_data['confusion_matrix'] = MetricsCalculator.get_confusion_matrix(
                    self.y_true_val, 
                    self.y_pred_val
                )
            except Exception as e:
                print(f"[Trainer] Error computing confusion matrix: {e}", file=sys.stderr)
        
        return curves_data
