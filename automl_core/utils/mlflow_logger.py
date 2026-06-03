"""
MLFlow Logger for AutoML experiments.

Provides unified logging interface for all model types.
"""

import os
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import mlflow.lightgbm
import mlflow.pytorch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile


class MLFlowLogger:
    """Logger for AutoML experiments in MLFlow"""

    def __init__(self, experiment_name: str = "AutoML_Experiments", tracking_uri: str = None, use_docker: bool = False):
        """
        Initialize MLFlow logger.

        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: URI of MLFlow tracking server (default: from env or local ./mlruns)
            use_docker: If True, use Docker MLFlow service (http://mlflow-service:5000)
        """
        if tracking_uri is None:
            if use_docker:
                tracking_uri = "http://mlflow-service:5000"
            elif os.getenv("MLFLOW_TRACKING_URI"):
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            else:
                tracking_uri = f"file://{Path('./mlruns').absolute()}"

        print(f"[MLFlow] Using tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.active_run = None

    def start_run(self, run_name: str):
        """Start a new MLFlow run"""
        self.active_run = mlflow.start_run(run_name=run_name)
        print(f"[MLFlow] Started run: {run_name}")

    def end_run(self):
        """End the current MLFlow run"""
        if self.active_run:
            mlflow.end_run()
            print(f"[MLFlow] Ended run: {self.active_run.info.run_id}")
            self.active_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters"""
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping params logging")
            return

        clean_params = {}
        for key, value in params.items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    clean_params[key] = str(value)
                elif isinstance(value, (int, float, str, bool)):
                    clean_params[key] = value
                else:
                    clean_params[key] = str(value)

        mlflow.log_params(clean_params)
        print(f"[MLFlow] Logged {len(clean_params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log model metrics"""
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping metrics logging")
            return

        clean_metrics = {k: v for k, v in metrics.items() if v is not None and not np.isnan(v)}

        if step is not None:
            for key, value in clean_metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(clean_metrics)

        print(f"[MLFlow] Logged {len(clean_metrics)} metrics")

    def _get_model_type(self, model_name: str, model_obj: Any) -> str:
        """
        Determine model type for proper MLFlow flavor.

        Args:
            model_name: Name of the model (e.g., 'catboost', 'deep_mlp_classifier')
            model_obj: The actual model object

        Returns:
            Model type string: 'pytorch', 'xgboost', 'catboost', 'lightgbm', or 'sklearn'
        """
        model_name_lower = model_name.lower()

        if 'deep_mlp' in model_name_lower:
            return 'pytorch'
        elif 'xgboost' in model_name_lower:
            return 'xgboost'
        elif 'catboost' in model_name_lower:
            return 'catboost'
        elif 'lightgbm' in model_name_lower:
            return 'lightgbm'

        model_type = type(model_obj).__module__.lower()
        if 'torch' in model_type:
            return 'pytorch'
        elif 'xgboost' in model_type:
            return 'xgboost'
        elif 'catboost' in model_type:
            return 'catboost'
        elif 'lightgbm' in model_type:
            return 'lightgbm'

        return 'sklearn'

    def log_model(self, model: Any, model_name: str, model_obj: Any = None):
        """
        Log model artifact to MLFlow.

        Args:
            model_name: Name of the model
            model_obj: The actual model object (if None, will try to extract from model)
        """
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping model logging")
            return

        try:
            # Extract actual model object if needed
            if model_obj is None:
                # Try to get model from trainer
                if hasattr(model, 'model'):
                    model_obj = model.model
                elif hasattr(model, 'best_model_'):
                    model_obj = model.best_model_
                else:
                    model_obj = model

            model_type = self._get_model_type(model_name, model_obj)
            print(f"[MLFlow] Logging model '{model_name}' as type: {model_type}")

            # Log using appropriate flavor
            if model_type == 'pytorch':
                try:
                    mlflow.pytorch.log_model(model_obj, "model")
                    print("[MLFlow] Model logged as PyTorch")
                except Exception as e:
                    print(f"[MLFlow] PyTorch logging failed: {e}, falling back to pickle")
                    self._log_model_pickle(model_obj, model_name)

            elif model_type == 'xgboost':
                try:
                    mlflow.xgboost.log_model(model_obj, "model")
                    print("[MLFlow] Model logged as XGBoost")
                except Exception as e:
                    print(f"[MLFlow] XGBoost logging failed: {e}, falling back to pickle")
                    self._log_model_pickle(model_obj, model_name)

            elif model_type == 'catboost':
                try:
                    mlflow.catboost.log_model(model_obj, "model")
                    print("[MLFlow] Model logged as CatBoost")
                except Exception as e:
                    print(f"[MLFlow] CatBoost logging failed: {e}, falling back to pickle")
                    self._log_model_pickle(model_obj, model_name)

            elif model_type == 'lightgbm':
                try:
                    mlflow.lightgbm.log_model(model_obj, "model")
                    print("[MLFlow] Model logged as LightGBM")
                except Exception as e:
                    print(f"[MLFlow] LightGBM logging failed: {e}, falling back to pickle")
                    self._log_model_pickle(model_obj, model_name)

            else:
                try:
                    mlflow.sklearn.log_model(model_obj, "model")
                    print("[MLFlow] Model logged as sklearn")
                except Exception as e:
                    print(f"[MLFlow] sklearn logging failed: {e}, falling back to pickle")
                    self._log_model_pickle(model_obj, model_name)

        except Exception as e:
            print(f"[MLFlow] Error logging model: {e}")
            try:
                self._log_model_pickle(model_obj if model_obj else model, model_name)
            except Exception as e2:
                print(f"[MLFlow] Final fallback failed: {e2}")

    def _log_model_pickle(self, model: Any, model_name: str):
        """Save model as pickle artifact (fallback method)"""

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, f"models/{model_name}")
            print(f"[MLFlow] Model saved as pickle artifact")
        finally:
            os.unlink(temp_path)

    def log_feature_importance(self, importance: Dict[str, float], top_n: int = 20, model_name: str = None):
        """
        Log feature importance as metrics and artifact.

        Args:
            importance: Dictionary of {feature_name: importance_value}
            top_n: Number of top features to log as metrics
            model_name: Optional model name prefix for metrics and artifact path
        """
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping feature importance logging")
            return

        if not importance:
            print("[MLFlow] No feature importance data")
            return

        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, imp_value) in enumerate(sorted_importance[:top_n]):
            safe_name = feature.replace('/', '_').replace(' ', '_').replace('.', '_')
            mlflow.log_metric(safe_name, float(imp_value))

        try:

            df_importance = pd.DataFrame(sorted_importance, columns=['feature', 'importance'])

            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
                df_importance.to_csv(f, index=False)
                temp_path = f.name

            artifact_path = f"feature_importance/{model_name}" if model_name else "feature_importance"
            mlflow.log_artifact(temp_path, artifact_path)
            os.unlink(temp_path)

            print(f"[MLFlow] Logged feature importance for {len(sorted_importance)} features")

        except Exception as e:
            print(f"[MLFlow] Error logging feature importance: {e}")

    def log_training_curves(self, history, model_name: str):
        """
        Log training curves as metrics and plot artifact.

        Args:
            history: TrainingHistory object or Dict with training history
            model_name: Name of the model
        """
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping training curves logging")
            return

        if not history:
            print("[MLFlow] No training history data")
            return

        try:
            # Convert TrainingHistory object to dict
            if hasattr(history, 'to_dataframe'):
                df = history.to_dataframe()
                history_dict = {}
                for col in df.columns:
                    if col != 'epoch':
                        history_dict[col] = df[col].tolist()
            else:
                history_dict = history

            # Log metrics per epoch/iteration
            for metric_name, values in history_dict.items():
                if isinstance(values, list) and len(values) > 0:
                    for step, value in enumerate(values):
                        if value is not None and not np.isnan(value):
                            safe_name = metric_name.replace('/', '_').replace(' ', '_')
                            mlflow.log_metric(f"{model_name}_{safe_name}", float(value), step=step)

            plot_path = self._create_training_curve_plot(history_dict, model_name)
            if plot_path:
                mlflow.log_artifact(plot_path, "training_curves")
                os.unlink(plot_path)
                print(f"[MLFlow] Logged training curves for {model_name}")

        except Exception as e:
            print(f"[MLFlow] Error logging training curves: {e}")

    def _create_training_curve_plot(self, history: Dict[str, List[float]], model_name: str) -> Optional[str]:
        """
        Create training curve plot and save to file.

        Args:
            history: Training history dictionary
            model_name: Model name for title

        Returns:
            Path to saved plot file or None
        """

        if not history:
            return None

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
            metric_keys = [k for k in history.keys() if 'loss' not in k.lower()]

            if loss_keys and axes[0]:
                for key in loss_keys:
                    if isinstance(history[key], list) and len(history[key]) > 0:
                        axes[0].plot(history[key], label=key)
                axes[0].set_xlabel('Epoch/Iteration')
                axes[0].set_ylabel('Loss')
                axes[0].set_title(f'{model_name} - Loss Curves')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            if metric_keys and axes[1]:
                for key in metric_keys:
                    if isinstance(history[key], list) and len(history[key]) > 0:
                        axes[1].plot(history[key], label=key)
                axes[1].set_xlabel('Epoch/Iteration')
                axes[1].set_ylabel('Metric')
                axes[1].set_title(f'{model_name} - Metric Curves')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f, dpi=150, bbox_inches='tight')
                plot_path = f.name

            plt.close(fig)
            return plot_path

        except Exception as e:
            print(f"[MLFlow] Error creating training curve plot: {e}")
            return None

    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str], model_name: str):
        """
        Log confusion matrix as artifact.

        Args:
            cm: Confusion matrix array
            class_names: List of class names
            model_name: Model name
        """
        if not self.active_run:
            print("[MLFlow] Warning: No active run, skipping confusion matrix logging")
            return

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'{model_name} - Confusion Matrix')

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f, dpi=150, bbox_inches='tight')
                plot_path = f.name

            mlflow.log_artifact(plot_path, "confusion_matrices")
            os.unlink(plot_path)
            plt.close(fig)

            print(f"[MLFlow] Logged confusion matrix for {model_name}")

        except Exception as e:
            print(f"[MLFlow] Error logging confusion matrix: {e}")
