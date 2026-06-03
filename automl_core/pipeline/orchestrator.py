"""
Pipeline Orchestrator.

Coordinates all stages of the AutoML pipeline.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from automl_core.data import DataLoader, DataAnalyzer
from automl_core.preprocessing import DataCleaner, DataEncoder
from automl_core.models.registry import ModelRegistry
from automl_core.training.trainer import ModelTrainer
from automl_core.evaluation import MetricsCalculator
from automl_core.utils.mlflow_logger import MLFlowLogger
from .config import PipelineConfig


class AutoMLPipeline:
    """Main orchestrator for the AutoML pipeline"""

    def __init__(self, config: PipelineConfig, use_mlflow: bool = False, use_mlflow_docker: bool = False):
        self.config = config
        self.registry = ModelRegistry()
        self.report: Dict[str, Any] = {}
        self.models_trained: List[Dict] = []
        self.best_model: Optional[ModelTrainer] = None
        self.best_score: float = 0
        self.use_mlflow = use_mlflow
        self.use_mlflow_docker = use_mlflow_docker
        self.mlflow_logger = None
        
        if use_mlflow:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.mlflow_logger = MLFlowLogger(
                experiment_name=f"AutoML_{self.config.task_type}",
                use_docker=use_mlflow_docker
            )
            run_name = f"{self.config.task_type}_{timestamp}"
            self.mlflow_logger.start_run(run_name)

    def run(self, filepath: Optional[str], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the complete AutoML pipeline
        
        Args:
            filepath: Path to CSV file (for uploaded files)
            df: DataFrame (for built-in datasets)
        """
        if df is not None:
            pass
        elif filepath is not None:
            # Load from file
            df = DataLoader.load(filepath)
        else:
            raise ValueError("Either filepath or df must be provided")
        
        is_unsupervised = self.config.task_type in ["clustering", "anomaly_detection"]
        task_type_str = "unsupervised" if is_unsupervised else "supervised"
        DataLoader.validate(df, self.config.target_column, task_type_str)
        
        self.report["data_info"] = DataAnalyzer.get_summary(df)
        self.report["column_types"] = DataAnalyzer.get_column_types(df)
        
        if self.config.target_column:
            self.report["target_info"] = DataAnalyzer.get_target_info(df, self.config.target_column)

        col_types = self.report["column_types"]
        cleaner = DataCleaner(
            fill_strategy=self.config.preprocessing.fill_missing,
            outlier_method="iqr" if self.config.preprocessing.handle_outliers else None,
        )
        df_clean = cleaner.fit_transform(df, col_types["numeric"])

        encoder = DataEncoder(
            categorical_strategy=self.config.preprocessing.encode_categorical,
            scale=self.config.preprocessing.scale,
        )
        X, y, feature_names = encoder.fit_transform(
            df_clean, self.config.target_column, col_types["categorical"], col_types["numeric"]
        )
        self.report["features"] = feature_names
        self.report["preprocessing"] = {
            "fill_strategy": self.config.preprocessing.fill_missing,
            "encode_strategy": self.config.preprocessing.encode_categorical,
            "scaled": self.config.preprocessing.scale,
        }

        if self.use_mlflow and self.mlflow_logger:
            self.mlflow_logger.log_params({
                "task_type": self.config.task_type,
                "metric": self.config.metric,
                "target_column": self.config.target_column,
                "num_models": len(self.config.models),
                "fill_strategy": self.config.preprocessing.fill_missing,
                "encode_strategy": self.config.preprocessing.encode_categorical,
                "scale": self.config.preprocessing.scale,
                "handle_outliers": self.config.preprocessing.handle_outliers,
            })

        results = []
        for model_cfg in self.config.models:
            try:
                mlp_config = model_cfg.mlp_config if hasattr(model_cfg, 'mlp_config') else None
                use_cv = model_cfg.use_cv if hasattr(model_cfg, 'use_cv') else True
                
                trainer = ModelTrainer(
                    model_name=model_cfg.name,
                    task_type=self.config.task_type,
                    tune=model_cfg.tune_hyperparams,
                    n_trials=model_cfg.n_trials,
                    mlp_config=mlp_config,
                    use_cv=use_cv
                )
                trainer.fit(X, y)
                metrics = trainer.evaluate(X, y)

                result = {
                    "model": model_cfg.name,
                    "metrics": metrics,
                    "tuned": model_cfg.tune_hyperparams,
                }
                results.append(result)
                self.models_trained.append({"name": model_cfg.name, "trainer": trainer})

                if self.use_mlflow and self.mlflow_logger:
                    metrics_with_prefix = {f"{model_cfg.name}_{k}": v for k, v in metrics.items()}
                    self.mlflow_logger.log_metrics(metrics_with_prefix, step=0)
                    print(f"[MLFlow] Logged metrics for {model_cfg.name}")

                if self.use_mlflow and self.mlflow_logger:
                    self.mlflow_logger.log_model(trainer.model, model_cfg.name)
                    print(f"[MLFlow] Logged model {model_cfg.name}")

                if self.use_mlflow and self.mlflow_logger and trainer.feature_importance is not None:
                    importance_dict = dict(zip(self.report["features"], trainer.feature_importance))
                    importance_with_prefix = {f"{model_cfg.name}_importance_{k}": v for k, v in importance_dict.items()}
                    self.mlflow_logger.log_feature_importance(importance_with_prefix, model_name=model_cfg.name)
                    print(f"[MLFlow] Logged feature importance for {model_cfg.name}")

                if self.use_mlflow and self.mlflow_logger and trainer.has_learning_curves():
                    self.mlflow_logger.log_training_curves(trainer.training_history, model_cfg.name)
                    print(f"[MLFlow] Logged training curves for {model_cfg.name}")

                if is_unsupervised:
                    if self.config.task_type == "clustering":
                        score = metrics.get("silhouette", 0)
                    else:
                        score = metrics.get("f1", metrics.get("anomaly_rate", 0))
                else:
                    score = metrics.get(self.config.metric, metrics.get("accuracy", metrics.get("r2", 0)))
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = trainer

            except Exception as e:
                results.append({"model": model_cfg.name, "error": str(e)})
                print(f"[MLFlow] Error training {model_cfg.name}: {e}")

        self.report["results"] = results
        self.report["best_model"] = {
            "name": self.models_trained[0]["name"] if self.models_trained else None,
            "score": self.best_score,
        }

        if self.use_mlflow and self.mlflow_logger:
            self.mlflow_logger.end_run()

        return self.report

    def save_best_model(self, filepath: str):
        """Save the best model to file"""
        if self.best_model:
            self.best_model.save(filepath)
            self.report["saved_model_path"] = filepath

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model"""
        if self.best_model and self.best_model.feature_importance is not None:
            return dict(zip(self.report["features"], self.best_model.feature_importance))
        return {}