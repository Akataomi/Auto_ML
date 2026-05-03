"""
Pipeline Orchestration Module.

Main orchestrator class that coordinates all stages of the AutoML pipeline.
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from automl_core.data import DataLoader, DataAnalyzer
from automl_core.preprocessing import DataCleaner, DataEncoder
from automl_core.models import ModelRegistry, ModelTrainer
from .config import PipelineConfig


class AutoMLPipeline:
    """
    Main orchestrator for the AutoML pipeline.
    
    Coordinates all stages:
        1. Data loading and validation
        2. Exploratory data analysis (EDA)
        3. Preprocessing (cleaning, encoding, scaling)
        4. Model training with optional hyperparameter tuning
        5. Evaluation and reporting
    
    Attributes:
        config: Pipeline configuration.
        report: Detailed execution report.
        best_model: Best performing model.
        
    Example:
        >>> config = PipelineConfig(target_column="target", task_type="classification", models=[])
        >>> pipeline = AutoMLPipeline(config)
        >>> report = pipeline.run("dataset.csv")
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the AutoML pipeline.
        
        Args:
            config: Configuration object defining pipeline behavior.
        """
        self.config = config
        self.registry = ModelRegistry()
        self.report: Dict[str, Any] = {}
        self.models_trained: List[Dict] = []
        self.best_model: Optional[ModelTrainer] = None
        self.best_score: float = 0

    def run(self, filepath: str) -> Dict[str, Any]:
        df = DataLoader.load(filepath)
        DataLoader.validate(df, self.config.target_column)
        self.report["data_info"] = DataAnalyzer.get_summary(df)
        self.report["column_types"] = DataAnalyzer.get_column_types(df)
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

        results = []
        for model_cfg in self.config.models:
            try:
                model = self.registry.get(model_cfg.name, self.config.task_type)
                trainer = ModelTrainer(
                    model, tune_hyperparams=model_cfg.tune_hyperparams, n_trials=model_cfg.n_trials
                )
                trainer.fit(X, y, self.config.task_type)
                metrics = trainer.evaluate(X, y, self.config.task_type)

                result = {
                    "model": model_cfg.name,
                    "metrics": metrics,
                    "tuned": model_cfg.tune_hyperparams,
                }
                results.append(result)
                self.models_trained.append({"name": model_cfg.name, "trainer": trainer})

                score = metrics.get(self.config.metric, metrics.get("accuracy", metrics.get("r2", 0)))
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = trainer

            except Exception as e:
                results.append({"model": model_cfg.name, "error": str(e)})

        self.report["results"] = results
        self.report["best_model"] = {
            "name": self.models_trained[0]["name"] if self.models_trained else None,
            "score": self.best_score,
        }

        return self.report

    def save_best_model(self, filepath: str):
        """
        Save the best model to file.
        
        Args:
            filepath: Path to save the model.
        """
        if self.best_model:
            self.best_model.save(filepath)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the best model.
        
        Returns:
            Dict: Feature names mapped to importance scores.
        """
        if self.best_model and self.best_model.feature_importance is not None:
            return dict(zip(self.report["features"], self.best_model.feature_importance))
        return {}