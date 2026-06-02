from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from automl_core.config.constants import (
    VALIDATION_SPLIT_RATIO,
    CV_FOLDS,
    DEFAULT_N_TRIALS,
    MLP_ARCHITECTURE_PRESETS,
    MLP_DEFAULT_ACTIVATION,
    MLP_DEFAULT_BATCHNORM,
    MLP_DEFAULT_DROPOUT,
    MLP_DEFAULT_OPTIMIZER,
    MLP_DEFAULT_LEARNING_RATE,
    MLP_DEFAULT_BATCH_SIZE,
    MLP_DEFAULT_EPOCHS,
    MLP_EARLY_STOPPING_PATIENCE,
)

MLP_PRESET_SMALL = MLP_ARCHITECTURE_PRESETS["Small"]


class PreprocessingConfig(BaseModel):
    fill_missing: str = Field(default="median", description="median, mean, or drop")
    encode_categorical: str = Field(default="onehot", description="onehot, label")
    scale: bool = Field(default=True)
    handle_outliers: bool = Field(default=False)


class ModelConfig(BaseModel):
    name: str = Field(..., description="model name from registry")
    tune_hyperparams: bool = Field(default=False)
    n_trials: int = Field(default=30)
    mlp_config: Optional[Dict[str, Any]] = Field(default=None, description="MLP architecture config for deep_mlp models")
    use_cv: bool = Field(default=True, description="Use cross-validation during evaluation")


class PipelineConfig(BaseModel):
    target_column: Optional[str] = Field(default=None, description="Target column (optional for unsupervised)")
    task_type: Literal["classification", "regression", "clustering", "anomaly_detection"] = Field(default="classification")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: List[ModelConfig]
    metric: str = Field(default="accuracy")
    test_size: float = Field(default=VALIDATION_SPLIT_RATIO)
    cv_folds: int = Field(default=CV_FOLDS)

    class Config:
        json_schema_extra = {
            "example": {
                "target_column": "target",
                "task_type": "classification",
                "preprocessing": {"fill_missing": "median", "encode_categorical": "onehot", "scale": True},
                "models": [{"name": "catboost", "tune_hyperparams": True, "n_trials": 50}],
                "metric": "accuracy",
            }
        }


class ModelPreset:
    """Presets for common model configurations"""
    
    @staticmethod
    def quick(model_name: str) -> ModelConfig:
        """Quick model config without tuning"""
        return ModelConfig(name=model_name, tune_hyperparams=False, use_cv=False)
    
    @staticmethod
    def tuned(model_name: str, n_trials: int = DEFAULT_N_TRIALS) -> ModelConfig:
        """Model config with hyperparameter tuning"""
        return ModelConfig(name=model_name, tune_hyperparams=True, n_trials=n_trials, use_cv=True)
    
    @staticmethod
    def deep_mlp(
        hidden_layers: List[int] = None,
        activation: str = MLP_DEFAULT_ACTIVATION,
        use_batchnorm: bool = MLP_DEFAULT_BATCHNORM,
        dropout_rate: float = MLP_DEFAULT_DROPOUT,
        optimizer: str = MLP_DEFAULT_OPTIMIZER,
        learning_rate: float = MLP_DEFAULT_LEARNING_RATE,
        batch_size: int = MLP_DEFAULT_BATCH_SIZE,
        epochs: int = MLP_DEFAULT_EPOCHS,
        early_stopping_patience: int = MLP_EARLY_STOPPING_PATIENCE,
        use_cv: bool = True
    ) -> ModelConfig:
        """Deep MLP model config with custom architecture"""
        mlp_config = {
            "hidden_layers": hidden_layers or MLP_PRESET_SMALL,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
            "dropout_rate": dropout_rate,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
        }
        return ModelConfig(
            name="deep_mlp_classifier",
            mlp_config=mlp_config,
            use_cv=use_cv
        )


class PipelinePreset:
    """Presets for common pipeline configurations"""
    
    @staticmethod
    def quick_classification(models: List[str] = None) -> PipelineConfig:
        """Quick classification pipeline without tuning"""
        if models is None:
            models = ["random_forest", "catboost"]
        
        return PipelineConfig(
            task_type="classification",
            models=[ModelPreset.quick(m) for m in models],
            metric="accuracy"
        )
    
    @staticmethod
    def tuned_classification(models: List[str] = None) -> PipelineConfig:
        """Classification pipeline with hyperparameter tuning"""
        if models is None:
            models = ["random_forest", "xgboost", "catboost", "deep_mlp_classifier"]
        
        return PipelineConfig(
            task_type="classification",
            models=[ModelPreset.tuned(m) for m in models],
            metric="accuracy"
        )
    
    @staticmethod
    def quick_regression(models: List[str] = None) -> PipelineConfig:
        """Quick regression pipeline without tuning"""
        if models is None:
            models = ["random_forest", "catboost"]
        
        return PipelineConfig(
            task_type="regression",
            models=[ModelPreset.quick(m) for m in models],
            metric="r2"
        )
    
    @staticmethod
    def deep_learning_classification(
        mlp_architecture: List[int] = None,
        epochs: int = 100
    ) -> PipelineConfig:
        """Classification pipeline focused on deep learning"""
        return PipelineConfig(
            task_type="classification",
            models=[
                ModelPreset.deep_mlp(
                    hidden_layers=mlp_architecture,
                    epochs=epochs
                )
            ],
            metric="accuracy"
        )
