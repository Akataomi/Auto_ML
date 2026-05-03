from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class PreprocessingConfig(BaseModel):
    fill_missing: str = Field(default="median", description="median, mean, or drop")
    encode_categorical: str = Field(default="onehot", description="onehot, label")
    scale: bool = Field(default=True)
    handle_outliers: bool = Field(default=False)


class ModelConfig(BaseModel):
    name: str = Field(..., description="model name from registry")
    tune_hyperparams: bool = Field(default=False)
    n_trials: int = Field(default=30)


class PipelineConfig(BaseModel):
    target_column: str
    task_type: Literal["classification", "regression"]
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: List[ModelConfig]
    metric: str = Field(default="accuracy")
    test_size: float = Field(default=0.2)
    cv_folds: int = Field(default=5)

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