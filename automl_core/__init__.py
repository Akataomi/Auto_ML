"""
AutoML Core Library.

Modular machine learning pipeline system with support for:
- Data loading and validation
- Preprocessing (cleaning, encoding, scaling)
- Model training with hyperparameter tuning
- Evaluation and reporting

Example:
    >>> from automl_core.pipeline import AutoMLPipeline, PipelineConfig
    >>> config = PipelineConfig(target_column="target", task_type="classification", models=[])
    >>> pipeline = AutoMLPipeline(config)
    >>> report = pipeline.run("data.csv")
"""

__version__ = "1.0.0"
__author__ = "AutoML Team"