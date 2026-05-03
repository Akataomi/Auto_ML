import pytest
import pandas as pd
import tempfile
import os

from automl_core.data.loader import DataLoader
from automl_core.pipeline.config import PipelineConfig, PreprocessingConfig, ModelConfig
from automl_core.pipeline.orchestrator import AutoMLPipeline


def test_data_loader():
    """Тест загрузки данных"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
        tmp_path = f.name

    df = DataLoader.load(tmp_path)
    assert df.shape == (4, 3)
    os.remove(tmp_path)


def test_pipeline_config():
    """Тест валидации конфига"""
    config = PipelineConfig(
        target_column="target",
        task_type="classification",
        preprocessing=PreprocessingConfig(),
        models=[ModelConfig(name="random_forest", tune_hyperparams=False)],
    )
    assert config.target_column == "target"
    assert config.task_type == "classification"


def test_orchestrator():
    """Тест полного пайплайна"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n")
        for i in range(20):
            target = i % 2
            f.write(f"{i*2},{i*3},{target}\n")
        tmp_path = f.name

    config = PipelineConfig(
        target_column="target",
        task_type="classification",
        preprocessing=PreprocessingConfig(),
        models=[ModelConfig(name="random_forest", tune_hyperparams=False)],
    )

    pipeline = AutoMLPipeline(config)
    report = pipeline.run(tmp_path)

    assert "results" in report
    assert "data_info" in report
    assert len(report["results"]) > 0

    os.remove(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])