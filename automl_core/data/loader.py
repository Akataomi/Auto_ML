"""
Data Loading Module.

Provides classes for loading and validating datasets from various file formats.
"""

import pandas as pd
from pathlib import Path
from typing import Union


class DataLoader:
    """
    Loader for datasets with basic validation.
    
    Supported formats: CSV, Parquet
    
    Example:
        >>> df = DataLoader.load("dataset.csv")
        >>> DataLoader.validate(df, target_col="target")
    """

    @staticmethod
    def load(filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load a dataset from file.
        
        Args:
            filepath: Path to the data file (CSV or Parquet).
            
        Returns:
            pd.DataFrame: Loaded dataset.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported or empty.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {path.suffix}")

        if df.empty:
            raise ValueError("Датасет пустой")

        return df

    @staticmethod
    def validate(df: pd.DataFrame, target_col: str) -> bool:
        """
        Validate dataset for ML pipeline compatibility.
        
        Args:
            df: DataFrame to validate.
            target_col: Name of the target variable column.
            
        Returns:
            bool: True if validation passes.
            
        Raises:
            ValueError: If validation fails.
        """
        if target_col not in df.columns:
            raise ValueError(f"Целевая колонка '{target_col}' не найдена")
        if df[target_col].isna().all():
            raise ValueError("Целевая колонка содержит только NaN")
        if df.shape[0] < 10:
            raise ValueError("Слишком мало записей для обучения")
        return True