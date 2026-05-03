import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataAnalyzer:
    """Первичный анализ данных (EDA)"""

    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict:
        """Базовая статистика по датасету"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isna().sum().to_dict(),
            "missing_pct": (df.isna().sum() / len(df) * 100).round(2).to_dict(),
        }

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Классификация колонок по типам"""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()

        return {"numeric": numeric, "categorical": categorical, "datetime": datetime}

    @staticmethod
    def get_target_info(df: pd.DataFrame, target_col: str) -> Dict:
        """Анализ целевой переменной"""
        target = df[target_col]
        info = {
            "type": str(target.dtype),
            "missing": int(target.isna().sum()),
            "unique": int(target.nunique()),
        }

        if target.dtype in ["object", "category", "bool"]:
            info["class_distribution"] = target.value_counts().to_dict()
            info["task_type"] = "classification"
        else:
            info["mean"] = float(target.mean())
            info["std"] = float(target.std())
            info["min"] = float(target.min())
            info["max"] = float(target.max())
            info["task_type"] = "regression"

        return info