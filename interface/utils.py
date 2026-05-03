"""
Utility Functions.

Helper functions for the interface.
"""

import pandas as pd
from typing import Tuple


def detect_task_type(df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
    """
    Heuristically detect task type based on target column.
    
    Args:
        df: DataFrame with data.
        target_col: Name of target column.
        
    Returns:
        Tuple of (task_type, reason)
    """
    target_data = df[target_col]
    unique_count = target_data.nunique()
    total_count = len(target_data)
    
    if target_data.dtype in ["object", "category", "bool"]:
        return "classification", "Строковый/категориальный тип данных"
    
    if unique_count == 2:
        return "classification", "Бинарная классификация (2 уникальных значения)"
    
    if unique_count < 10 and unique_count / total_count < 0.05:
        return "classification", f"Мало уникальных значений ({unique_count}) — вероятно классификация"
    
    if unique_count >= 10 and target_data.dtype in ["int64", "float64"]:
        return "regression", f"Много уникальных значений ({unique_count}) — вероятно регрессия"
    
    return "classification", "По умолчанию (классификация)"


def format_metric(value) -> str:
    """Format metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value) if value is not None else "-"


def get_model_emoji(task_type: str) -> str:
    """Get emoji for task type."""
    return "📌" if task_type == "classification" else "📈"


def get_model_color(task_type: str) -> str:
    """Get color indicator for task type."""
    return "🟢" if task_type == "classification" else "🔵"