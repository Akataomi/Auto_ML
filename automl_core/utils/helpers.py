"""
Helper Utilities.

Common utility functions used across the AutoML framework:
- Data validation
- Type checking
- Format conversion
- Common operations
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Any


def ensure_numpy_array(
    data: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    name: str = "data"
) -> np.ndarray:
    """
    Convert input to numpy array.
    
    Args:
        data: Input data (array, Series, DataFrame, or list)
        name: Name for error messages
        
    Returns:
        Numpy array
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"{name} must be array-like, got {type(data)}")


def validate_data_shape(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    min_samples: int = 10,
    min_features: int = 1
) -> None:
    """
    Validate data shapes for training.
    
    Args:
        X: Features array
        y: Target array (optional)
        min_samples: Minimum number of samples
        min_features: Minimum number of features
        
    Raises:
        ValueError: If data validation fails
    """
    if len(X.shape) != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    n_samples, n_features = X.shape
    
    if n_samples < min_samples:
        raise ValueError(
            f"Too few samples: {n_samples} (minimum: {min_samples})"
        )
    
    if n_features < min_features:
        raise ValueError(
            f"Too few features: {n_features} (minimum: {min_features})"
        )
    
    if y is not None:
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        
        if len(y) != n_samples:
            raise ValueError(
                f"X and y have different number of samples: "
                f"{n_samples} vs {len(y)}"
            )


def safe_convert_to_float(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    default_value: float = 0.0
) -> np.ndarray:
    """
    Safely convert data to float, handling NaN and non-numeric values.
    
    Args:
        data: Input data
        default_value: Value to use for non-convertible entries
        
    Returns:
        Float array
    """
    arr = ensure_numpy_array(data)
    
    try:
        return arr.astype(np.float64)
    except (ValueError, TypeError):
        # If conversion fails, fill with default value
        return np.full_like(arr, default_value, dtype=np.float64)


def get_array_shape(data: Any) -> Tuple:
    """
    Get shape of array-like object safely.
    
    Args:
        data: Array-like object
        
    Returns:
        Shape tuple
    """
    if hasattr(data, 'shape'):
        return data.shape
    elif isinstance(data, list):
        return (len(data),)
    else:
        return ()


def is_unsupervised_task(task_type: str) -> bool:
    """
    Check if task is unsupervised (no target required).
    
    Args:
        task_type: Task type string
        
    Returns:
        True if unsupervised
    """
    return task_type in ['clustering', 'anomaly_detection']


def get_default_metric_name(task_type: str) -> str:
    """
    Get default metric name for task type.
    
    Args:
        task_type: Task type string
        
    Returns:
        Metric name
    """
    metrics = {
        'classification': 'accuracy',
        'regression': 'r2',
        'clustering': 'silhouette',
        'anomaly_detection': 'f1'
    }
    return metrics.get(task_type, 'accuracy')


def get_default_metric_display_name(task_type: str) -> str:
    """
    Get human-readable metric name for task type.
    
    Args:
        task_type: Task type string
        
    Returns:
        Display name
    """
    metrics = {
        'classification': 'Accuracy',
        'regression': 'R² Score',
        'clustering': 'Silhouette Score',
        'anomaly_detection': 'F1 Score'
    }
    return metrics.get(task_type, 'Accuracy')
