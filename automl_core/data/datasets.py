"""
Встроенные тестовые датасеты для демонстрации AutoML.

Доступны датасеты для всех типов задач:
- Classification
- Regression
- Clustering
- Anomaly Detection
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_diabetes, fetch_california_housing
)


def get_classification_datasets():
    """Вернуть список датасетов для классификации."""
    return [
        {
            "name": "Iris (Цветы)",
            "id": "iris",
            "description": "150 образцов, 3 вида ирисов, 4 признака",
            "samples": 150,
            "features": 4,
            "classes": 3,
            "target": "species"
        },
        {
            "name": "Wine (Вина)",
            "id": "wine",
            "id": "wine",
            "description": "178 образцов вин, 3 сорта, 13 признаков",
            "samples": 178,
            "features": 13,
            "classes": 3,
            "target": "class"
        },
        {
            "name": "Breast Cancer (Рак)",
            "id": "cancer",
            "description": "569 образцов, 2 класса (benign/malignant), 30 признаков",
            "samples": 569,
            "features": 30,
            "classes": 2,
            "target": "diagnosis"
        }
    ]


def get_regression_datasets():
    """Вернуть список датасетов для регрессии."""
    return [
        {
            "name": "Diabetes (Диабет)",
            "id": "diabetes",
            "description": "442 пациента, прогноз прогрессирования диабета",
            "samples": 442,
            "features": 10,
            "target": "target"
        }
    ]


def get_clustering_datasets():
    """Вернуть список датасетов для кластеризации."""
    return [
        {
            "name": "Iris (без меток)",
            "id": "iris_unsupervised",
            "description": "Iris dataset без target для кластеризации",
            "samples": 150,
            "features": 4,
            "true_clusters": 3
        },
        {
            "name": "Wine (без меток)",
            "id": "wine_unsupervised",
            "description": "Wine dataset без target для кластеризации",
            "samples": 178,
            "features": 13,
            "true_clusters": 3
        }
    ]


def get_anomaly_datasets():
    """Вернуть список датасетов для детекции аномалий."""
    return [
        {
            "name": "Breast Cancer (аномалии)",
            "id": "cancer_anomaly",
            "description": "Cancer dataset: malignant как аномалии",
            "samples": 569,
            "features": 30,
            "anomaly_ratio": 0.37
        },
        {
            "name": "Diabetes (аномалии)",
            "id": "diabetes_anomaly",
            "description": "Diabetes dataset с синтетическими аномалиями",
            "samples": 442,
            "features": 10,
            "anomaly_ratio": 0.1
        }
    ]


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Загрузить датасет по ID и вернуть как DataFrame.

    Args:
        dataset_id: ID датасета (iris, wine, cancer, diabetes, housing)
    
    Returns:
        pd.DataFrame с данными
    """

    if dataset_id == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['species'] = data.target_names[data.target]
        return df
    
    elif dataset_id == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['class'] = data.target_names[data.target]
        return df
    
    elif dataset_id == "cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['diagnosis'] = data.target_names[data.target]
        return df

    elif dataset_id == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_id == "housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['MedHouseVal'] = data.target
        return df

    elif dataset_id == "iris_unsupervised":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df
    
    elif dataset_id == "wine_unsupervised":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df

    elif dataset_id == "cancer_anomaly":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['anomaly'] = data.target
        return df
    
    elif dataset_id == "diabetes_anomaly":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)

        np.random.seed(42)
        n_anomalies = int(len(df) * 0.1)
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)

        df['anomaly'] = 0
        df.loc[anomaly_indices, 'anomaly'] = 1
        return df
    else:
        raise ValueError(f"Unknown dataset ID: {dataset_id}")


def get_all_datasets():
    """Вернуть все доступные датасеты."""
    return {
        "classification": get_classification_datasets(),
        "regression": get_regression_datasets(),
        "clustering": get_clustering_datasets(),
        "anomaly_detection": get_anomaly_datasets()
    }
