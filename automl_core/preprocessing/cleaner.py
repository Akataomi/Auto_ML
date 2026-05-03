import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple


class DataCleaner:
    """Очистка данных: пропуски, выбросы"""

    def __init__(self, fill_strategy: str = "median", outlier_method: str = None):
        self.fill_strategy = fill_strategy
        self.outlier_method = outlier_method
        self.imputers = {}
        self.outlier_bounds = {}

    def fit(self, df: pd.DataFrame, numeric_cols: list):
        """Подготовка обработчиков"""
        for col in numeric_cols:
            if self.fill_strategy == "median":
                strategy = "median"
            elif self.fill_strategy == "mean":
                strategy = "mean"
            else:
                strategy = "most_frequent"

            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(df[[col]])
            self.imputers[col] = imputer

            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        return self

    def transform(self, df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """Применение очистки"""
        df_clean = df.copy()

        for col in numeric_cols:
            if col in self.imputers:
                df_clean[[col]] = self.imputers[col].transform(df_clean[[col]])

            if self.outlier_method == "iqr" and col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                df_clean[col] = df_clean[col].clip(lower, upper)

        return df_clean

    def fit_transform(self, df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """Fit + Transform"""
        self.fit(df, numeric_cols)
        return self.transform(df, numeric_cols)