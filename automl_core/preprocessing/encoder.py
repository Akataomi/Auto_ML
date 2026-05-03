import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple, List


class DataEncoder:
    """Кодирование и масштабирование признаков"""

    def __init__(self, categorical_strategy: str = "onehot", scale: bool = True):
        self.categorical_strategy = categorical_strategy
        self.scale = scale
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = None
        self.categorical_cols = []
        self.numeric_cols = []

    def fit(self, df: pd.DataFrame, target_col: str, categorical_cols: list, numeric_cols: list):
        """Подготовка энкодеров"""
        self.categorical_cols = [c for c in categorical_cols if c != target_col]
        self.numeric_cols = [c for c in numeric_cols if c != target_col]

        # Кодирование категориальных
        if self.categorical_strategy == "label":
            for col in self.categorical_cols:
                le = LabelEncoder()
                df_col = df[col].astype(str).fillna("Unknown")
                le.fit(df_col)
                self.label_encoders[col] = le

        elif self.categorical_strategy == "onehot":
            if self.categorical_cols:
                self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.onehot_encoder.fit(df[self.categorical_cols].fillna("Unknown"))

        # Масштабирование числовых
        if self.scale and self.numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numeric_cols])

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Применение кодирования"""
        df_processed = df.copy()

        # Кодирование категориальных
        if self.categorical_strategy == "label":
            for col, le in self.label_encoders.items():
                df_processed[col] = le.transform(df_processed[col].astype(str).fillna("Unknown"))

        elif self.categorical_strategy == "onehot":
            if self.categorical_cols and self.onehot_encoder:
                ohe_array = self.onehot_encoder.transform(
                    df_processed[self.categorical_cols].fillna("Unknown")
                )
                ohe_df = pd.DataFrame(
                    ohe_array,
                    columns=self.onehot_encoder.get_feature_names_out(self.categorical_cols),
                    index=df_processed.index,
                )
                df_processed = df_processed.drop(columns=self.categorical_cols)
                df_processed = pd.concat([df_processed, ohe_df], axis=1)

        # Масштабирование числовых
        if self.scale and self.numeric_cols and self.scaler:
            df_processed[self.numeric_cols] = self.scaler.transform(df_processed[self.numeric_cols])

        return df_processed

    def fit_transform(
        self, df: pd.DataFrame, target_col: str, categorical_cols: list, numeric_cols: list
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Полная обработка"""
        self.fit(df, target_col, categorical_cols, numeric_cols)
        df_processed = self.transform(df)

        X = df_processed.drop(columns=[target_col])
        y = df[target_col]

        return X, y, list(X.columns)