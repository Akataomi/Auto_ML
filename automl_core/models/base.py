from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class ModelConfig(ABC):
    """Configuration for a single model"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier"""
        pass
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """'classification' or 'regression'"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Model category: 'linear', 'tree', 'boosting', 'neural'"""
        pass
    
    @abstractmethod
    def get_search_space(self, trial) -> Dict[str, Any]:
        """Define hyperparameter search space for Optuna"""
        pass
    
    @abstractmethod
    def create_model(self, **params) -> Any:
        """Create model instance with parameters"""
        pass


class TrainableModel(ABC):
    """Interface for a trained model wrapper"""
    
    @abstractmethod
    def fit(self, X, y):
        "Train the model"
        pass
    
    @abstractmethod
    def predict(self, X):
        "Make predictions"
        pass
    
    @property
    @abstractmethod
    def feature_importance(self) -> Optional[np.ndarray]:
        "Feature importance if available"
        pass
    
    @property
    @abstractmethod
    def coefficients(self) -> Optional[np.ndarray]:
        """Coefficients for linear models"""
        pass
    
    def get_feature_importance_df(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        importance = self.feature_importance
        if importance is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
    
    def get_coefficients_df(self, feature_names: List[str]) -> pd.DataFrame:
        """Get coefficients as DataFrame"""
        coef = self.coefficients
        if coef is None:
            return pd.DataFrame()
        flat_coef = coef.flatten() if hasattr(coef, 'flatten') else coef
        return pd.DataFrame({
            "feature": feature_names,
            "coefficient": flat_coef
        }).sort_values("coefficient", key=abs, ascending=False)