"""
MLP (Multi-Layer Perceptron) Models.

Simple wrappers around sklearn's MLP models for compatibility.
"""

from typing import Optional
from sklearn.neural_network import MLPClassifier as SKMLPClassifier
from sklearn.neural_network import MLPRegressor as SKMLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as np


class MLPClassifier:
    """
    Multi-Layer Perceptron for classification.
    Compatible wrapper around sklearn's MLPClassifier.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
    ):
        
        self._model = SKMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=42 if random_state is None else int(random_state),
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
        )
        
        self._history = {
            "losses": [],
            "val_losses": [],
            "epochs": 0
        }
    
    def fit(self, X, y):
        """Fit the model."""
        self._model.fit(X, y)

        self._history = {
            "losses": self._model.loss_curve_ if hasattr(self._model, 'loss_curve_') else [],
            "val_losses": [],
            "epochs": self._model.n_iter_ if hasattr(self._model, 'n_iter_') else 0
        }
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def score(self, X, y):
        """Return accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    @property
    def classes_(self):
        """Get unique classes."""
        return self._model.classes_
    
    @property
    def loss_curve_(self):
        """Get loss curve."""
        return self._model.loss_curve_
    
    @property
    def n_iter_(self):
        """Get number of iterations."""
        return self._model.n_iter_


class MLPRegressor:
    """
    Multi-Layer Perceptron for regression.
    Compatible wrapper around sklearn's MLPRegressor.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
    ):
        
        self._model = SKMLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=42 if random_state is None else int(random_state),
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
        )
        
        self._history = {
            "losses": [],
            "val_losses": [],
            "epochs": 0
        }
    
    def fit(self, X, y):
        """Fit the model."""
        y = np.array(y).ravel()
        self._model.fit(X, y)

        self._history = {
            "losses": self._model.loss_curve_ if hasattr(self._model, 'loss_curve_') else [],
            "val_losses": [],
            "epochs": self._model.n_iter_ if hasattr(self._model, 'n_iter_') else 0
        }
        
        return self
    
    def predict(self, X):
        """Predict continuous values."""
        return self._model.predict(X)
    
    def score(self, X, y):
        """Return R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    @property
    def loss_curve_(self):
        """Get loss curve."""
        return self._model.loss_curve_
    
    @property
    def n_iter_(self):
        """Get number of iterations."""
        return self._model.n_iter_
