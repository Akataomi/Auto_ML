"""Training Callbacks"""

from typing import Optional, Callable
from .history import TrainingHistory


class TrainingCallback:
    """Base callback for training monitoring."""
    
    def __init__(self, history: TrainingHistory):
        self.history = history
    
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Called at the end of each epoch"""
        pass


class BoostingCallback(TrainingCallback):
    """Callback for XGBoost/LightGBM/CatBoost"""
    
    def __init__(self, history: TrainingHistory, metric_name: str = "accuracy"):
        super().__init__(history)
        self.metric_name = metric_name
        self.best_iteration = 0
        self.best_score = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Store metrics from boosting model"""
        if logs:
            train_loss = logs.get("train_loss", logs.get("train_logloss", logs.get("train_rmse")))
            val_loss = logs.get("val_loss", logs.get("val_logloss", logs.get("val_rmse")))
            train_metric = logs.get(f"train_{self.metric_name}", logs.get("train_accuracy"))
            val_metric = logs.get(f"val_{self.metric_name}", logs.get("validation_accuracy"))
            
            self.history.add(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metric=train_metric,
                val_metric=val_metric
            )


class EarlyStoppingCallback:
    """Early stopping based on validation metric"""
    
    def __init__(self, patience: int = 10, metric: str = "val_metric", mode: str = "max"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def check(self, score: float, epoch: int) -> bool:
        """Check if training should stop"""
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop