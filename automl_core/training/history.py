"""Training History"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Stores training metrics per epoch/iteration"""
    
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metric: List[float] = field(default_factory=list)
    val_metric: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    
    def add(self, epoch: int, train_loss: float = None, val_loss: float = None,
            train_metric: float = None, val_metric: float = None):
        """Add metrics for one epoch"""
        self.epochs.append(epoch)
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if train_metric is not None:
            self.train_metric.append(train_metric)
        if val_metric is not None:
            self.val_metric.append(val_metric)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for plotting"""
        data = {"epoch": self.epochs}
        if self.train_loss:
            data["train_loss"] = self.train_loss
        if self.val_loss:
            data["val_loss"] = self.val_loss
        if self.train_metric:
            data["train_metric"] = self.train_metric
        if self.val_metric:
            data["val_metric"] = self.val_metric
        return pd.DataFrame(data)
    
    def is_empty(self) -> bool:
        return len(self.epochs) == 0
    
    def get_best_epoch(self, metric: str = "val_metric", mode: str = "max") -> int:
        """Get epoch with best metric"""
        if self.is_empty():
            return 0
        
        df = self.to_dataframe()
        if metric not in df.columns:
            return 0
        
        if mode == "max":
            return df.loc[df[metric].idxmax(), "epoch"]
        else:
            return df.loc[df[metric].idxmin(), "epoch"]