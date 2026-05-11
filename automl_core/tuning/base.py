"""Base Tuner Interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTuner(ABC):
    """Abstract tuner interface"""
    
    @abstractmethod
    def tune(self, model_name: str, task_type: str, X, y, **kwargs) -> Dict[str, Any]:
        """Run optimization"""
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict]:
        """Get optimization history"""
        pass