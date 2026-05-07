import optuna
from sklearn.model_selection import cross_val_score
from typing import Callable, Dict, Any


class HyperparameterTuner:
    """Оптимизация гиперпараметров через Optuna"""

    def __init__(self, model_class: Callable, task_type: str, n_trials: int = 50):
        self.model_class = model_class
        self.task_type = task_type
        self.n_trials = n_trials
        self.best_params = {}
        self.best_score = 0

    def _get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Пространство параметров для разных моделей"""
        model_name = self.model_class.__name__.lower()

        if "xgb" in model_name or "xgboost" in model_name:
            return {
                "n_estimators": trial.randint(100, 500),
                "max_depth": trial.randint(3, 10),
                "learning_rate": trial.loguniform(0.01, 0.3),
                "subsample": trial.uniform(0.7, 1.0),
                "colsample_bytree": trial.uniform(0.7, 1.0),
            }
        elif "lgbm" in model_name or "lightgbm" in model_name:
            return {
                "n_estimators": trial.randint(100, 500),
                "max_depth": trial.randint(3, 10),
                "learning_rate": trial.loguniform(0.01, 0.3),
                "num_leaves": trial.randint(20, 100),
                "subsample": trial.uniform(0.7, 1.0),
            }
        elif "catboost" in model_name:
            return {
                "iterations": trial.randint(100, 500),
                "depth": trial.randint(3, 10),
                "learning_rate": trial.loguniform(0.01, 0.3),
                "l2_leaf_reg": trial.loguniform(1, 10),
            }
        elif "randomforest" in model_name or "random_forest" in model_name:
            return {
                "n_estimators": trial.randint(100, 500),
                "max_depth": trial.randint(3, 20),
                "min_samples_split": trial.randint(2, 20),
                "min_samples_leaf": trial.randint(1, 10),
            }
        else:
            return {}

    def _get_scoring(self) -> str:
        """Метрика для оптимизации"""
        if self.task_type == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"

    def _objective(self, trial: optuna.Trial, X, y, cv: int = 5):
        """Целевая функция для Optuna"""
        params = self._get_search_space(trial)
        model = self.model_class(**params)

        scoring = self._get_scoring()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        if self.task_type == "regression":
            return -scores.mean()  # Инвертируем для MSE
        return scores.mean()

    def tune(self, X, y, cv: int = 5) -> Dict[str, Any]:
        """Запуск оптимизации"""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, X, y, cv), n_trials=self.n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        return {"best_params": self.best_params, "best_score": self.best_score}

    def get_best_model(self) -> Any:
        """Получение модели с лучшими параметрами"""
        return self.model_class(**self.best_params)