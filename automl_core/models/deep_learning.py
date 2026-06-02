"""
Deep Learning модуль на PyTorch с гибкой архитектурой.
Поддержка:
- Конфигурируемые слои (количество, размер)
- BatchNorm, Dropout, Activation
- Разные оптимизаторы и scheduler'ы
- Early stopping
- Графики обучения (loss curves)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings
from typing import List, Dict, Any, Optional

DEBUG = False
warnings.filterwarnings('ignore', category=UserWarning)


def debug_print(msg):
    if DEBUG:
        print(f"[DeepMLP DEBUG] {msg}")


class MLPBlock(nn.Module):
    """Один блок MLP: Linear -> (BatchNorm) -> Activation -> (Dropout)"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'relu',
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        
        activation_fn = self._get_activation(activation)
        if activation_fn is not None:
            layers.append(activation_fn)
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> Optional[nn.Module]:
        
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'none': None
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x):
        return self.block(x)


class DeepMLP(nn.Module):
    """
    Гибкая архитектура MLP с конфигурируемыми слоями.
    
    Пример конфига:
    {
        "input_dim": 100,
        "output_dim": 10,
        "hidden_layers": [256, 128, 64],
        "activation": "relu",
        "use_batchnorm": True,
        "dropout_rate": 0.3,
        "task_type": "classification"  # или "regression"
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        hidden_layers = config.get('hidden_layers', [128, 64])
        activation = config.get('activation', 'relu')
        use_batchnorm = config.get('use_batchnorm', False)
        dropout_rate = config.get('dropout_rate', 0.0)
        task_type = config.get('task_type', 'classification')
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(MLPBlock(
                prev_dim,
                hidden_dim,
                activation=activation,
                use_batchnorm=use_batchnorm,
                dropout_rate=dropout_rate
            ))
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        if task_type == 'classification':
            self.output_layer = nn.Linear(prev_dim, output_dim)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.output_layer = nn.Linear(prev_dim, output_dim)
            self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        x = self.backbone(x)
        return self.output_layer(x)


class DeepMLPTrainer:
    """
    Тренер для DeepMLP с поддержкой:
    - Разных оптимизаторов
    - Learning rate scheduler'ов
    - Early stopping
    - Истории обучения (для графиков)
    """
    
    def __init__(
        self,
        model: DeepMLP,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler: str = 'none',
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0001,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.optimizer = self._create_optimizer(optimizer, learning_rate, weight_decay)
        self.scheduler = self._create_scheduler(scheduler)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'epochs': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def _create_optimizer(self, name: str, lr: float, weight_decay: float):
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'sgd': optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
            'rmsprop': optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'adamw': optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        }
        return optimizers.get(name, optim.Adam(self.model.parameters(), lr=lr))
    
    def _create_scheduler(self, name: str):
        schedulers = {
            'none': None,
            'step': optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5),
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        }
        return schedulers.get(name, None)
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str):
        """Вычисление метрики (accuracy для classification, R² для regression)"""
        if task_type == 'classification':
            predicted_classes = np.argmax(y_pred, axis=1)
            return np.mean(predicted_classes == y_true)
        else:
            ss_res = np.sum((y_true - y_pred.flatten()) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Обучение модели.
        
        Returns:
            history: Dict с train_loss, val_loss, train_metric, val_metric
        """
        if verbose:
            print(f"[DeepMLP] Starting training: hidden={self.model.config.get('hidden_layers')}, act={self.model.config.get('activation')}, opt={self.optimizer.__class__.__name__.lower()}, lr={self.optimizer.defaults['lr']:.4f}")
        
        task_type = self.model.config.get('task_type', 'classification')

        X_train_array = np.asarray(X_train)
        y_train_array = np.asarray(y_train)
        
        X_tensor = torch.FloatTensor(X_train_array).to(self.device)
        y_tensor = torch.LongTensor(y_train_array) if task_type == 'classification' else torch.FloatTensor(y_train_array)
        y_tensor = y_tensor.to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        # drop_last=True чтобы избежать batch_size=1 который ломает BatchNorm
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Валидационные данные
        if X_val is not None and y_val is not None:
            X_val_array = np.asarray(X_val)
            y_val_array = np.asarray(y_val)
            
            X_val_tensor = torch.FloatTensor(X_val_array).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_array) if task_type == 'classification' else torch.FloatTensor(y_val_array)
            y_val_tensor = y_val_tensor.to(self.device)
        
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(self.epochs):
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(loader):
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    
                    if task_type == 'classification':
                        loss = self.model.loss_fn(outputs, batch_y)
                    else:
                        loss = self.model.loss_fn(outputs, batch_y.unsqueeze(1) if batch_y.dim() == 1 else batch_y)
                    
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    train_preds.append(outputs.detach().cpu().numpy())
                    train_targets.append(batch_y.cpu().numpy())
                except Exception as e:
                    debug_print(f"ERROR in batch {batch_idx}: {e}")
                    import traceback
                    debug_print(traceback.format_exc())
                    raise
            
            train_loss /= len(loader)
            train_preds = np.vstack(train_preds)
            train_targets = np.concatenate(train_targets)
            
            if task_type == 'classification':
                train_metric = self._compute_metric(train_targets, train_preds, task_type)
            else:
                train_metric = self._compute_metric(train_targets, train_preds.flatten(), task_type)
            
            val_loss = 0.0
            val_metric = 0.0
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    
                    if task_type == 'classification':
                        val_loss = self.model.loss_fn(val_outputs, y_val_tensor).item()
                        val_preds = val_outputs.cpu().numpy()
                        val_y = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else np.array(y_val)
                        val_metric = self._compute_metric(val_y, val_preds, task_type)
                    else:
                        val_loss = self.model.loss_fn(
                            val_outputs,
                            y_val_tensor.unsqueeze(1) if y_val_tensor.dim() == 1 else y_val_tensor
                        ).item()
                        val_preds = val_outputs.cpu().numpy()
                        val_y = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else np.array(y_val)
                        val_metric = self._compute_metric(val_y, val_preds.flatten(), task_type)
                
                self.model.train()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss if val_loss > 0 else train_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['epochs'].append(epoch + 1)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss > 0 else train_loss)
                else:
                    self.scheduler.step()

            if val_loss > 0 and val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            elif val_loss > 0:
                self.patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", val_loss: {val_loss:.4f}, val_metric: {val_metric:.4f}" if val_loss > 0 else ""
                print(f"[DeepMLP] Epoch {epoch+1}/{self.epochs} - loss: {train_loss:.4f}, metric: {train_metric:.4f}{val_str}")
            
            if self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"[DeepMLP] Early stopping at epoch {epoch+1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X)
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.model.config.get('task_type', 'classification') == 'classification':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.cpu().numpy().flatten()
        
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей (для classification)"""
        if self.model.config.get('task_type', 'classification') != 'classification':
            raise ValueError("predict_proba available only for classification")

        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X)
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
