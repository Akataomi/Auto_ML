"""
Constants and default configuration for AutoML framework.

Centralized constants for:
- Default hyperparameters
- Validation split ratios
- Early stopping parameters
- Cross-validation settings
- Model-specific defaults
"""

VALIDATION_SPLIT_RATIO = 0.2
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_FOR_STRATIFY = 10
CV_FOLDS = 5
CV_ENABLED_BY_DEFAULT = True
DEFAULT_N_TRIALS = 50
TUNING_ENABLED_BY_DEFAULT = False
BOOSTING_EARLY_STOPPING_ROUNDS = 50

MLP_EARLY_STOPPING_PATIENCE = 15
MLP_DEFAULT_HIDDEN_LAYERS = [128, 64]
MLP_DEFAULT_ACTIVATION = "relu"
MLP_DEFAULT_BATCHNORM = False
MLP_DEFAULT_DROPOUT = 0.2
MLP_DEFAULT_OPTIMIZER = "adam"
MLP_DEFAULT_LEARNING_RATE = 0.001
MLP_DEFAULT_BATCH_SIZE = 64
MLP_DEFAULT_EPOCHS = 100
MLP_DEFAULT_DEVICE = "cpu"


SGD_DEFAULT_EPOCHS = 100
SGD_LOG_INTERVAL = 20


CATBOOST_DEFAULT_ITERATIONS = 100
XGBOOST_DEFAULT_N_ESTIMATORS = 100
LIGHTGBM_DEFAULT_N_ESTIMATORS = 100


ITERATIVE_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "deep_mlp_classifier",
    "deep_mlp_regressor",
    "sgd_regressor",
    "sgd_classifier",
]
EARLY_STOPPING_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "deep_mlp_classifier",
    "deep_mlp_regressor",
]


CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "auc_pr",
]
REGRESSION_METRICS = [
    "r2",
    "mae",
    "mse",
    "rmse",
]
CLUSTERING_METRICS = [
    "silhouette_score",
    "n_clusters",
]
ANOMALY_METRICS = [
    "anomaly_ratio",
    "f1_score",
]


PCA_N_COMPONENTS = 2
MAX_VISUALIZATION_SAMPLES = 1000


DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"


DEFAULT_CLASSIFICATION_MODELS = ["random_forest", "catboost"]
DEFAULT_REGRESSION_MODELS = ["random_forest", "catboost"]
DEFAULT_CLUSTERING_MODELS = ["kmeans"]
DEFAULT_ANOMALY_MODELS = ["isolation_forest"]
MLP_ARCHITECTURE_PRESETS = {
    "Small": [32],
    "Medium": [64, 32],
    "Large": [128, 64, 32],
}
MLP_ACTIVATION_FUNCTIONS = [
    "relu",
    "leaky_relu",
    "gelu",
    "tanh",
    "sigmoid",
]
MLP_OPTIMIZERS = [
    "adam",
    "sgd",
    "rmsprop",
    "adamw",
]
MLP_LEARNING_RATES = {
    "0.1": 0.1,
    "0.01": 0.01,
    "0.001": 0.001,
    "0.0001": 0.0001,
}
MLP_BATCH_SIZES = [16, 32, 64, 128, 256]
MLP_EPOCH_OPTIONS = [25, 50, 100, 150, 200]
