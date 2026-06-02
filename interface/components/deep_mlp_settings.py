"""
Deep MLP Settings Component.

Allows user to configure neural network architecture:
- Hidden layers
- Activation functions
- Batch normalization
- Dropout
- Optimizer and learning rate
- Early stopping
"""

import streamlit as st
from typing import Dict, Any, List
from automl_core.config.constants import (
    MLP_ARCHITECTURE_PRESETS,
    MLP_DEFAULT_BATCHNORM,
    MLP_DEFAULT_DROPOUT,
    MLP_DEFAULT_LEARNING_RATE,
    MLP_DEFAULT_BATCH_SIZE,
    MLP_DEFAULT_EPOCHS,
    MLP_EARLY_STOPPING_PATIENCE,
)

MLP_PRESET_SMALL = MLP_ARCHITECTURE_PRESETS["Small"]
MLP_PRESET_MEDIUM = MLP_ARCHITECTURE_PRESETS["Medium"]
MLP_PRESET_LARGE = MLP_ARCHITECTURE_PRESETS["Large"]


def render_deep_mlp_settings(task_type: str) -> Dict[str, Any]:
    """
    Render Deep MLP configuration UI in sidebar.
    
    Args:
        task_type: 'classification' or 'regression'
        
    Returns:
        Dict with MLP configuration parameters
    """
    st.sidebar.subheader("🧠 Настройки нейросети")
    
    with st.sidebar.expander("⚙️ Архитектура MLP", expanded=False):
        # Hidden layers
        st.markdown("**Скрытые слои**")
        layer_preset = st.selectbox(
            "Пресет архитектуры",
            options=["small", "medium", "large", "custom"],
            format_func=lambda x: {
                "small": "Small (64)",
                "medium": "Medium (128, 64)",
                "large": "Large (256, 128, 64)",
                "custom": "⚙️ Custom"
            }[x],
            key="mlp_layer_preset"
        )
        
        if layer_preset == "small":
            default_layers = MLP_PRESET_SMALL
        elif layer_preset == "medium":
            default_layers = MLP_PRESET_MEDIUM
        elif layer_preset == "large":
            default_layers = MLP_PRESET_LARGE
        else:
            default_layers = MLP_PRESET_MEDIUM  # fallback to medium
        
        if layer_preset == "custom":
            custom_layers = st.text_input(
                "Размеры слоев (через запятую)",
                value=",".join(map(str, default_layers)),
                key="mlp_custom_layers"
            )
            try:
                hidden_layers = [int(x.strip()) for x in custom_layers.split(",") if x.strip()]
                if not hidden_layers:
                    hidden_layers = [128, 64]
            except ValueError:
                st.error("Неверный формат! Используйте числа через запятую")
                hidden_layers = [128, 64]
        else:
            hidden_layers = default_layers
        
        st.markdown(f"**Текущая архитектура:** {[hidden_layers]}")

        activation = st.selectbox(
            "Функция активации",
            options=["relu", "leaky_relu", "gelu", "tanh", "sigmoid"],
            format_func=lambda x: {
                "relu": "ReLU (по умолчанию)",
                "leaky_relu": "Leaky ReLU",
                "gelu": "GELU",
                "tanh": "Tanh",
                "sigmoid": "Sigmoid"
            }[x],
            index=0,
            key="mlp_activation"
        )

        use_batchnorm = st.checkbox(
            "Batch Normalization",
            value=MLP_DEFAULT_BATCHNORM,
            key="mlp_batchnorm"
        )

        st.markdown("**Dropout**")
        dropout_rate = st.slider(
            "Dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=MLP_DEFAULT_DROPOUT,
            step=0.1,
            key="mlp_dropout"
        )
        
    with st.sidebar.expander("⚡ Оптимизация", expanded=False):
        optimizer = st.selectbox(
            "Оптимизатор",
            options=["adam", "sgd", "rmsprop", "adamw"],
            format_func=lambda x: {
                "adam": "Adam (по умолчанию)",
                "sgd": "SGD",
                "rmsprop": "RMSprop",
                "adamw": "AdamW"
            }[x],
            index=0,
            key="mlp_optimizer"
        )

        learning_rate = st.selectbox(
            "Learning rate",
            options=[0.1, 0.01, 0.001, 0.0001],
            format_func=lambda x: f"{x:.4f}",
            index=[0.1, 0.01, 0.001, 0.0001].index(MLP_DEFAULT_LEARNING_RATE),
            key="mlp_lr"
        )
        
        batch_size = st.selectbox(
            "Размер батча",
            options=[16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(MLP_DEFAULT_BATCH_SIZE),
            key="mlp_batch_size"
        )
        
    with st.sidebar.expander("📊 Обучение", expanded=False):
        epochs = st.slider(
            "Количество эпох",
            min_value=25,
            max_value=200,
            value=MLP_DEFAULT_EPOCHS,
            step=25,
            key="mlp_epochs"
        )

        early_stopping = st.checkbox(
            "Early stopping",
            value=True,
            key="mlp_early_stopping"
        )
        
        if early_stopping:
            patience = st.slider(
                "Patience (эпох)",
                min_value=5,
                max_value=50,
                value=MLP_EARLY_STOPPING_PATIENCE,
                step=5,
                key="mlp_patience"
            )
        else:
            patience = MLP_DEFAULT_EPOCHS

    config = {
        "hidden_layers": hidden_layers,
        "activation": activation,
        "use_batchnorm": use_batchnorm,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": patience if early_stopping else epochs
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧠 Конфигурация MLP:**")
    summary = (
        f"• Слои: {config['hidden_layers']}\n"
        f"• Activation: {config['activation']}\n"
        f"• Dropout: {config['dropout_rate']}\n"
        f"• Optimizer: {config['optimizer']} (lr={config['learning_rate']})\n"
        f"• Epochs: {config['epochs']}, Batch: {config['batch_size']}"
    )
    st.sidebar.text(summary)
    
    return config
