"""
Model Settings Component.

Handles model selection and hyperparameter tuning configuration.
"""

import streamlit as st
from typing import List, Dict
from automl_core.models.registry import ModelRegistry
from interface.components.deep_mlp_settings import render_deep_mlp_settings


def render_model_settings(task_type: str) -> Dict:
    """
    Render model selection and tuning settings in sidebar.
    
    Args:
        task_type: Type of ML task (classification/regression/clustering/anomaly_detection).
        
    Returns:
        Dict: Model configuration.
    """
    st.sidebar.subheader("🤖 Модели")
    
    registry = ModelRegistry()
    available_models = registry.get_available_models(task_type)

    # Default models based on task type
    if task_type == "clustering":
        default_models = ["kmeans"]
    elif task_type == "anomaly_detection":
        default_models = ["isolation_forest"]
    elif task_type in ["classification", "regression"]:
        default_models = ["catboost", "lightgbm", "deep_mlp_classifier" if task_type == "classification" else "deep_mlp_regressor"]
    else:
        default_models = ["catboost", "lightgbm"]
    
    default_models = [m for m in default_models if m in available_models]
    
    # Уникальный key для каждого task_type чтобы сбрасывать выбор при смене задачи
    model_selector_key = f"model_selector_{task_type}"
    
    # Проверяем, сменился ли task_type и сбрасываем если нужно
    if "last_task_type" not in st.session_state:
        st.session_state.last_task_type = task_type
    elif st.session_state.last_task_type != task_type:
        st.session_state.last_task_type = task_type
        # Сброс выбора моделей при смене типа задачи
        if model_selector_key in st.session_state:
            del st.session_state[model_selector_key]
    
    selected_models = st.sidebar.multiselect(
        "Выберите модели",
        available_models,
        default=default_models,
        key=model_selector_key
    )
    
    st.sidebar.subheader("⚡ Оптимизация")
    
    tune_hyperparams = st.sidebar.checkbox(
        "Оптимизация гиперпараметров (Optuna)", 
        value=False, 
        key="tune_checkbox"
    )
    
    n_trials = 30
    if tune_hyperparams:
        n_trials = st.sidebar.slider(
            "Количество испытаний", 
            10, 100, 30, 
            key="trials_slider"
        )

    use_cv = st.sidebar.checkbox(
        "Cross-validation (5 folds)", 
        value=True, 
        key="cv_checkbox"
    )

    mlp_config = None
    if task_type in ["classification", "regression"]:
        mlp_model_name = "deep_mlp_classifier" if task_type == "classification" else "deep_mlp_regressor"
        if mlp_model_name in selected_models:
            mlp_config = render_deep_mlp_settings(task_type)
    
    return {
        "selected_models": selected_models,
        "tune_hyperparams": tune_hyperparams,
        "n_trials": n_trials,
        "mlp_config": mlp_config,
        "use_cv": use_cv,
    }
