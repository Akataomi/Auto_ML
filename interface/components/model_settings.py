"""
Model Settings Component.

Handles model selection and hyperparameter tuning configuration.
"""

import streamlit as st
from typing import List, Dict
from automl_core.models.registry import ModelRegistry


def render_model_settings(task_type: str) -> Dict:
    """
    Render model selection and tuning settings in sidebar.
    
    Args:
        task_type: Type of ML task (classification/regression).
        
    Returns:
        Dict: Model configuration.
    """
    st.sidebar.subheader("🤖 Модели")
    
    registry = ModelRegistry()
    available_models = registry.get_available_models(task_type)

    default_models = ["catboost", "lightgbm"]
    default_models = [m for m in default_models if m in available_models]
    
    selected_models = st.sidebar.multiselect(
        "Выберите модели",
        available_models,
        default=default_models,
        key="model_selector"
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
    
    return {
        "selected_models": selected_models,
        "tune_hyperparams": tune_hyperparams,
        "n_trials": n_trials,
    }