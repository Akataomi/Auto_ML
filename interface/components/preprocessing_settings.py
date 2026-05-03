"""
Preprocessing Settings Component.

Handles data preprocessing configuration.
"""

import streamlit as st
from typing import Dict


def render_preprocessing_settings() -> Dict:
    """
    Render preprocessing settings in sidebar.
    
    Returns:
        Dict: Preprocessing configuration.
    """
    st.sidebar.subheader("🔧 Предобработка")
    
    config = {
        "fill_missing": st.sidebar.selectbox(
            "Заполнение пропусков",
            ["median", "mean", "most_frequent"]
        ),
        "encode_categorical": st.sidebar.selectbox(
            "Кодирование категориальных",
            ["onehot", "label"]
        ),
        "scale": st.sidebar.checkbox("Масштабирование", value=True),
        "handle_outliers": st.sidebar.checkbox("Обработка выбросов", value=False),
    }
    
    return config