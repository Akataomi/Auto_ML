"""
Target Selector Component.

Handles target column selection and task type detection.
"""

import streamlit as st
import pandas as pd
from interface.utils import detect_task_type, get_model_emoji, get_model_color
from interface.state import set_task_type


def render_target_selector(df: pd.DataFrame) -> str:
    """
    Render target column and task type selector.
    
    Args:
        df: DataFrame with loaded data.
        
    Returns:
        str: Selected task type (classification/regression/clustering/anomaly_detection)
    """
    st.subheader("🎯 Целевая переменная")

    # Unsupervised mode toggle
    unsupervised_mode = st.checkbox(
        "🔵 Unsupervised обучение (без целевой переменной)",
        value=False,
        key="unsupervised_toggle"
    )
    
    target_col = None
    
    if unsupervised_mode:
        # Unsupervised - no target needed
        st.info("ℹ️ Целевая переменная не требуется для кластеризации и детекции аномалий")
        
        # Select task type for unsupervised
        task_type = st.selectbox(
            "Тип задачи",
            ["clustering", "anomaly_detection"],
            index=0,
            key="task_type_selector",
            label_visibility="collapsed"
        )
        
        auto_type = "unsupervised"
        reason = "Выбрано пользователем"
        
        # Show info about selected task
        if task_type == "clustering":
            st.success("🟣 🔵 **clustering** - Группировка объектов по схожести")
        else:
            st.success("🔴 ⚠️ **anomaly_detection** - Поиск аномальных наблюдений")
    else:
        # Supervised mode - select target
        target_col = st.selectbox(
            "Выберите целевую переменную",
            df.columns.tolist(),
            key="target_selector"
        )

        auto_type, reason = detect_task_type(df, target_col)

        st.info(f"💡 Авто-определение: **{auto_type}** — {reason}")

        col1, col2 = st.columns([3, 1])
        with col1:
            task_type = st.selectbox(
                "Тип задачи",
                ["classification", "regression"],
                index=0 if auto_type == "classification" else 1,
                key="task_type_selector",
                label_visibility="collapsed"
            )

        emoji = get_model_emoji(task_type)
        color = get_model_color(task_type)
        note = "*(переопределено)*" if task_type != auto_type else "*(авто)*"
        st.success(f"{color} {emoji} **{task_type}** {note}")

        with st.expander("📊 Статистика целевой переменной", expanded=False):
            target_data = df[target_col]
            st.write(f"**Тип данных:** `{target_data.dtype}`")
            st.write(f"**Уникальных значений:** {target_data.nunique()}")
            st.write(f"**Пропуски:** {target_data.isna().sum()}")
            
            if target_data.nunique() <= 20:
                st.write("**Распределение:**")
                st.write(target_data.value_counts())
    
    set_task_type(task_type, auto_type, reason)
    
    # Store target_col in session state
    st.session_state.target_col = target_col
    
    return task_type
