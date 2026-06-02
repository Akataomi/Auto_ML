"""
Компонент выбора встроенного датасета.
"""

import streamlit as st
from automl_core.data.datasets import get_all_datasets


def render_dataset_selector() -> str | None:
    """
    Рендерит UI для выбора встроенного датасета.
    
    Returns:
        dataset_id: ID выбранного датасета или None
    """
    st.subheader("📦 Выберите встроенный датасет")
    
    all_datasets = get_all_datasets()

    flat_datasets = []
    for category, datasets in all_datasets.items():
        for ds in datasets:
            ds_with_category = ds.copy()
            ds_with_category["category"] = category
            flat_datasets.append(ds_with_category)

    options = {}
    for ds in flat_datasets:
        category_emoji = {
            "classification": "🏷️",
            "regression": "📈",
            "clustering": "🔵",
            "anomaly_detection": "⚠️"
        }.get(ds["category"], "📊")
        
        label = f"{category_emoji} {ds['name']} - {ds['description']}"
        options[label] = ds["id"]

    selected_label = st.selectbox(
        "Доступные датасеты:",
        list(options.keys()),
        help="Выберите датасет для тестирования AutoML"
    )
    
    if selected_label:
        ds = next((d for d in flat_datasets if d["id"] == options[selected_label]), None)
        if ds:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Образцов", ds["samples"])
            with col2:
                st.metric("📝 Признаков", ds["features"])
            with col3:
                if "classes" in ds:
                    st.metric("🎯 Классов", ds["classes"])
                elif "true_clusters" in ds:
                    st.metric("🔵 Кластеров", ds["true_clusters"])
                elif "anomaly_ratio" in ds:
                    st.metric("⚠️ Аномалий", f"{ds['anomaly_ratio']*100:.0f}%")

            st.info(f"💡 **Рекомендация:** Используйте этот датасет для тестирования задачи **{ds['category'].replace('_', ' ')}**")
    
    return options.get(selected_label)
