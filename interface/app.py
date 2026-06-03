"""
AutoML Expert System - Main Interface.

Streamlit application for automated machine learning.
"""


import streamlit as st
from automl_core.pipeline.config import PipelineConfig, PreprocessingConfig, ModelConfig
from automl_core.pipeline.orchestrator import AutoMLPipeline
from automl_core.data.datasets import get_all_datasets, load_dataset

from interface.state import initialize_state, clear_file, set_report
from interface.components import (
    render_file_upload,
    render_target_selector,
    render_preprocessing_settings,
    render_model_settings,
    render_results,
    render_dataset_selector,
)
from interface.styles import get_custom_css


def main():
    """Main application entry point."""

    st.set_page_config(
        page_title="AutoML Expert System",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown(get_custom_css(), unsafe_allow_html=True)

    initialize_state()

    st.title(" AutoML Expert System")
    st.markdown("---")

    data_source = st.radio(
        "📊 Источник данных:",
        ["📁 Загрузить CSV файл", "📦 Использовать встроенный датасет"],
        horizontal=True
    )
    
    if data_source == "📦 Использовать встроенный датасет":
        dataset_choice = render_dataset_selector()
        if dataset_choice:
            try:
                df = load_dataset(dataset_choice)
                st.session_state.uploaded_file_path = None
                st.session_state.df_preview = df
                st.success(f"✅ Загружен датасет: {df.shape[0]} строк, {df.shape[1]} признаков")

                with st.expander("👁️ Предпросмотр данных"):
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Ошибка загрузки датасета: {str(e)}")
                return
        else:
            return
    else:
        if not render_file_upload():
            return
        
        df = st.session_state.df_preview
    
    st.markdown("---")

    task_type = render_target_selector(df)
    
    st.markdown("---")

    preprocess_config = render_preprocessing_settings()
    model_config = render_model_settings(task_type)

    st.markdown("---")
    if st.button("🚀 Запустить AutoML", type="primary", width="stretch"):
        if not model_config["selected_models"]:
            st.error("❌ Выберите хотя бы одну модель!")
            return
        
        with st.spinner("⏳ Обучение моделей..."):
            try:
                target_col = st.session_state.get("target_col")

                models_config = []
                for m in model_config["selected_models"]:
                    if m in ["deep_mlp_classifier", "deep_mlp_regressor"] and model_config.get("mlp_config"):
                        models_config.append(
                            ModelConfig(
                                name=m,
                                tune_hyperparams=model_config["tune_hyperparams"],
                                n_trials=model_config["n_trials"],
                                mlp_config=model_config["mlp_config"],
                                use_cv=model_config.get("use_cv", True)
                            )
                        )
                    else:
                        models_config.append(
                            ModelConfig(
                                name=m,
                                tune_hyperparams=model_config["tune_hyperparams"],
                                n_trials=model_config["n_trials"],
                                use_cv=model_config.get("use_cv", True)
                            )
                        )
                
                config = PipelineConfig(
                    target_column=target_col,
                    task_type=task_type,
                    preprocessing=PreprocessingConfig(
                        fill_missing=preprocess_config["fill_missing"],
                        encode_categorical=preprocess_config["encode_categorical"],
                        scale=preprocess_config["scale"],
                        handle_outliers=preprocess_config["handle_outliers"],
                    ),
                    models=models_config,
                )

                use_mlflow = model_config.get("use_mlflow", False)
                use_mlflow_docker = model_config.get("use_mlflow_docker", False)
                pipeline = AutoMLPipeline(config, use_mlflow=use_mlflow, use_mlflow_docker=use_mlflow_docker)
                report = pipeline.run(
                    filepath=st.session_state.uploaded_file_path,
                    df=st.session_state.get("df_preview")
                )

                set_report(report, pipeline)

                render_results(report, pipeline)
                
                if use_mlflow:
                    st.info("💡 Эксперимент сохранен в MLFlow!")
                    
                    if use_mlflow_docker:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔗 Открыть MLFlow UI"):
                                import webbrowser
                                webbrowser.open("http://localhost:5050")
                        with col2:
                            st.markdown("**MLFlow сервер:** http://localhost:5050")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Режим:** Local (файлы)")
                        with col2:
                            st.markdown("**Путь:** `./mlruns`")
                
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown(
        """
        **AutoML Expert System v1.0** | 
        Поддерживаемые задачи: Классификация, Регрессия, Кластеризация, Детекция аномалий  
        Модели: CatBoost, LightGBM, XGBoost, Random Forest, Linear, KMeans, DBSCAN, IsolationForest, LOF
        **Deep Learning:** PyTorch MLP с настраиваемой архитектурой (Custom слои, BatchNorm, Dropout)
        """
    )


if __name__ == "__main__":
    main()