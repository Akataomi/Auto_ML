"""
AutoML Expert System - Main Interface.

Streamlit application for automated machine learning.
"""


import streamlit as st
from automl_core.pipeline.config import PipelineConfig, PreprocessingConfig, ModelConfig
from automl_core.pipeline.orchestrator import AutoMLPipeline

from interface.state import initialize_state, clear_file, set_report
from interface.components import (
    render_file_upload,
    render_target_selector,
    render_preprocessing_settings,
    render_model_settings,
    render_results,
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
                target_col = st.session_state.get("target_selector", df.columns[0])
                if target_col is None or target_col not in df.columns:
                    target_col = df.columns[0]

                config = PipelineConfig(
                    target_column=target_col,
                    task_type=task_type,
                    preprocessing=PreprocessingConfig(
                        fill_missing=preprocess_config["fill_missing"],
                        encode_categorical=preprocess_config["encode_categorical"],
                        scale=preprocess_config["scale"],
                        handle_outliers=preprocess_config["handle_outliers"],
                    ),
                    models=[
                        ModelConfig(
                            name=m,
                            tune_hyperparams=model_config["tune_hyperparams"],
                            n_trials=model_config["n_trials"]
                        )
                        for m in model_config["selected_models"]
                    ],
                )

                pipeline = AutoMLPipeline(config)
                report = pipeline.run(st.session_state.uploaded_file_path)

                set_report(report, pipeline)

                render_results(report, pipeline)
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown(
        """
        **AutoML Expert System v1.0** | 
        Поддерживаемые модели: CatBoost, LightGBM, XGBoost, Random Forest, LinearModels
        """
    )


if __name__ == "__main__":
    main()