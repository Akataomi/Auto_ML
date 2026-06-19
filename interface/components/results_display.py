"""
Results Display Component.

Renders training results, plots, and model comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import sys
from interface.utils import format_metric
from interface.components.training_plots import (
    render_training_curves,
    render_model_comparison,
    render_clustering_visualization,
    render_anomaly_visualization,
)
from interface.components.optuna_plots import (
    render_optimization_plot,
    render_optuna_visualizations
)
from interface.components.classification_curves import (
    render_classification_curves
)

ITERATIVE_MODELS = [
    "xgboost", "lightgbm", "catboost", 
    "sgd_regressor", "sgd_classifier",
    "deep_mlp_classifier", "deep_mlp_regressor"
]


def render_results(report: Dict, pipeline) -> None:
    """Render pipeline results"""
    if not report:
        st.error("❌ Отчет пустой!")
        return
    
    st.success("✅ Обучение завершено!")
    
    task_type = pipeline.config.task_type
    is_unsupervised = task_type in ["clustering", "anomaly_detection"]
    
    # Debug
    print(f"\n [UI] Report keys: {report.keys()}", file=sys.stderr)
    print(f" [UI] Task type: {task_type}", file=sys.stderr)
    print(f" [UI] Models trained: {len(pipeline.models_trained) if pipeline else 0}", file=sys.stderr)
    
    # Model Results Table - different columns for different task types
    st.subheader("🏆 Результаты моделей")
    
    if report.get("results"):
        print(f" [UI] Results count: {len(report['results'])}", file=sys.stderr)
        results_data = []
        for r in report["results"]:
            if "error" not in r:
                metrics = r.get("metrics", {})
                if is_unsupervised:
                    if task_type == "clustering":
                        results_data.append({
                            "Модель": r.get("model"),
                            "Silhouette": format_metric(metrics.get("silhouette")),
                            "N кластеров": metrics.get("n_clusters", "-"),
                            "Tuned": "Yes" if r.get("tuned", False) else "No",
                        })
                    else:  # anomaly_detection
                        results_data.append({
                            "Модель": r.get("model"),
                            "F1": format_metric(metrics.get("f1")),
                            "% аномалий": f"{metrics.get('anomaly_rate', 0) * 100:.1f}%",
                            "Tuned": "Yes" if r.get("tuned", False) else "No",
                        })
                else:
                    # Supervised - old logic
                    results_data.append({
                        "Модель": r.get("model"),
                        "Accuracy": format_metric(metrics.get("accuracy")),
                        "F1": format_metric(metrics.get("f1")),
                        "AUC-ROC": format_metric(metrics.get("auc_roc")),
                        "RMSE": format_metric(metrics.get("rmse")),
                        "R²": format_metric(metrics.get("r2")),
                        "CV Mean": format_metric(metrics.get("cv_mean")),
                    })
            else:
                print(f"⚠️ [UI] Model {r.get('model')} had error: {r.get('error')}", file=sys.stderr)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, width="stretch")
            
            # Best model selection
            if is_unsupervised:
                if task_type == "clustering" and "Silhouette" in results_df.columns:
                    results_df["Silhouette Num"] = pd.to_numeric(results_df["Silhouette"], errors="coerce")
                    best_idx = results_df["Silhouette Num"].idxmax()
                    if pd.notna(best_idx):
                        best_model = results_df.loc[best_idx, "Модель"]
                        best_score = results_df.loc[best_idx, "Silhouette"]
                        st.success(f"🏆 Лучшая модель: **{best_model}** (Silhouette: {best_score})")
                elif task_type == "anomaly_detection" and "F1" in results_df.columns:
                    results_df["F1 Num"] = pd.to_numeric(results_df["F1"], errors="coerce")
                    best_idx = results_df["F1 Num"].idxmax()
                    if pd.notna(best_idx):
                        best_model = results_df.loc[best_idx, "Модель"]
                        best_score = results_df.loc[best_idx, "F1"]
                        st.success(f"🏆 Лучшая модель: **{best_model}** (F1: {best_score})")
            elif "CV Mean" in results_df.columns and not results_df.empty:
                results_df["CV Mean Num"] = pd.to_numeric(results_df["CV Mean"], errors="coerce")
                best_idx = results_df["CV Mean Num"].idxmax()
                if pd.notna(best_idx):
                    best_model = results_df.loc[best_idx, "Модель"]
                    best_score = results_df.loc[best_idx, "CV Mean"]
                    st.success(f"🏆 Лучшая модель: **{best_model}** (CV: {best_score})")
        else:
            st.warning("⚠️ Нет успешных результатов обучения")
    
    # Unsupervised Visualizations
    if is_unsupervised and pipeline and pipeline.models_trained:
        st.markdown("---")
        
        # Get data from session state
        if st.session_state.get("df_preview") is not None:
            df = st.session_state.df_preview
            target_col = st.session_state.get("target_col")
            
            # Prepare features (exclude target if present)
            if target_col and target_col in df.columns:
                X_viz = df.drop(columns=[target_col])
            else:
                X_viz = df.copy()
            
            # Encode categorical for visualization
            from automl_core.preprocessing import DataEncoder
            encoder = DataEncoder(categorical_strategy="onehot", scale=True)
            col_types = report.get("column_types", {})
            numeric_cols = col_types.get("numeric", [])
            categorical_cols = col_types.get("categorical", [])
            
            try:
                X_encoded, _, _ = encoder.fit_transform(X_viz, target_col, categorical_cols, numeric_cols)
                X_for_viz = X_encoded.values
            except Exception as e:
                X_for_viz = X_viz.select_dtypes(include=[np.number]).values
            
            # Get best model predictions/labels
            best_trainer = pipeline.best_model
            
            if task_type == "clustering" and hasattr(best_trainer, 'labels'):
                render_clustering_visualization(X_for_viz, best_trainer.labels)
            elif task_type == "anomaly_detection" and hasattr(best_trainer, 'predictions'):
                render_anomaly_visualization(X_for_viz, best_trainer.predictions)
    
    # Training Curves
    st.markdown("---")
    st.subheader("📊 Графики обучения (Loss Curves)")
    
    if pipeline and pipeline.models_trained:
        print(f" [UI] Checking {len(pipeline.models_trained)} models for plots...", file=sys.stderr)
        has_iterative_plots = False
        
        for model_info in pipeline.models_trained:
            trainer = model_info.get("trainer")
            model_name = model_info.get("name")
            
            print(f" [UI] Checking model: {model_name}", file=sys.stderr)

            if trainer and model_name in ITERATIVE_MODELS:
                print(f"🔍 [UI] Model {model_name} is iterative", file=sys.stderr)
                history_df = trainer.get_training_history_df()
                
                if history_df is not None and not history_df.empty:
                    print(f"🔍 [UI] History shape: {history_df.shape}", file=sys.stderr)
                    render_training_curves(history_df, model_name, pipeline.config.task_type)
                    has_iterative_plots = True
                    st.markdown("")
                else:
                    print(f"⚠️ [UI] Model {model_name} has empty history", file=sys.stderr)
            else:
                print(f"ℹ️ [UI] Model {model_name} is not iterative (category: {model_name})", file=sys.stderr)
        
        if not has_iterative_plots:
            st.info("ℹ️ Графики обучения доступны только для итеративных моделей. Линейные модели обучаются аналитически за 1 шаг и не имеют кривых обучения.")
            st.markdown("**Доступные итеративные модели:** " + ", ".join(ITERATIVE_MODELS))
    else:
        st.warning("⚠️ Pipeline или models_trained пусты")
    
    # Optuna Optimization
    if pipeline and pipeline.models_trained:
        has_optuna = any(m.get("trainer") and m["trainer"].tune for m in pipeline.models_trained)
        if has_optuna:
            st.markdown("---")
            for model_info in pipeline.models_trained:
                trainer = model_info.get("trainer")
                if trainer and trainer.tune:
                    opt_df = trainer.get_optimization_history_df()
                    if opt_df is not None and not opt_df.empty:
                        render_optimization_plot(opt_df, pipeline.config.task_type)
    
    # Optuna Visualizations
    if pipeline and pipeline.models_trained:
        for model_info in pipeline.models_trained:
            trainer = model_info.get("trainer")
            if trainer and trainer.tune and hasattr(trainer, 'tuner') and trainer.tuner is not None:
                study = trainer.tuner.get_study()
                if study:
                    st.markdown("---")
                    st.subheader(f"📊 Подробная визуализация Optuna ({model_info['name']})")
                    
                    with st.expander("📊 Развернуть визуализации Optuna", expanded=True):
                        render_optuna_visualizations(study)
    
    # Model Comparison
    if report.get("results"):
        render_model_comparison(report["results"], pipeline.config.task_type)

    # Classification Curves (ROC, PR, Confusion Matrix)
    if pipeline.config.task_type == "classification" and pipeline and pipeline.models_trained:
        st.markdown("---")
        st.subheader("📊 Классификационные кривые")
        
        for model_info in pipeline.models_trained:
            trainer = model_info.get("trainer")
            model_name = model_info.get("name")
            
            if trainer and trainer.task_type == "classification":
                print(f" [UI] Getting classification curves for {model_name}", file=sys.stderr)
                curves_data = trainer.get_classification_curves_data()
                
                if curves_data and (curves_data.get('roc') or curves_data.get('pr') or curves_data.get('confusion_matrix')):
                    with st.expander(f"📈 {model_name}: ROC, PR, Confusion Matrix", expanded=False):
                        render_classification_curves(curves_data, model_name)
    
    # Feature Importance
    st.subheader("📈 Важность признаков")
    if pipeline and pipeline.best_model and pipeline.best_model.feature_importance is not None:
        importance = pipeline.get_feature_importance()
        if importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df = pd.DataFrame({
                "Признак": list(importance.keys()),
                "Важность": list(importance.values())
            }).sort_values("Важность", ascending=False).head(15)
            
            ax.barh(imp_df['Признак'], imp_df['Важность'], color='#FF6B6B', alpha=0.8)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)
        else:
            st.info("Важность признаков недоступна для этой модели")
    
    # Linear Model Coefficients
    st.subheader("📊 Коэффициенты линейной модели")
    if pipeline and pipeline.best_model and pipeline.best_model.coefficients is not None:
        feature_names = report.get("features", [])
        if feature_names:
            coef_df = pipeline.best_model.get_coefficients_df(feature_names)
            if not coef_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_coef = coef_df.head(15)
                
                colors = ['green' if c > 0 else 'red' for c in top_coef['coefficient']]
                ax.barh(top_coef['feature'], top_coef['coefficient'], color=colors, alpha=0.8)
                ax.set_xlabel('Coefficient Value', fontsize=11)
                ax.set_title('Model Coefficients', fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                ax.axvline(x=0, color='black', linewidth=1)
                
                plt.tight_layout()
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
        else:
            st.info("Коэффициенты недоступны")
    
    # Save Model
    st.subheader("💾 Сохранение модели")
    
    # Инициализация состояния для сохранения модели
    if "model_saved" not in st.session_state:
        st.session_state.model_saved = False
        st.session_state.model_save_path = None
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("💾 Сохранить лучшую модель", key="save_model_btn"):
            try:
                target_col = st.session_state.get("target_col", "target")
                task_type = pipeline.config.task_type
                save_path = f"models/best_model_{task_type}_{target_col}.joblib"
                
                # Создаем директорию если не существует
                from pathlib import Path
                Path("models").mkdir(exist_ok=True)
                
                pipeline.save_best_model(save_path)
                st.session_state.model_saved = True
                st.session_state.model_save_path = save_path
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {str(e)}")
    
    with col2:
        if st.session_state.model_saved and st.session_state.model_save_path:
            save_path = st.session_state.model_save_path
            
            # Проверяем существует ли файл
            import os
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
                
                st.success(f"✅ Модель сохранена: `{save_path}` ({file_size:.2f} MB)")
                
                # Кнопка для скачивания
                with open(save_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать модель",
                        data=f,
                        file_name=f"best_model_{pipeline.config.task_type}.joblib",
                        mime="application/octet-stream",
                        key="download_model_btn"
                    )
                
                # Кнопка чтобы сбросить и сохранить заново
                if st.button("🔄 Сохранить заново", key="reset_save_btn"):
                    st.session_state.model_saved = False
                    st.session_state.model_save_path = None
                    st.rerun()
            else:
                st.error(f"❌ Файл не найден: {save_path}")
                st.session_state.model_saved = False
