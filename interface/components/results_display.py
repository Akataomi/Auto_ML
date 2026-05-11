"""
Results Display Component.

Renders training results, plots, and model comparisons.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import sys
from interface.utils import format_metric
from interface.components.training_plots import (
    render_training_curves,
    render_model_comparison
)
from interface.components.optuna_plots import (
    render_optimization_plot,
    render_optuna_visualizations
)

ITERATIVE_MODELS = [
    "xgboost", "lightgbm", "catboost", 
    "sgd_regressor", "sgd_classifier"
]


def render_results(report: Dict, pipeline) -> None:
    """Render pipeline results"""
    if not report:
        st.error("❌ Отчет пустой!")
        return
    
    st.success("✅ Обучение завершено!")
    
    # Debug
    print(f"\n [UI] Report keys: {report.keys()}", file=sys.stderr)
    print(f" [UI] Models trained: {len(pipeline.models_trained) if pipeline else 0}", file=sys.stderr)
    
    # Model Results Table
    st.subheader("🏆 Результаты моделей")
    
    if report.get("results"):
        print(f" [UI] Results count: {len(report['results'])}", file=sys.stderr)
        results_data = []
        for r in report["results"]:
            if "error" not in r:
                results_data.append({
                    "Модель": r.get("model"),
                    "Accuracy": format_metric(r.get("metrics", {}).get("accuracy")),
                    "F1": format_metric(r.get("metrics", {}).get("f1")),
                    "AUC-ROC": format_metric(r.get("metrics", {}).get("auc_roc")),
                    "RMSE": format_metric(r.get("metrics", {}).get("rmse")),
                    "R²": format_metric(r.get("metrics", {}).get("r2")),
                    "CV Mean": format_metric(r.get("metrics", {}).get("cv_mean")),
                })
            else:
                print(f"⚠️ [UI] Model {r.get('model')} had error: {r.get('error')}", file=sys.stderr)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, width="stretch")
            
            if "CV Mean" in results_df.columns and not results_df.empty:
                results_df["CV Mean Num"] = pd.to_numeric(results_df["CV Mean"], errors="coerce")
                best_idx = results_df["CV Mean Num"].idxmax()
                best_model = results_df.loc[best_idx, "Модель"]
                best_score = results_df.loc[best_idx, "CV Mean"]
                st.success(f"🏆 Лучшая модель: **{best_model}** (CV: {best_score})")
        else:
            st.warning("⚠️ Нет успешных результатов обучения")
    
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
    if st.button("Сохранить лучшую модель"):
        target_col = st.session_state.get("target_selector", "target")
        save_path = f"best_model_{target_col}.joblib"
        pipeline.save_best_model(save_path)
        st.success(f"Модель сохранена: {save_path}")