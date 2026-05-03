"""
Results Display Component.

Handles visualization of pipeline results.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
from interface.utils import format_metric


def render_results(report: Dict, pipeline) -> None:
    """
    Render pipeline results.
    
    Args:
        report: Pipeline execution report.
        pipeline: Trained AutoMLPipeline instance.
    """
    if not report:
        return
    
    st.success("✅ Обучение завершено!")

    st.subheader("🏆 Результаты моделей")
    
    if report.get("results"):
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
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            if "CV Mean" in results_df.columns and not results_df.empty:
                results_df["CV Mean Num"] = pd.to_numeric(results_df["CV Mean"], errors="coerce")
                best_idx = results_df["CV Mean Num"].idxmax()
                best_model = results_df.loc[best_idx, "Модель"]
                best_score = results_df.loc[best_idx, "CV Mean"]
                st.success(f"🏆 Лучшая модель: **{best_model}** (CV: {best_score})")

    st.subheader("📈 Важность признаков")
    if pipeline and pipeline.best_model and pipeline.best_model.feature_importance is not None:
        importance = pipeline.get_feature_importance()
        if importance:
            imp_df = pd.DataFrame({
                "Признак": list(importance.keys()),
                "Важность": list(importance.values())
            }).sort_values("Важность", ascending=False)
            st.bar_chart(imp_df.set_index("Признак").head(15))

    st.subheader("💾 Сохранение модели")
    if st.button("Сохранить лучшую модель"):
        target_col = st.session_state.get("target_col", "target")
        save_path = f"best_model_{target_col}.joblib"
        pipeline.save_best_model(save_path)
        st.success(f"Модель сохранена: {save_path}")