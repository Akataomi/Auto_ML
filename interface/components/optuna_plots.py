"""
Optuna Visualization Component
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
            plot_contour
        )
import warnings
warnings.filterwarnings('ignore')


def render_optimization_plot(optimization_df: pd.DataFrame, task_type: str) -> None:
    """
    Render basic Optuna optimization history using matplotlib.
    
    Args:
        optimization_df: DataFrame with trial history
        task_type: 'classification' or 'regression'
    """
    if optimization_df is None or optimization_df.empty:
        return
    
    st.markdown("---")
    st.subheader("⚡ Оптимизация гиперпараметров (Optuna)")

    value_column = 'value' if 'value' in optimization_df.columns else 'cv_mean'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optuna Hyperparameter Optimization', fontsize=14, fontweight='bold', color='white')

    ax = axes[0]
    trials = optimization_df['trial'].values
    scores = optimization_df[value_column].values
    
    ax.scatter(trials, scores, alpha=0.6, s=30, c='blue', label='Trials')

    if task_type == "classification":
        best_scores = np.maximum.accumulate(scores)
    else:
        best_scores = np.minimum.accumulate(scores)
    
    ax.plot(trials, best_scores, 'r-', linewidth=2, label='Best Score', marker='s', markersize=4)
    ax.set_xlabel('Trial Number', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Optimization Progress', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(scores, bins=20, alpha=0.7, color='green', edgecolor='white')
    best_val = scores.max() if task_type == "classification" else scores.min()
    ax.axvline(best_val, color='red', linestyle='--', linewidth=2, label=f'Best: {best_val:.4f}')
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)
    
    with st.expander("📉 Детальная статистика оптимизации"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего испытаний", len(optimization_df))
        with col2:
            best = optimization_df[value_column].max() if task_type == "classification" else optimization_df[value_column].min()
            st.metric("Лучший score", f"{best:.4f}")
        with col3:
            st.metric("Средний score", f"{optimization_df[value_column].mean():.4f}")

        st.markdown("**🏆 Топ-5 испытаний:**")
        top_trials = optimization_df.nlargest(5, value_column) if task_type == "classification" else optimization_df.nsmallest(5, value_column)
        st.dataframe(top_trials[["trial", value_column, "state"]].reset_index(drop=True), width="stretch")


def render_optuna_visualizations(study) -> None:
    """
    Render Optuna visualizations
    
    Args:
        study: Optuna study object
    """
    if study is None:
        st.warning(" Optuna study not available")
        return
    
    try: 
        # Optimization History
        st.markdown(" 📈 Optimization History")
        fig = plot_optimization_history(study)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter Importances
        st.markdown(" 🔍 Parameter Importances")
        fig = plot_param_importances(study)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Parallel Coordinate
        st.markdown(" 📊 Parallel Coordinate")
        fig = plot_parallel_coordinate(study)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Slice Plot
        st.markdown(" 📉 Slice Plot")
        fig = plot_slice(study)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Contour Plot
        st.markdown(" 🔀 Contour Plot")
        fig = plot_contour(study)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Ошибка при построении визуализаций Optuna: {e}")
        import sys
        print(f" [UI] Optuna visualization error: {e}", file=sys.stderr)