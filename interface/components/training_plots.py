"""
Training Plots Component
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#1E1E1E'
plt.rcParams['axes.facecolor'] = '#2D2D2D'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#FFFFFF'
plt.rcParams['ytick.color'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['font.size'] = 10


def render_training_curves(history_df: pd.DataFrame, model_name: str, task_type: str) -> None:
    """
    Render training curves
    
    Shows:
    - Training Loss vs Validation Loss
    - Training Metric vs Validation Metric
    """
    if history_df is None or history_df.empty:
        st.warning(f"⚠️ **{model_name}**: Нет данных для отображения графиков")
        return
    
    st.markdown(f" 📈 {model_name} - Кривые обучения")
    
    metric_name = "Accuracy" if task_type == "classification" else "R² Score"
    
    has_loss = "val_loss" in history_df.columns or "train_loss" in history_df.columns
    has_metric = "val_metric" in history_df.columns or "train_metric" in history_df.columns
    
    if not has_loss and not has_metric:
        st.warning("Нет данных для отображения")
        return
    
    # Create figure with subplots (2 columns if we have both loss and metric)
    if has_loss and has_metric:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    
    fig.suptitle(f'{model_name} - Training Progress', fontsize=14, fontweight='bold', color='white')
    
    epochs = history_df.get('epoch', range(len(history_df)))
    plot_idx = 0
    
    # Plot 1: Loss curves
    if has_loss:
        ax = axes[plot_idx]
        
        if "train_loss" in history_df.columns and not history_df["train_loss"].isna().all():
            ax.plot(epochs, history_df['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3, alpha=0.7)
        
        if "val_loss" in history_df.columns and not history_df["val_loss"].isna().all():
            ax.plot(epochs, history_df['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
        
        ax.set_xlabel('Iteration / Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot 2: Metric curves
    if has_metric:
        ax = axes[plot_idx]
        
        if "train_metric" in history_df.columns and not history_df["train_metric"].isna().all():
            ax.plot(epochs, history_df['train_metric'], 'g-', linewidth=2, label=f'Train {metric_name}', marker='o', markersize=3, alpha=0.7)
        
        if "val_metric" in history_df.columns and not history_df["val_metric"].isna().all():
            ax.plot(epochs, history_df['val_metric'], 'm-', linewidth=2, label=f'Val {metric_name}', marker='s', markersize=3)
        
        ax.set_xlabel('Iteration / Epoch', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Curves', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # overfitting
        if len(history_df) > 5 and "val_metric" in history_df.columns and "train_metric" in history_df.columns:
            last_train = history_df["train_metric"].iloc[-1]
            last_val = history_df["val_metric"].iloc[-1]
            gap = last_train - last_val
            
            if gap > 0.1:
                ax.text(0.02, 0.98, f'⚠️ Overfitting!\nGap: {gap:.3f}', 
                       transform=ax.transAxes, fontsize=9, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)


def render_optimization_plot(optimization_df: pd.DataFrame, task_type: str) -> None:
    """Render Optuna optimization history using matplotlib"""
    if optimization_df is None or optimization_df.empty:
        return
    
    st.markdown("---")
    st.subheader("⚡ Оптимизация гиперпараметров (Optuna)")

    value_column = 'value' if 'value' in optimization_df.columns else 'cv_mean'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optuna Hyperparameter Optimization', fontsize=14, fontweight='bold', color='white')
    
    # Plot 1: Score progression
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
    
    # Plot 2: Score distribution
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


def render_model_comparison(results: List[Dict], task_type: str) -> None:
    """Render model comparison using matplotlib/seaborn"""
    if not results:
        return
    
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return
    
    st.subheader("🏆 Сравнение моделей")
    
    metric_name = "accuracy" if task_type == "classification" else "r2"
    metric_display = "Accuracy" if task_type == "classification" else "R² Score"
    
    comparison_data = []
    for r in valid_results:
        metrics = r.get("metrics", {})
        comparison_data.append({
            "Model": r.get("model"),
            metric_display: metrics.get(metric_name, 0),
            "CV Mean": metrics.get("cv_mean", 0),
            "Tuned": "Yes" if r.get("tuned", False) else "No",
        })
    
    comparison_df = pd.DataFrame(comparison_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison', fontsize=14, fontweight='bold', color='white')
    
    # Plot 1: Primary metric
    ax = axes[0]
    models = comparison_df['Model']
    scores = comparison_df[metric_display]
    colors = ['#FF6B6B' if tuned == 'Yes' else '#4ECDC4' for tuned in comparison_df['Tuned']]
    
    bars = ax.barh(models, scores, color=colors, alpha=0.8)
    ax.set_xlabel(metric_display, fontsize=11)
    ax.set_title(f'{metric_display} by Model', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', fontsize=9, color='white')
    
    # Plot 2: CV Mean
    ax = axes[1]
    cv_scores = comparison_df['CV Mean']
    bars = ax.barh(models, cv_scores, color=colors, alpha=0.8)
    ax.set_xlabel('CV Mean Score', fontsize=11)
    ax.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', fontsize=9, color='white')
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)
    
    st.markdown("**🎨 Легенда:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔴 **Красный** = С оптимизацией гиперпараметров")
    with col2:
        st.markdown("🔵 **Синий** = Без оптимизации")