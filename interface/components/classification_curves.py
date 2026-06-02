"""
Classification Curves Display Component.

Renders ROC curves, Precision-Recall curves, and Confusion Matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


def render_roc_curve(roc_data: Dict, model_name: str) -> None:
    """Render ROC curve"""
    if not roc_data or not roc_data.get('fpr'):
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(roc_data['fpr']) == 1:
        fpr = roc_data['fpr'][0]
        tpr = roc_data['tpr'][0]
        auc_score = roc_data['auc'][0]
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', 
                color='blue', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    else:
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i, (fpr, tpr, auc_score) in enumerate(zip(
            roc_data['fpr'], roc_data['tpr'], roc_data['auc']
        )):
            color = colors[i % len(colors)]
            ax.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})', 
                    color=color, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve (OvR) - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)


def render_precision_recall_curve(pr_data: Dict, model_name: str) -> None:
    """Render Precision-Recall curve"""
    if not pr_data or not pr_data.get('precision'):
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if len(pr_data['precision']) == 1:
        precision = pr_data['precision'][0]
        recall = pr_data['recall'][0]
        ap_score = pr_data['average_precision'][0]
        
        ax.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})', 
                color='green', linewidth=2)
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    else:
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i, (precision, recall, ap_score) in enumerate(zip(
            pr_data['precision'], pr_data['recall'], pr_data['average_precision']
        )):
            color = colors[i % len(colors)]
            ax.plot(recall, precision, label=f'Class {i} (AP = {ap_score:.3f})', 
                    color=color, linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision-Recall Curve (OvR) - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)


def render_confusion_matrix(cm_data: Dict, model_name: str) -> None:
    """Render Confusion Matrix"""
    if not cm_data or cm_data.get('matrix') is None:
        return
    
    cm = cm_data['matrix']
    classes = cm_data['classes']
    
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion Matrix - {model_name}',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)


def render_classification_curves(curves_data: Dict, model_name: str) -> None:
    """Render all classification curves (ROC, PR, Confusion Matrix)"""
    if not curves_data:
        return

    if curves_data.get('roc'):
        st.subheader(f"📈 ROC Curve - {model_name}")
        render_roc_curve(curves_data['roc'], model_name)
        st.markdown("")

    if curves_data.get('pr'):
        st.subheader(f"📈 Precision-Recall Curve - {model_name}")
        render_precision_recall_curve(curves_data['pr'], model_name)
        st.markdown("")

    if curves_data.get('confusion_matrix'):
        st.subheader(f"📊 Confusion Matrix - {model_name}")
        render_confusion_matrix(curves_data['confusion_matrix'], model_name)
        st.markdown("")
