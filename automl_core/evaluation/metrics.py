"""
Metrics Calculator.

Classification, regression, clustering, and anomaly detection metrics
with support for advanced visualization data
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)
from typing import Dict, List


class MetricsCalculator:
    """Calculator for all ML metrics"""

    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Metrics for classification"""

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:
                try:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                    metrics["auc_pr"] = float(average_precision_score(y_true, y_pred_proba[:, 1]))
                except Exception:
                    metrics["auc_roc"] = None
                    metrics["auc_pr"] = None
            else:
                try:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
                    metrics["auc_pr"] = float(average_precision_score(y_true, y_pred_proba, average='weighted'))
                except Exception:
                    metrics["auc_roc"] = None
                    metrics["auc_pr"] = None

        return metrics

    @staticmethod
    def get_roc_curve_data(y_true, y_pred_proba) -> Dict[str, List]:
        """Get ROC curve data for visualization"""
        
        roc_data = {
            'fpr': [],
            'tpr': [],
            'thresholds': [],
            'auc': []
        }
        
        if len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
                roc_score = auc(fpr, tpr)
                roc_data['fpr'].append(fpr.tolist())
                roc_data['tpr'].append(tpr.tolist())
                roc_data['thresholds'].append(thresholds.tolist())
                roc_data['auc'].append(float(roc_score))
            except Exception:
                pass
        else:
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                n_classes = y_pred_proba.shape[1]
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_score = auc(fpr, tpr)
                    roc_data['fpr'].append(fpr.tolist())
                    roc_data['tpr'].append(tpr.tolist())
                    roc_data['thresholds'].append(_.tolist())
                    roc_data['auc'].append(float(roc_score))
            except Exception:
                pass
        
        return roc_data

    @staticmethod
    def get_precision_recall_curve_data(y_true, y_pred_proba) -> Dict[str, List]:
        """Get Precision-Recall curve data for visualization"""
        
        pr_data = {
            'precision': [],
            'recall': [],
            'thresholds': [],
            'average_precision': []
        }

        if len(np.unique(y_true)) == 2:
            try:
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
                ap_score = average_precision_score(y_true, y_pred_proba[:, 1])
                pr_data['precision'].append(precision.tolist())
                pr_data['recall'].append(recall.tolist())
                pr_data['thresholds'].append(thresholds.tolist())
                pr_data['average_precision'].append(float(ap_score))
            except Exception:
                pass
        else:
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                n_classes = y_pred_proba.shape[1]
                
                for i in range(n_classes):
                    precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    ap_score = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                    pr_data['precision'].append(precision.tolist())
                    pr_data['recall'].append(recall.tolist())
                    pr_data['thresholds'].append(thresholds.tolist())
                    pr_data['average_precision'].append(float(ap_score))
            except Exception:
                pass
        
        return pr_data

    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> Dict[str, np.ndarray]:
        """Get confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            return {
                'matrix': cm,
                'classes': np.unique(y_true)
            }
        except Exception:
            return {
                'matrix': np.array([[0]]),
                'classes': [0]
            }

    @staticmethod
    def regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Metrics for regression"""

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def clustering_metrics(X, labels) -> Dict[str, float]:
        """Metrics for clustering"""
        
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2 or n_clusters >= len(labels):
            return {
                "silhouette": 0.0,
                "n_clusters": 0,
                "n_outliers": int(np.sum(labels == -1)) if -1 in labels else 0,
            }
        
        try:
            return {
                "silhouette": float(silhouette_score(X, labels)),
                "n_clusters": int(len(np.unique(labels))),
                "n_outliers": int(np.sum(labels == -1)) if -1 in labels else 0,
            }
        except Exception:
            return {
                "silhouette": 0.0,
                "n_clusters": len(np.unique(labels)),
                "n_outliers": int(np.sum(labels == -1)) if -1 in labels else 0,
            }

    @staticmethod
    def anomaly_metrics(X, predictions, y_true=None) -> Dict[str, float]:
        """Metrics for anomaly detection"""
        
        y_pred_binary = (predictions == -1).astype(int)

        n_anomalies = int(np.sum(y_pred_binary == 1))
        n_total = len(predictions)
        anomaly_rate = float(n_anomalies / n_total) if n_total > 0 else 0.0
        
        metrics = {
            "n_anomalies": n_anomalies,
            "anomaly_rate": float(anomaly_rate),
        }

        if y_true is not None:
            try:
                if hasattr(y_true, 'mode'):
                    y_true_binary = (y_true != y_true.mode().iloc[0]).astype(int)
                else:
                    y_true_binary = (y_true == -1).astype(int)
                
                metrics["f1"] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics["precision"] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics["recall"] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            except Exception:
                pass
        
        return metrics
