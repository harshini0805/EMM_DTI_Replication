"""
Evaluation metrics for DTI prediction.

Implements standard ML evaluation metrics: AUC, AUPR, Accuracy, Precision, Recall.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """
    Comprehensive metrics calculator for DTI prediction.

    Computes AUC, AUPR, Accuracy, Precision, and Recall from predictions.
    """

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        y_pred_binary: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred_prob: Predicted probabilities [0, 1]
            y_pred_binary: Predicted binary labels (computed from prob if None)
            threshold: Threshold for converting probabilities to binary labels

        Returns:
            Dictionary with metric values

        Raises:
            ValueError: If inputs have inconsistent shapes
        """
        if len(y_true) != len(y_pred_prob):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred_prob={len(y_pred_prob)}"
            )

        # Convert probabilities to binary if not provided
        if y_pred_binary is None:
            y_pred_binary = (y_pred_prob >= threshold).astype(int)

        # Ensure binary labels
        y_true = y_true.astype(int)

        metrics = {}

        # ===== AUC-ROC =====
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred_prob)
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["auc"] = np.nan

        # ===== AUC-PR =====
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
            metrics["aupr"] = auc(recall, precision)
        except ValueError as e:
            logger.warning(f"Could not compute AUPR: {e}")
            metrics["aupr"] = np.nan

        # ===== Accuracy =====
        metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)

        # ===== Precision =====
        try:
            metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
        except ValueError as e:
            logger.warning(f"Could not compute Precision: {e}")
            metrics["precision"] = np.nan

        # ===== Recall =====
        try:
            metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
        except ValueError as e:
            logger.warning(f"Could not compute Recall: {e}")
            metrics["recall"] = np.nan

        # ===== F1-Score =====
        if not np.isnan(metrics.get("precision", np.nan)) and not np.isnan(
            metrics.get("recall", np.nan)
        ):
            p = metrics["precision"]
            r = metrics["recall"]
            if p + r > 0:
                metrics["f1"] = 2 * (p * r) / (p + r)
            else:
                metrics["f1"] = 0.0
        else:
            metrics["f1"] = np.nan

        # ===== Confusion Matrix Components =====
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where not all classes are present
            tn = cm[0, 0] if cm.shape[0] > 0 else 0
            fp = cm[0, 1] if cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 else 0
            tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        metrics["tp"] = int(tp)
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)

        # ===== Specificity =====
        if tn + fp > 0:
            metrics["specificity"] = tn / (tn + fp)
        else:
            metrics["specificity"] = np.nan

        # ===== Sensitivity (same as Recall) =====
        metrics["sensitivity"] = metrics["recall"]

        # ===== Matthews Correlation Coefficient =====
        try:
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred_binary)
        except ValueError as e:
            logger.warning(f"Could not compute MCC: {e}")
            metrics["mcc"] = np.nan

        return metrics

    @staticmethod
    def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
        """
        Format metrics for logging.

        Args:
            metrics: Dictionary of metric values
            prefix: Optional prefix for metric names

        Returns:
            Formatted string representation
        """
        lines = []

        # Primary metrics
        for key in ["auc", "aupr", "accuracy", "precision", "recall", "f1", "specificity", "mcc"]:
            if key in metrics:
                value = metrics[key]
                if np.isnan(value):
                    lines.append(f"{prefix}{key}: NaN")
                else:
                    lines.append(f"{prefix}{key}: {value:.4f}")

        return " | ".join(lines)

    @staticmethod
    def best_metric_for_checkpoint(metrics: Dict[str, float]) -> float:
        """
        Determine best metric for model checkpointing.

        Prefers AUC as primary metric, falls back to accuracy.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Best metric value
        """
        if "auc" in metrics and not np.isnan(metrics["auc"]):
            return metrics["auc"]
        elif "accuracy" in metrics and not np.isnan(metrics["accuracy"]):
            return metrics["accuracy"]
        else:
            return -np.inf


class MetricsTracker:
    """
    Track metrics over multiple iterations/epochs.

    Maintains running history for logging and analysis.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.history: Dict[str, list] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics history.

        Args:
            metrics: Metrics dictionary to add
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best(self, metric_name: str) -> Tuple[float, int]:
        """
        Get best value for a metric and its index.

        Args:
            metric_name: Name of metric to query

        Returns:
            Tuple of (best_value, best_index)
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return np.nan, -1

        values = self.history[metric_name]
        best_idx = int(np.nanargmax(values))
        return values[best_idx], best_idx

    def get_latest(self, metric_name: str) -> float:
        """
        Get latest value for a metric.

        Args:
            metric_name: Name of metric to query

        Returns:
            Latest metric value
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return np.nan

        return self.history[metric_name][-1]

    def reset(self) -> None:
        """Clear history."""
        self.history.clear()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MetricsTracker(metrics={list(self.history.keys())}, "
            f"length={len(self.history.get(list(self.history.keys())[0], []))})"
        )
