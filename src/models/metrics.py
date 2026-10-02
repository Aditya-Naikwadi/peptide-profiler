"""Standard Evaluation Metrics Module for Immunological Screening.

Computes:
1. Discrimination: ROC-AUC, PR-AUC, Balanced Accuracy, MCC.
2. Calibration: Brier score, Log Loss, Adaptive-bin Expected Calibration Error (ECE, 15 bins),
   and reliability diagram coordinates.
3. Decision utility: Sensitivity, Specificity, PPV, NPV at operating threshold.
4. Stratification: Peptide length bins [5-9], [10-14], [15-24], [25-49], [50+].
5. Cluster-level bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

LENGTH_BINS: List[Tuple[str, int, int]] = [
    ("5-9", 5, 9),
    ("10-14", 10, 14),
    ("15-24", 15, 24),
    ("25-49", 25, 49),
    ("50+", 50, 100000),
]


def assign_length_bin(length: int) -> str:
    """Assign sequence length to standardized length bin."""
    for name, low, high in LENGTH_BINS:
        if low <= length <= high:
            return name
    return "50+"


def compute_adaptive_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Adaptive-Bin Expected Calibration Error (equal sample count per bin).
    
    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Target number of adaptive quantile bins (default 15).
        
    Returns:
        Adaptive ECE value.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0

    # Sort probabilities
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    # Split into n_bins approximately equal-sized chunks
    bin_chunks = np.array_split(np.arange(n), min(n_bins, n))
    ece = 0.0

    for chunk in bin_chunks:
        if len(chunk) == 0:
            continue
        bin_prob = y_prob_sorted[chunk]
        bin_true = y_true_sorted[chunk]

        mean_prob = np.mean(bin_prob)
        mean_true = np.mean(bin_true)
        ece += (len(chunk) / n) * abs(mean_true - mean_prob)

    return float(round(ece, 4))


def compute_reliability_diagram_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> List[Dict[str, float]]:
    """Compute coordinates for reliability diagram using adaptive binning."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return []

    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    bin_chunks = np.array_split(np.arange(n), min(n_bins, n))

    points = []
    for chunk in bin_chunks:
        if len(chunk) == 0:
            continue
        bin_prob = y_prob_sorted[chunk]
        bin_true = y_true_sorted[chunk]

        points.append({
            "mean_predicted_prob": float(round(np.mean(bin_prob), 4)),
            "empirical_accuracy": float(round(np.mean(bin_true), 4)),
            "bin_min_prob": float(round(np.min(bin_prob), 4)),
            "bin_max_prob": float(round(np.max(bin_prob), 4)),
            "sample_count": int(len(chunk)),
        })

    return points


def compute_standard_classifier_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    positive_class_label: str = "positive",
) -> Dict[str, Any]:
    """Compute full suite of standard metrics for a classifier.
    
    Positive class is the event of interest (explicitly documented).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_positives = int(np.sum(y_true))
    n_negatives = int(n_samples - n_positives)

    # ROC-AUC
    try:
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    # PR-AUC
    try:
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    except Exception:
        auprc = float(np.mean(y_true)) if len(y_true) > 0 else 0.0

    # Confusion matrix
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        # Edge case single class
        if y_true[0] == 1:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            tn, fp = 0, 0
        else:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            tp, fn = 0, 0

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    bal_acc = float(0.5 * (sensitivity + specificity))

    # MCC
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        mcc = 0.0

    # Calibration metrics
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        ll = float(log_loss(y_true, y_prob))
    except Exception:
        ll = 0.0
    ece = compute_adaptive_ece(y_true, y_prob, n_bins=15)

    return {
        "positive_class": positive_class_label,
        "n_samples": n_samples,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "threshold": round(threshold, 3),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "mcc": round(mcc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "adaptive_ece_15bins": round(ece, 4),
    }


def cluster_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clusters: List[str],
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% cluster-level bootstrap confidence intervals for all standard metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    cluster_to_indices = defaultdict(list)
    for idx, c in enumerate(clusters):
        cluster_to_indices[c].append(idx)

    rng = np.random.RandomState(random_state)
    boot_metrics: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = []
        for sc in sampled_clusters:
            sampled_indices.extend(cluster_to_indices[sc])

        sub_y_true = y_true[sampled_indices]
        sub_y_prob = y_prob[sampled_indices]

        # Only evaluate if both classes are represented
        if len(np.unique(sub_y_true)) < 2:
            continue

        m = compute_standard_classifier_metrics(sub_y_true, sub_y_prob, threshold=threshold)
        for k in ["auroc", "auprc", "balanced_accuracy", "mcc", "sensitivity", "specificity", "ppv", "npv", "brier_score", "log_loss", "adaptive_ece_15bins"]:
            boot_metrics[k].append(m[k])

    ci_results = {}
    for metric_name, values in boot_metrics.items():
        if len(values) > 0:
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            point = float(np.mean(values))
            ci_results[metric_name] = {
                "point_estimate": round(point, 4),
                "ci_lower_95": round(low, 4),
                "ci_upper_95": round(high, 4),
            }
        else:
            ci_results[metric_name] = {
                "point_estimate": 0.0,
                "ci_lower_95": 0.0,
                "ci_upper_95": 0.0,
            }

    return ci_results


def evaluate_length_stratified(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lengths: List[int],
    threshold: float = 0.5,
    positive_class_label: str = "positive",
) -> Dict[str, Dict[str, Any]]:
    """Compute performance breakdown stratified by length bins: [5-9], [10-14], [15-24], [25-49], [50+]."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    length_bins = [assign_length_bin(l) for l in lengths]

    stratified = {}
    for bin_name, _, _ in LENGTH_BINS:
        mask = np.array([lb == bin_name for lb in length_bins])
        count = int(np.sum(mask))
        if count == 0:
            stratified[bin_name] = {
                "sample_count": 0,
                "positive_count": 0,
                "auroc": None,
                "balanced_accuracy": None,
                "sensitivity": None,
                "specificity": None,
                "ppv": None,
                "npv": None,
            }
            continue

        bin_y_true = y_true[mask]
        bin_y_prob = y_prob[mask]
        pos_count = int(np.sum(bin_y_true))

        m = compute_standard_classifier_metrics(bin_y_true, bin_y_prob, threshold=threshold, positive_class_label=positive_class_label)
        stratified[bin_name] = {
            "sample_count": count,
            "positive_count": pos_count,
            "negative_count": count - pos_count,
            "auroc": m["auroc"] if len(np.unique(bin_y_true)) > 1 else None,
            "balanced_accuracy": m["balanced_accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "ppv": m["ppv"],
            "npv": m["npv"],
            "brier_score": m["brier_score"],
        }

    return stratified
