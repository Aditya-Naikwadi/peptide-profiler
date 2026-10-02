"""Shared Evaluation Infrastructure for Immunological Classifiers.

Implements Task 6 of Shared Infrastructure:
Provides unified evaluation across all screening models:
- Discrimination: ROC-AUC, PR-AUC, Balanced Accuracy, MCC.
- Calibration: Brier score, Log Loss, Adaptive-bin ECE (15 bins), and reliability coordinates.
- Operating point: Sensitivity, Specificity, PPV, NPV at explicit threshold.
- Stratification: Peptide length bins [5-9], [10-14], [15-24], [25-49], [50+].
- Statistical Rigor: 95% cluster-bootstrap confidence intervals (1,000 resamples).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models.metrics import (
    assign_length_bin,
    cluster_bootstrap_confidence_intervals,
    compute_adaptive_ece,
    compute_reliability_diagram_points,
    compute_standard_classifier_metrics,
    evaluate_length_stratified,
)

logger = logging.getLogger(__name__)


def evaluate_screening_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cluster_ids: List[str],
    lengths: List[int],
    threshold: float = 0.5,
    positive_class_label: str = "positive",
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Comprehensive evaluation of a classifier following EVALUATION_PROTOCOL.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        cluster_ids: Cluster ID string for each sample (for cluster bootstrap).
        lengths: Sequence length for each sample (for length-bin stratification).
        threshold: Operating decision threshold.
        positive_class_label: Name of positive class (hazard/target).
        n_bootstraps: Number of bootstrap iterations.
        random_state: Seed for bootstrap reproducibility.

    Returns:
        Dictionary containing point estimates, 95% CIs, length breakdown, and calibration curve.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # 1. Point metrics
    metrics = compute_standard_classifier_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        positive_class_label=positive_class_label,
    )

    # 2. Cluster-level bootstrap 95% confidence intervals
    cis = cluster_bootstrap_confidence_intervals(
        y_true=y_true,
        y_prob=y_prob,
        clusters=cluster_ids,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )

    # 3. Stratified by length bins: [5-9], [10-14], [15-24], [25-49], [50+]
    length_breakdown = evaluate_length_stratified(
        y_true=y_true,
        y_prob=y_prob,
        lengths=lengths,
        threshold=threshold,
        positive_class_label=positive_class_label,
    )

    # 4. Adaptive-bin reliability diagram points (15 bins)
    reliability_points = compute_reliability_diagram_points(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=15,
    )

    return {
        "metrics": metrics,
        "confidence_intervals_95": cis,
        "length_stratified": length_breakdown,
        "reliability_diagram": reliability_points,
    }
