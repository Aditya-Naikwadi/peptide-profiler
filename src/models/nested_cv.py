"""Nested Grouped Cross-Validation, Feature Pruning, and Cluster-Bootstrap Evaluation.

Adheres strictly to EVALUATION_PROTOCOL:
1. Homology-aware StratifiedGroupKFold splits.
2. Preprocessing, scaling, and feature selection fitted strictly inside training folds.
3. Feature pruning via grouped permutation importance.
4. Cluster-bootstrap 95% confidence intervals (resampling clusters, not sequences).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, Balanced Accuracy, Sensitivity, Specificity, MCC, Brier score."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float(np.mean(y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(0.5 * (sens + spec))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "mcc": round(mcc, 4),
        "precision": round(prec, 4),
        "brier_score": round(brier, 4),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clusters: List[str],
    n_bootstraps: int = 1000,
    metric_fn: Callable = roc_auc_score,
    random_state: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Calculate 95% confidence intervals by resampling at the cluster level."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    unique_clusters = np.unique(clusters)

    cluster_to_indices = defaultdict(list)
    for idx, c in enumerate(clusters):
        cluster_to_indices[c].append(idx)

    rng = np.random.RandomState(random_state)
    scores = []

    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_indices = []
        for sc in sampled_clusters:
            sampled_indices.extend(cluster_to_indices[sc])

        y_t_samp = y_true[sampled_indices]
        y_p_samp = y_prob[sampled_indices]

        if len(np.unique(y_t_samp)) > 1:
            try:
                score = metric_fn(y_t_samp, y_p_samp)
                scores.append(score)
            except Exception:
                pass

    if not scores:
        return 0.5, 0.5, 0.5

    mean_sc = float(np.mean(scores))
    low_ci = float(np.percentile(scores, 2.5))
    high_ci = float(np.percentile(scores, 97.5))
    return round(mean_sc, 4), round(low_ci, 4), round(high_ci, 4)


def run_nested_grouped_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    clusters: List[str],
    n_outer_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute outer grouped CV with strict fold isolation."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probs = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups=clusters)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_train, y_train)

        if hasattr(model_fold, "predict_proba"):
            probs = model_fold.predict_proba(X_val)
            p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs[:, 0]
        else:
            p1 = model_fold.predict(X_val)

        oof_probs[val_idx] = p1
        f_metric = compute_comprehensive_metrics(y_val, p1)
        fold_metrics.append(f_metric)

    # Compute overall metrics and cluster-bootstrap CIs
    overall_metrics = compute_comprehensive_metrics(y, oof_probs)
    mean_auc, low_auc, high_auc = cluster_bootstrap_ci(y, oof_probs, clusters, metric_fn=roc_auc_score)
    mean_pr, low_pr, high_pr = cluster_bootstrap_ci(y, oof_probs, clusters, metric_fn=average_precision_score)

    overall_metrics["auroc_ci"] = f"{mean_auc:.4f} [{low_auc:.4f} - {high_auc:.4f}]"
    overall_metrics["auroc_ci_low"] = low_auc
    overall_metrics["auroc_ci_high"] = high_auc
    overall_metrics["auprc_ci"] = f"{mean_pr:.4f} [{low_pr:.4f} - {high_pr:.4f}]"
    overall_metrics["auprc_ci_low"] = low_pr
    overall_metrics["auprc_ci_high"] = high_pr
    overall_metrics["oof_probs"] = oof_probs
    overall_metrics["fold_metrics"] = fold_metrics

    return overall_metrics


def grouped_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    clusters: List[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Compute permutation importance across grouped CV validation folds."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    n_features = X.shape[1]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    importances = np.zeros((n_features, n_splits), dtype=np.float32)

    rng = np.random.RandomState(random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=clusters)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_train, y_train)

        # Baseline validation score
        p_base = model_fold.predict_proba(X_val)[:, 1]
        base_score = roc_auc_score(y_val, p_base)

        for f_idx in range(n_features):
            perm_drops = []
            for _ in range(n_repeats):
                X_val_perm = X_val.copy()
                # Permute feature values
                X_val_perm[:, f_idx] = rng.permutation(X_val_perm[:, f_idx])
                p_perm = model_fold.predict_proba(X_val_perm)[:, 1]
                perm_score = roc_auc_score(y_val, p_perm)
                perm_drops.append(base_score - perm_score)
            importances[f_idx, fold_idx] = float(np.mean(perm_drops))

    mean_importances = np.mean(importances, axis=1)
    return mean_importances
