"""Calibration, Prior-Shift Correction, and Prevalence Analysis Module.

Implements:
1. Out-of-fold calibration (Platt scaling & Isotonic regression) on group-held-out folds.
2. Expected Calibration Error (ECE) and Brier Score quantification.
3. Bayesian prior-shift correction from training prevalence to deployment prevalence.
4. Positive Predictive Value (PPV) at deployment prevalences (1%, 2%, 3%).
5. Length-stratified evaluation across biophysical length bins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.models.calibrator import (
    BaseCalibrator,
    IsotonicCalibrator,
    ModelCalibratorWrapper,
    PlattCalibrator,
    TemperatureCalibrator,
    UnifiedCalibrator,
    apply_prior_shift,
    compute_brier_score,
    compute_ece,
    compute_reliability_diagram,
)
from src.models.thresholds import (
    ProfileDecisionEngine,
    compute_bayes_optimal_threshold,
    compute_prevalence_sensitivity_table,
    evaluate_threshold_performance,
    find_constrained_threshold,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        mask = (y_prob >= bin_low) & (y_prob < bin_high if i < n_bins - 1 else y_prob <= bin_high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, List[float]]:
    """Compute binned empirical accuracy vs mean predicted confidence."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = []
    accuracies = []
    counts = []

    for i in range(n_bins):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        mask = (y_prob >= bin_low) & (y_prob < bin_high if i < n_bins - 1 else y_prob <= bin_high)
        bin_count = int(np.sum(mask))
        counts.append(bin_count)

        if bin_count > 0:
            accuracies.append(float(round(np.mean(y_true[mask]), 4)))
            confidences.append(float(round(np.mean(y_prob[mask]), 4)))
        else:
            accuracies.append(float(round((bin_low + bin_high) / 2.0, 4)))
            confidences.append(float(round((bin_low + bin_high) / 2.0, 4)))

    return {
        "bin_midpoints": [float(round((bin_boundaries[i] + bin_boundaries[i + 1]) / 2.0, 3)) for i in range(n_bins)],
        "mean_confidences": confidences,
        "empirical_accuracies": accuracies,
        "sample_counts": counts,
    }


def bayesian_prior_shift(
    probabilities: np.ndarray,
    train_prior: float = 0.50,
    deploy_prior: float = 0.01,
    eps: float = 1e-6,
) -> np.ndarray:
    """Adjust probabilities from training prevalence to deployment prevalence via Bayes' theorem.

    Formula:
        odds_deploy = odds_train * (deploy_prior / (1 - deploy_prior)) * ((1 - train_prior) / train_prior)
        p_deploy = odds_deploy / (1 + odds_deploy)
    """
    clipped_p = np.clip(probabilities, eps, 1.0 - eps)
    odds_train = clipped_p / (1.0 - clipped_p)
    prior_ratio = (deploy_prior / (1.0 - deploy_prior)) * ((1.0 - train_prior) / train_prior)
    odds_deploy = odds_train * prior_ratio
    p_deploy = odds_deploy / (1.0 + odds_deploy)
    return np.clip(p_deploy, 0.0, 1.0)


def calculate_ppv_at_prevalence(
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> float:
    """Calculate Positive Predictive Value (PPV) / Precision at deployment prevalence."""
    tpr = sensitivity
    fpr = 1.0 - specificity
    denom = (tpr * prevalence) + (fpr * (1.0 - prevalence))
    if denom == 0:
        return 0.0
    return float(round((tpr * prevalence) / denom, 4))


def run_group_heldout_calibration(
    y_true: np.ndarray,
    y_raw_prob: np.ndarray,
    clusters: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit Platt scaling and Isotonic regression strictly on group-held-out folds."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y_platt = np.zeros(len(y_true))
    y_iso = np.zeros(len(y_true))

    raw_brier = float(round(brier_score_loss(y_true, y_raw_prob), 4))
    raw_ece = compute_ece(y_true, y_raw_prob)
    raw_rel = compute_reliability_curve(y_true, y_raw_prob)

    for train_idx, val_idx in sgkf.split(y_raw_prob.reshape(-1, 1), y_true, groups=clusters):
        p_tr, y_tr = y_raw_prob[train_idx], y_true[train_idx]
        p_val = y_raw_prob[val_idx]

        # 1. Platt Scaling (Logistic Regression on log-odds)
        eps = 1e-6
        log_odds_tr = np.log(np.clip(p_tr, eps, 1 - eps) / (1.0 - np.clip(p_tr, eps, 1 - eps))).reshape(-1, 1)
        log_odds_val = np.log(np.clip(p_val, eps, 1 - eps) / (1.0 - np.clip(p_val, eps, 1 - eps))).reshape(-1, 1)

        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(log_odds_tr, y_tr)
        y_platt[val_idx] = lr.predict_proba(log_odds_val)[:, 1]

        # 2. Isotonic Regression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_tr, y_tr)
        y_iso[val_idx] = iso.predict(p_val)

    # Compute post-calibration metrics
    platt_brier = float(round(brier_score_loss(y_true, y_platt), 4))
    platt_ece = compute_ece(y_true, y_platt)
    platt_rel = compute_reliability_curve(y_true, y_platt)

    iso_brier = float(round(brier_score_loss(y_true, y_iso), 4))
    iso_ece = compute_ece(y_true, y_iso)
    iso_rel = compute_reliability_curve(y_true, y_iso)

    return {
        "raw": {
            "brier_score": raw_brier,
            "ece": raw_ece,
            "reliability_curve": raw_rel,
        },
        "platt_calibrated": {
            "brier_score": platt_brier,
            "ece": platt_ece,
            "brier_reduction": float(round(raw_brier - platt_brier, 4)),
            "ece_reduction": float(round(raw_ece - platt_ece, 4)),
            "reliability_curve": platt_rel,
            "probabilities": y_platt.tolist(),
        },
        "isotonic_calibrated": {
            "brier_score": iso_brier,
            "ece": iso_ece,
            "brier_reduction": float(round(raw_brier - iso_brier, 4)),
            "ece_reduction": float(round(raw_ece - iso_ece, 4)),
            "reliability_curve": iso_rel,
            "probabilities": y_iso.tolist(),
        },
    }


def evaluate_length_stratified(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lengths: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate performance across 5 biophysical length bins."""
    bins = [
        ("< 15 aa", lambda l: l < 15),
        ("15-50 aa", lambda l: (l >= 15) and (l <= 50)),
        ("50-200 aa", lambda l: (l > 50) and (l <= 200)),
        ("200-500 aa", lambda l: (l > 200) and (l <= 500)),
        ("> 500 aa", lambda l: l > 500),
    ]

    stratified_results = {}

    for bin_name, condition in bins:
        mask = np.array([condition(l) for l in lengths])
        sub_y_true = y_true[mask]
        sub_y_prob = y_prob[mask]
        n_samples = int(np.sum(mask))
        n_pos = int(np.sum(sub_y_true))

        if n_samples == 0:
            continue

        base_prev = n_pos / n_samples if n_samples > 0 else 0.0

        if len(np.unique(sub_y_true)) > 1:
            try:
                prauc = float(round(average_precision_score(sub_y_true, sub_y_prob), 4))
            except Exception:
                prauc = float(round(base_prev, 4))
            try:
                auroc = float(round(roc_auc_score(sub_y_true, sub_y_prob), 4))
            except Exception:
                auroc = 0.5
        else:
            prauc = float(round(base_prev, 4))
            auroc = 0.5

        # Confusion matrix at threshold
        sub_pred = (sub_y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(sub_y_true, sub_pred, labels=[0, 1]).ravel()
        sens = float(round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4))
        spec = float(round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4))
        mcc = float(round(matthews_corrcoef(sub_y_true, sub_pred) if len(np.unique(sub_pred)) > 1 else 0.0, 4))

        # Precision@k & Enrichment Factor
        k_eval = min(top_k, n_samples)
        top_k_indices = np.argsort(sub_y_prob)[::-1][:k_eval]
        hits_in_k = int(np.sum(sub_y_true[top_k_indices]))
        prec_at_k = float(round(hits_in_k / k_eval, 4))
        ef_k = float(round(prec_at_k / base_prev, 2)) if base_prev > 0 else 1.0

        # Recall at fixed Precision (0.50 and 0.80)
        sorted_indices = np.argsort(sub_y_prob)[::-1]
        cum_pos = np.cumsum(sub_y_true[sorted_indices])
        cum_total = np.arange(1, n_samples + 1)
        prec_curve = cum_pos / cum_total

        # Recall @ Precision >= 0.80
        mask_80 = prec_curve >= 0.80
        recall_at_p80 = float(round(cum_pos[mask_80][-1] / n_pos, 4)) if (np.any(mask_80) and n_pos > 0) else 0.0

        # Recall @ Precision >= 0.50
        mask_50 = prec_curve >= 0.50
        recall_at_p50 = float(round(cum_pos[mask_50][-1] / n_pos, 4)) if (np.any(mask_50) and n_pos > 0) else 0.0

        stratified_results[bin_name] = {
            "n_samples": n_samples,
            "n_positives": n_pos,
            "local_prevalence": float(round(base_prev, 4)),
            "auroc": auroc,
            "pr_auc": prauc,
            "sensitivity": sens,
            "specificity": spec,
            "mcc": mcc,
            f"precision_at_{k_eval}": prec_at_k,
            f"enrichment_factor_{k_eval}": ef_k,
            "recall_at_precision_80pct": recall_at_p80,
            "recall_at_precision_50pct": recall_at_p50,
        }

    return stratified_results
