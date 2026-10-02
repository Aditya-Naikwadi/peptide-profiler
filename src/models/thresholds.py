"""Cost-Utility and Constrained Decision Threshold Optimization Module.

Implements:
1. Bayes-optimal threshold derivation from explicit cost matrices:
   p* = C_FP / (C_FP + C_FN)
2. Constrained threshold optimization on calibration data:
   - Target recall / min sensitivity (e.g., target_recall: 0.90, min_sensitivity_on_hazard: 0.98).
   - Maximum False Positive Rate (e.g., max_fpr: 0.02, max_fpr: 0.20).
3. YAML profile loader (vaccine and therapeutic profiles).
4. Prevalence sensitivity tables demonstrating collapsing PPV at low target prevalence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "cost_profiles.yaml"


def compute_bayes_optimal_threshold(cost_fn: float, cost_fp: float) -> float:
    """Derive Bayes-optimal decision threshold minimizing expected financial/wet-lab loss.

    Formula:
        p* = C_FP / (C_FP + C_FN)

    Args:
        cost_fn: Regret or cost of a False Negative (missed true positive).
        cost_fp: Regret or cost of a False Positive (false alarm / synthesis cost).

    Returns:
        Bayes-optimal threshold p* in (0, 1).
    """
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError(f"Costs must be strictly positive: cost_fn={cost_fn}, cost_fp={cost_fp}")
    p_star = float(cost_fp / (cost_fp + cost_fn))
    return float(round(p_star, 4))


def find_constrained_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: Optional[float] = None,
    min_sensitivity: Optional[float] = None,
    max_fpr: Optional[float] = None,
    grid_size: int = 1000,
) -> Dict[str, Any]:
    """Optimize decision threshold on calibration data subject to recall and/or FPR constraints.

    Threshold selection rules:
    1. If target_recall or min_sensitivity is specified:
       Select the HIGHEST threshold achieving recall >= target, maximizing specificity.
    2. If max_fpr is specified:
       Select the LOWEST threshold achieving FPR <= max_fpr, maximizing recall.
    3. If both are specified:
       Search for threshold satisfying both; if joint satisfaction is unachievable,
       prioritize min_sensitivity and report constraint trade-off.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Calibrated / prior-adjusted probabilities in [0, 1].
        target_recall: Desired recall / sensitivity target (e.g. 0.90).
        min_sensitivity: Alternative name for desired sensitivity (e.g. 0.98).
        max_fpr: Maximum tolerable False Positive Rate (e.g. 0.02).
        grid_size: Number of threshold evaluation points.

    Returns:
        Dict containing optimal threshold, achieved metrics on calibration set, and rule metadata.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    req_recall = target_recall if target_recall is not None else min_sensitivity
    thresholds = np.linspace(0.001, 0.999, grid_size)

    candidates = []
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        candidates.append({
            "threshold": float(round(th, 4)),
            "recall": float(round(recall, 4)),
            "fpr": float(round(fpr, 4)),
            "specificity": float(round(spec, 4)),
            "ppv": float(round(ppv, 4)),
        })

    chosen = None
    selection_basis = ""

    # Case 1: Both min_recall and max_fpr specified
    if req_recall is not None and max_fpr is not None:
        valid = [c for c in candidates if c["recall"] >= req_recall and c["fpr"] <= max_fpr]
        if valid:
            # Pick highest threshold among valid to maximize specificity
            chosen = max(valid, key=lambda c: (c["threshold"], c["specificity"]))
            selection_basis = f"joint_constraint(recall >= {req_recall}, fpr <= {max_fpr})"
        else:
            # Joint satisfaction impossible: prioritize safety / min_sensitivity
            valid_recall = [c for c in candidates if c["recall"] >= req_recall]
            if valid_recall:
                chosen = max(valid_recall, key=lambda c: c["threshold"])
                selection_basis = f"prioritized_recall(recall >= {req_recall}, achieved_fpr={chosen['fpr']} > max_fpr={max_fpr})"
            else:
                chosen = min(candidates, key=lambda c: c["threshold"])
                selection_basis = "fallback_lowest_threshold"

    # Case 2: Only recall constrained
    elif req_recall is not None:
        valid = [c for c in candidates if c["recall"] >= req_recall]
        if valid:
            chosen = max(valid, key=lambda c: c["threshold"])
            selection_basis = f"target_recall(recall >= {req_recall})"
        else:
            chosen = min(candidates, key=lambda c: c["threshold"])
            selection_basis = "fallback_lowest_threshold"

    # Case 3: Only max_fpr constrained
    elif max_fpr is not None:
        valid = [c for c in candidates if c["fpr"] <= max_fpr]
        if valid:
            # Pick lowest threshold among valid to maximize recall
            chosen = min(valid, key=lambda c: c["threshold"])
            selection_basis = f"max_fpr(fpr <= {max_fpr})"
        else:
            chosen = max(candidates, key=lambda c: c["threshold"])
            selection_basis = "fallback_highest_threshold"

    else:
        # Default midpoint
        chosen = next((c for c in candidates if c["threshold"] >= 0.50), candidates[len(candidates) // 2])
        selection_basis = "default_midpoint(0.50)"

    return {
        "threshold": chosen["threshold"],
        "selection_basis": selection_basis,
        "calibration_achieved_recall": chosen["recall"],
        "calibration_achieved_fpr": chosen["fpr"],
        "calibration_achieved_specificity": chosen["specificity"],
        "calibration_achieved_ppv": chosen["ppv"],
    }


def evaluate_threshold_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate performance of a fixed threshold on held-out test data."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    preds = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    return {
        "threshold": float(round(threshold, 4)),
        "sensitivity": float(round(sensitivity, 4)),
        "specificity": float(round(specificity, 4)),
        "fpr": float(round(fpr, 4)),
        "ppv": float(round(ppv, 4)),
        "npv": float(round(npv, 4)),
        "balanced_accuracy": float(round(balanced_acc, 4)),
        "counts": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total": int(len(y_true)),
        },
    }


def compute_prevalence_sensitivity_table(
    sensitivity: float,
    specificity: float,
    target_prevalences: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Compute deployment PPV and screening efficiency across a spectrum of prevalences.

    Demonstrates the mathematical collapse of PPV at low prevalence:
        PPV(pi) = (TPR * pi) / (TPR * pi + FPR * (1 - pi))
        Peptides Screened per True Positive = 1 / PPV(pi)

    Args:
        sensitivity: Achieved true positive rate (TPR) at operating threshold.
        specificity: Achieved true negative rate (TNR) at operating threshold.
        target_prevalences: List of prevalences to evaluate.

    Returns:
        List of dicts with prevalence, expected PPV, and screening multiplier.
    """
    prevalences = target_prevalences or [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50]
    tpr = sensitivity
    fpr = 1.0 - specificity

    rows = []
    for pi in prevalences:
        denom = (tpr * pi) + (fpr * (1.0 - pi))
        ppv = (tpr * pi) / denom if denom > 0 else 0.0
        screened_per_tp = (1.0 / ppv) if ppv > 0 else float("inf")

        rows.append({
            "target_prevalence": float(pi),
            "prevalence_percent": f"{pi:.1%}",
            "sensitivity": float(round(tpr, 4)),
            "specificity": float(round(specificity, 4)),
            "fpr": float(round(fpr, 4)),
            "expected_ppv": float(round(ppv, 4)),
            "expected_ppv_percent": f"{ppv:.1%}",
            "peptides_screened_per_true_positive": (
                float(round(screened_per_tp, 1)) if screened_per_tp != float("inf") else None
            ),
        })

    return rows


class ProfileDecisionEngine:
    """Manages profile-specific threshold optimization from YAML configuration."""

    def __init__(self, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_prevalence_target(self, task_name: str) -> float:
        """Fetch target deployment prevalence for task."""
        prevalences = self.config.get("prevalence_target", {})
        # Map common task names
        norm_task = task_name.lower()
        if "antigen" in norm_task:
            return float(prevalences.get("antigen", 0.02))
        elif "tox" in norm_task:
            return float(prevalences.get("toxic", 0.01))
        elif "allergen" in norm_task or "immuno" in norm_task:
            return float(prevalences.get("allergen", 0.01))
        return 0.01

    def determine_threshold_for_task(
        self,
        profile_name: str,
        task_name: str,
        y_true_cal: np.ndarray,
        y_prob_cal: np.ndarray,
    ) -> Dict[str, Any]:
        """Determine threshold on calibration set using profile rule specification."""
        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            raise KeyError(f"Profile '{profile_name}' not defined in config.")

        profile_rules = profiles[profile_name]
        rule_spec = profile_rules.get(task_name, {})

        # Check for explicit costs
        if "cost_fn" in rule_spec and "cost_fp" in rule_spec:
            cost_fn = float(rule_spec["cost_fn"])
            cost_fp = float(rule_spec["cost_fp"])
            bayes_th = compute_bayes_optimal_threshold(cost_fn=cost_fn, cost_fp=cost_fp)
            opt_res = {
                "threshold": bayes_th,
                "selection_basis": f"bayes_optimal(C_FN={cost_fn}, C_FP={cost_fp})",
                "calibration_achieved_recall": None,
                "calibration_achieved_fpr": None,
            }
        else:
            opt_res = find_constrained_threshold(
                y_true=y_true_cal,
                y_prob=y_prob_cal,
                target_recall=rule_spec.get("target_recall"),
                min_sensitivity=rule_spec.get("min_sensitivity_on_hazard"),
                max_fpr=rule_spec.get("max_fpr"),
            )

        opt_res["profile"] = profile_name
        opt_res["task"] = task_name
        opt_res["rule_specification"] = rule_spec
        return opt_res
