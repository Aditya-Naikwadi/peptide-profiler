"""Cross-Predictor Error Correlation and Diversity Analysis (WP6.3).

Measures:
1. Absolute error vectors: e_i = |y_i - p_i| for each prediction task (antigenicity, allergenicity, toxicity).
2. Pearson and Spearman rank correlation across predictor errors.
3. Analysis of shared failure modes across sequence lengths and physicochemical properties.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def compute_cross_predictor_error_correlation(
    y_true_dict: Dict[str, np.ndarray],
    y_prob_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute Pearson and Spearman error correlation matrices across predictors.

    Args:
        y_true_dict: Map of task_name -> binary ground truth array.
        y_prob_dict: Map of task_name -> predicted probability array.

    Returns:
        Dict with correlation matrices and summary statistics.
    """
    task_names = list(y_true_dict.keys())
    error_dict = {}

    for task in task_names:
        y_t = np.asarray(y_true_dict[task], dtype=float)
        y_p = np.asarray(y_prob_dict[task], dtype=float)
        error_dict[task] = np.abs(y_t - y_p)

    df_errors = pd.DataFrame(error_dict)
    n_samples = len(df_errors)

    pearson_matrix = {}
    spearman_matrix = {}

    for t1 in task_names:
        pearson_matrix[t1] = {}
        spearman_matrix[t1] = {}
        for t2 in task_names:
            if t1 == t2:
                pearson_matrix[t1][t2] = {"correlation": 1.0, "p_value": 0.0}
                spearman_matrix[t1][t2] = {"correlation": 1.0, "p_value": 0.0}
            else:
                p_r, p_val = pearsonr(df_errors[t1], df_errors[t2])
                s_r, s_val = spearmanr(df_errors[t1], df_errors[t2])
                pearson_matrix[t1][t2] = {"correlation": round(float(p_r), 4), "p_value": round(float(p_val), 6)}
                spearman_matrix[t1][t2] = {"correlation": round(float(s_r), 4), "p_value": round(float(s_val), 6)}

    # Mean off-diagonal correlation
    off_diag_pearson = []
    off_diag_spearman = []
    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            off_diag_pearson.append(pearson_matrix[t1][t2]["correlation"])
            off_diag_spearman.append(spearman_matrix[t1][t2]["correlation"])

    mean_pearson = float(np.mean(off_diag_pearson)) if off_diag_pearson else 0.0
    mean_spearman = float(np.mean(off_diag_spearman)) if off_diag_spearman else 0.0

    return {
        "n_samples": n_samples,
        "tasks_evaluated": task_names,
        "mean_off_diagonal_pearson_correlation": round(mean_pearson, 4),
        "mean_off_diagonal_spearman_correlation": round(mean_spearman, 4),
        "pearson_correlation_matrix": pearson_matrix,
        "spearman_correlation_matrix": spearman_matrix,
        "interpretation": (
            "Predictor errors are largely uncorrelated"
            if abs(mean_spearman) < 0.20
            else "Moderate cross-predictor error correlation detected"
        ),
    }
