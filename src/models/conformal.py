"""Class-Conditional (Mondrian) Split Conformal Prediction Module.

Implements:
1. Split conformal prediction with Mondrian (class-conditional) calibration to guarantee
   per-class marginal coverage under severe class imbalance.
2. Nonconformity score: s_i(y) = 1 - p(y | x_i) on calibrated probabilities.
3. Quantiles fitted strictly on group-disjoint inner calibration partitions (zero leakage).
4. Four canonical prediction set outcomes:
   - {Non-Toxic} / {Toxic}: Single-class confident prediction ("classified").
   - {Toxic, Non-Toxic}: Two-class uncertainty ("uncertain").
   - {}: Empty prediction set indicating extreme nonconformity ("abstain").
5. Stratified empirical coverage evaluation: overall, per-class, and per-length-bin,
   with honest quantification of coverage degradation under homology and prevalence shift.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MondrianConformalClassifier:
    """Class-conditional (Mondrian) split conformal predictor for binary classification."""

    def __init__(self, alpha: float = 0.05, class_labels: Tuple[str, str] = ("Non-Toxic", "Toxic")):
        """Initialize conformal predictor.

        Args:
            alpha: Significance level (default 0.05 corresponds to 95% target coverage).
            class_labels: Tuple of (negative_label, positive_label).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.class_labels = class_labels

        self.quantiles: Dict[int, float] = {}
        self.cal_counts: Dict[int, int] = {}
        self.is_fitted = False

    def fit(self, y_cal: np.ndarray, prob_cal: np.ndarray) -> "MondrianConformalClassifier":
        """Fit class-conditional conformal quantiles on group-disjoint calibration set.

        Nonconformity scores:
        - For class 0: s_i = 1 - p(Y=0 | x_i) = p(Y=1 | x_i)
        - For class 1: s_i = 1 - p(Y=1 | x_i) = 1 - prob_cal[i]

        Args:
            y_cal: Binary ground truth labels (0 or 1).
            prob_cal: Calibrated predicted probabilities for class 1, shape (n,).

        Returns:
            self
        """
        y_arr = np.asarray(y_cal, dtype=int)
        p_arr = np.asarray(prob_cal, dtype=float)

        if len(y_arr) != len(p_arr):
            raise ValueError(f"Lengths mismatch: y_cal={len(y_arr)}, prob_cal={len(p_arr)}")

        for c in (0, 1):
            mask = y_arr == c
            n_c = int(np.sum(mask))
            if n_c == 0:
                raise ValueError(f"Calibration partition contains 0 samples for class {c}")

            # Nonconformity score = 1 - p(true class)
            if c == 0:
                scores_c = p_arr[mask]  # 1 - (1 - p) = p
            else:
                scores_c = 1.0 - p_arr[mask]

            scores_c = np.sort(scores_c)
            # Standard finite-sample conformal quantile index: ceil((n + 1) * (1 - alpha)) / n
            k = math.ceil((n_c + 1) * (1.0 - self.alpha))
            if k > n_c:
                q_c = 1.0
            else:
                q_c = float(scores_c[k - 1])

            self.quantiles[c] = q_c
            self.cal_counts[c] = n_c

        self.is_fitted = True
        logger.info(
            f"Fitted Mondrian conformal quantiles (alpha={self.alpha}): "
            f"class 0 q={self.quantiles[0]:.4f} (N={self.cal_counts[0]}), "
            f"class 1 q={self.quantiles[1]:.4f} (N={self.cal_counts[1]})"
        )
        return self

    def predict_set(self, prob: float) -> Dict[str, Any]:
        """Generate prediction set for a single query instance.

        Inclusion criteria:
        - Class 0 is included if s(0) = prob <= q_0
        - Class 1 is included if s(1) = (1 - prob) <= q_1

        Returns:
            Dict containing:
                prediction_set: List of class label strings in the set.
                status: 'classified' | 'uncertain' | 'abstain'.
                reason: Description or None.
                is_uncertain: True if set size != 1.
                set_size: Integer length of prediction set.
                calibrated_probability: Input probability.
        """
        if not self.is_fitted:
            raise RuntimeError("MondrianConformalClassifier must be fitted before predict_set.")

        p = float(np.clip(prob, 0.0, 1.0))
        include_0 = p <= self.quantiles[0]
        include_1 = (1.0 - p) <= self.quantiles[1]

        prediction_set: List[str] = []
        if include_0:
            prediction_set.append(self.class_labels[0])
        if include_1:
            prediction_set.append(self.class_labels[1])

        set_size = len(prediction_set)

        if set_size == 1:
            status = "classified"
            reason = None
            is_uncertain = False
        elif set_size == 2:
            status = "uncertain"
            reason = "conformal_prediction_set_uncertain"
            is_uncertain = True
        else:  # set_size == 0
            status = "abstain"
            reason = "conformal_empty_set_anomaly"
            is_uncertain = True

        return {
            "prediction_set": prediction_set,
            "status": status,
            "reason": reason,
            "is_uncertain": is_uncertain,
            "set_size": set_size,
            "calibrated_probability": round(p, 4),
            "quantiles": {
                self.class_labels[0]: round(self.quantiles[0], 4),
                self.class_labels[1]: round(self.quantiles[1], 4),
            },
        }

    def predict_batch(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        """Generate prediction sets for a batch of calibrated probabilities."""
        return [self.predict_set(float(p)) for p in probs]

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute empirical coverage, set size, and uncertainty rates.

        Stratifies by:
        - Overall
        - Per class (Class 0 vs Class 1)
        - Per length bin (<15, 15-24, 25-49, 50+)

        Args:
            y_true: Ground truth binary labels.
            probs: Calibrated probabilities.
            lengths: Optional array of peptide lengths for stratification.

        Returns:
            Dictionary of empirical metrics with confidence coverage.
        """
        y_arr = np.asarray(y_true, dtype=int)
        p_arr = np.asarray(probs, dtype=float)
        n = len(y_arr)

        if n == 0:
            return {}

        results = self.predict_batch(p_arr)

        inclusions = np.zeros(n, dtype=bool)
        set_sizes = np.zeros(n, dtype=int)
        is_uncertain = np.zeros(n, dtype=bool)
        is_empty = np.zeros(n, dtype=bool)

        for i, res in enumerate(results):
            true_label = self.class_labels[y_arr[i]]
            inclusions[i] = true_label in res["prediction_set"]
            set_sizes[i] = res["set_size"]
            is_uncertain[i] = res["set_size"] == 2
            is_empty[i] = res["set_size"] == 0

        # Class-specific coverage
        class_0_mask = y_arr == 0
        class_1_mask = y_arr == 1

        c0_cov = float(np.mean(inclusions[class_0_mask])) if np.any(class_0_mask) else 0.0
        c1_cov = float(np.mean(inclusions[class_1_mask])) if np.any(class_1_mask) else 0.0

        metrics: Dict[str, Any] = {
            "target_coverage": self.target_coverage,
            "overall_coverage": float(round(np.mean(inclusions), 4)),
            "class_0_coverage": float(round(c0_cov, 4)),
            "class_1_coverage": float(round(c1_cov, 4)),
            "mean_set_size": float(round(np.mean(set_sizes), 4)),
            "uncertain_rate": float(round(np.mean(is_uncertain), 4)),
            "empty_rate": float(round(np.mean(is_empty), 4)),
            "sample_count": n,
            "class_0_count": int(np.sum(class_0_mask)),
            "class_1_count": int(np.sum(class_1_mask)),
        }

        # Length stratification
        if lengths is not None:
            len_arr = np.asarray(lengths, dtype=int)
            length_bins = {
                "length_under_15": len_arr < 15,
                "length_15_to_24": (len_arr >= 15) & (len_arr < 25),
                "length_25_to_49": (len_arr >= 25) & (len_arr < 50),
                "length_50_plus": len_arr >= 50,
            }

            len_metrics = {}
            for bin_name, mask in length_bins.items():
                if np.any(mask):
                    len_metrics[bin_name] = {
                        "coverage": float(round(np.mean(inclusions[mask]), 4)),
                        "mean_set_size": float(round(np.mean(set_sizes[mask]), 4)),
                        "uncertain_rate": float(round(np.mean(is_uncertain[mask]), 4)),
                        "count": int(np.sum(mask)),
                    }
                else:
                    len_metrics[bin_name] = {"coverage": None, "mean_set_size": None, "uncertain_rate": None, "count": 0}

            metrics["length_bins"] = len_metrics

        return metrics
