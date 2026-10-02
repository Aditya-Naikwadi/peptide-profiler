"""Unified Probability Calibration Module.

Implements:
1. Unified Calibrator interface (fit, predict_proba, save, load) wrapping any model.
2. Calibrator candidates:
   - Platt Scaling (Logistic regression on log-odds; recommended for N < 1000).
   - Isotonic Regression (Non-parametric step function; guarded against thin-data overfitting).
   - Temperature Scaling (Single learnable parameter T > 0 on log-odds).
3. Automatic method selection and fallback to Platt when N < 1,000 samples.
4. Bayesian prior-shift correction (Saerens et al. 2002):
   odds' = odds * [pi_t / (1 - pi_t)] / [pi_s / (1 - pi_s)]
   p' = odds' / (1 + odds')
5. Expected Calibration Error (ECE) and Brier Score quantification.
6. Reliability curve calculations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

EPSILON = 1e-7


def prob_to_logit(p: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Convert probabilities to log-odds (logits) with numerical clipping."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def logit_to_prob(z: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities via the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= bin_low) & (y_prob <= bin_high)
        else:
            mask = (y_prob >= bin_low) & (y_prob < bin_high)

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute mean squared error between binary labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(round(brier_score_loss(y_true, y_prob), 4))


def compute_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute binned empirical accuracy, mean predicted confidence, and sample counts."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bin_midpoints = []
    mean_confidences = []
    empirical_accuracies = []
    sample_counts = []

    for i in range(n_bins):
        low = float(bin_boundaries[i])
        high = float(bin_boundaries[i + 1])
        mid = round((low + high) / 2.0, 3)
        bin_midpoints.append(mid)

        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        sample_counts.append(count)

        if count > 0:
            mean_confidences.append(float(round(np.mean(y_prob[mask]), 4)))
            empirical_accuracies.append(float(round(np.mean(y_true[mask]), 4)))
        else:
            mean_confidences.append(mid)
            empirical_accuracies.append(mid)

    return {
        "n_bins": n_bins,
        "bin_midpoints": bin_midpoints,
        "mean_confidences": mean_confidences,
        "empirical_accuracies": empirical_accuracies,
        "sample_counts": sample_counts,
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
        "brier_score": compute_brier_score(y_true, y_prob),
    }


def apply_prior_shift(
    probabilities: np.ndarray,
    train_prior: float,
    target_prior: float,
    eps: float = EPSILON,
) -> np.ndarray:
    """Adjust probabilities from training prevalence to deployment prevalence (Saerens / Bayes).

    Formula:
        odds' = odds * [pi_t / (1 - pi_t)] / [pi_s / (1 - pi_s)]
        p' = odds' / (1 + odds')

    Args:
        probabilities: Calibrated probabilities under training prevalence.
        train_prior: Effective training sample prevalence pi_s in (0, 1).
        target_prior: Real-world or deployment target prevalence pi_t in (0, 1).
        eps: Numerical clipping epsilon.

    Returns:
        Adjusted probabilities p' reflecting target prevalence.
    """
    p = np.asarray(probabilities, dtype=float)
    p_clipped = np.clip(p, eps, 1.0 - eps)

    # Validate priors
    pi_s = float(np.clip(train_prior, eps, 1.0 - eps))
    pi_t = float(np.clip(target_prior, eps, 1.0 - eps))

    odds_train = p_clipped / (1.0 - p_clipped)
    adjustment_factor = (pi_t / (1.0 - pi_t)) * ((1.0 - pi_s) / pi_s)
    odds_deploy = odds_train * adjustment_factor
    p_deploy = odds_deploy / (1.0 + odds_deploy)

    return np.clip(p_deploy, 0.0, 1.0)


class BaseCalibrator(ABC):
    """Abstract base class for probability calibrators."""

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_prob_or_logits: np.ndarray) -> "BaseCalibrator":
        """Fit calibrator on validation/calibration data."""
        pass

    @abstractmethod
    def predict_proba(self, y_prob_or_logits: np.ndarray) -> np.ndarray:
        """Transform uncalibrated scores to calibrated probabilities."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Persist calibrator artifact to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseCalibrator":
        """Load calibrator artifact from disk."""
        return joblib.load(path)


class PlattCalibrator(BaseCalibrator):
    """Platt scaling (Logistic Regression on log-odds).

    Fits a sigmoid mapping:
        P(Y=1 | p) = 1 / (1 + exp(-(A * logit(p) + B)))
    Well-suited for small datasets (N < 1,000) where non-parametric estimators overfit.
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self._lr: Optional[LogisticRegression] = None
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def fit(self, y_true: np.ndarray, y_prob_or_logits: np.ndarray) -> "PlattCalibrator":
        y_true = np.asarray(y_true, dtype=int)
        raw = np.asarray(y_prob_or_logits, dtype=float)

        # Convert probabilities to logits if in [0, 1]
        if np.all((raw >= 0.0) & (raw <= 1.0)):
            logits = prob_to_logit(raw)
        else:
            logits = raw

        self._lr = LogisticRegression(C=self.C, solver="lbfgs")
        self._lr.fit(logits.reshape(-1, 1), y_true)
        self.a_ = float(self._lr.coef_[0][0])
        self.b_ = float(self._lr.intercept_[0])
        return self

    def predict_proba(self, y_prob_or_logits: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise ValueError("PlattCalibrator has not been fitted.")
        raw = np.asarray(y_prob_or_logits, dtype=float)
        if np.all((raw >= 0.0) & (raw <= 1.0)):
            logits = prob_to_logit(raw)
        else:
            logits = raw
        probs = self._lr.predict_proba(logits.reshape(-1, 1))[:, 1]
        return np.clip(probs, 0.0, 1.0)


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator.

    Fits a non-decreasing piecewise constant function.
    Recommended ONLY for larger datasets (N >= 1,000) to avoid step-function overfitting.
    """

    def __init__(self, min_samples: int = 1000):
        self.min_samples = min_samples
        self._iso: Optional[IsotonicRegression] = None
        self.training_sample_count_: Optional[int] = None
        self.thin_data_warning_: bool = False

    def fit(self, y_true: np.ndarray, y_prob_or_logits: np.ndarray) -> "IsotonicCalibrator":
        y_true = np.asarray(y_true, dtype=int)
        raw = np.asarray(y_prob_or_logits, dtype=float)

        # Map to probability scale if raw looks like unbounded logits
        if np.any(raw < 0.0) or np.any(raw > 1.0):
            probs = logit_to_prob(raw)
        else:
            probs = raw

        n_samples = len(y_true)
        self.training_sample_count_ = n_samples
        if n_samples < self.min_samples:
            self.thin_data_warning_ = True
            logger.warning(
                "Isotonic regression fitted on only %d samples (< %d recommended). "
                "Step-function overfitting may occur; consider Platt scaling.",
                n_samples,
                self.min_samples,
            )

        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(probs, y_true)
        return self

    def predict_proba(self, y_prob_or_logits: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise ValueError("IsotonicCalibrator has not been fitted.")
        raw = np.asarray(y_prob_or_logits, dtype=float)
        if np.any(raw < 0.0) or np.any(raw > 1.0):
            probs = logit_to_prob(raw)
        else:
            probs = raw
        cal_probs = self._iso.predict(probs)
        return np.clip(cal_probs, 0.0, 1.0)


class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibrator (learns single T > 0 on logits).

    Maps:
        P(Y=1 | z) = sigma(z / T)
    Preserves top-1 rank order while softening overconfident predictions.
    """

    def __init__(self):
        self.temperature_: float = 1.0

    def fit(self, y_true: np.ndarray, y_prob_or_logits: np.ndarray) -> "TemperatureCalibrator":
        y_true = np.asarray(y_true, dtype=int)
        raw = np.asarray(y_prob_or_logits, dtype=float)

        if np.all((raw >= 0.0) & (raw <= 1.0)):
            logits = prob_to_logit(raw)
        else:
            logits = raw

        def nll_loss(t: float) -> float:
            scaled_logits = logits / t
            probs = logit_to_prob(scaled_logits)
            p_clip = np.clip(probs, EPSILON, 1.0 - EPSILON)
            loss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1.0 - p_clip))
            return float(loss)

        res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method="bounded")
        self.temperature_ = float(res.x)
        return self

    def predict_proba(self, y_prob_or_logits: np.ndarray) -> np.ndarray:
        raw = np.asarray(y_prob_or_logits, dtype=float)
        if np.all((raw >= 0.0) & (raw <= 1.0)):
            logits = prob_to_logit(raw)
        else:
            logits = raw
        scaled_logits = logits / self.temperature_
        return np.clip(logit_to_prob(scaled_logits), 0.0, 1.0)


class UnifiedCalibrator(BaseCalibrator):
    """Unified calibrator wrapper with automatic fallback on small sample sizes.

    Supports:
    - method="auto": selects Platt scaling if N < min_samples_for_isotonic (1,000),
      otherwise compares Brier score between candidate calibrators.
    - method="platt"
    - method="isotonic"
    - method="temperature"
    """

    def __init__(
        self,
        method: str = "auto",
        min_samples_for_isotonic: int = 1000,
        train_prior: Optional[float] = None,
        target_prior: Optional[float] = None,
    ):
        self.method = method.lower()
        self.min_samples_for_isotonic = min_samples_for_isotonic
        self.train_prior = train_prior
        self.target_prior = target_prior

        self.selected_method_: Optional[str] = None
        self.calibrator_: Optional[BaseCalibrator] = None
        self.raw_metrics_: Optional[Dict[str, float]] = None
        self.calibrated_metrics_: Optional[Dict[str, float]] = None

    def fit(self, y_true: np.ndarray, y_prob_or_logits: np.ndarray) -> "UnifiedCalibrator":
        y_true = np.asarray(y_true, dtype=int)
        raw = np.asarray(y_prob_or_logits, dtype=float)
        n_samples = len(y_true)

        # Baseline raw metrics
        raw_probs = logit_to_prob(raw) if (np.any(raw < 0.0) or np.any(raw > 1.0)) else raw
        self.raw_metrics_ = {
            "brier_score": compute_brier_score(y_true, raw_probs),
            "ece": compute_ece(y_true, raw_probs),
        }

        # Calculate empirical training prior if not provided
        if self.train_prior is None and n_samples > 0:
            self.train_prior = float(np.mean(y_true))

        # Select method
        if self.method == "platt":
            chosen = "platt"
            self.calibrator_ = PlattCalibrator().fit(y_true, raw)
        elif self.method == "isotonic":
            chosen = "isotonic"
            self.calibrator_ = IsotonicCalibrator(min_samples=self.min_samples_for_isotonic).fit(y_true, raw)
        elif self.method == "temperature":
            chosen = "temperature"
            self.calibrator_ = TemperatureCalibrator().fit(y_true, raw)
        elif self.method == "auto":
            # Guard against step overfitting when calibration set is thin (< 1,000 samples)
            if n_samples < self.min_samples_for_isotonic:
                logger.info(
                    "Calibration set size N=%d is below %d threshold. Selecting Platt scaling (falling back from Isotonic).",
                    n_samples,
                    self.min_samples_for_isotonic,
                )
                platt = PlattCalibrator().fit(y_true, raw)
                temp = TemperatureCalibrator().fit(y_true, raw)

                # Compare Platt vs Temperature on Brier score
                brier_platt = compute_brier_score(y_true, platt.predict_proba(raw))
                brier_temp = compute_brier_score(y_true, temp.predict_proba(raw))

                if brier_platt <= brier_temp:
                    chosen = "platt"
                    self.calibrator_ = platt
                else:
                    chosen = "temperature"
                    self.calibrator_ = temp
            else:
                # N >= 1,000: compare Platt vs Isotonic
                platt = PlattCalibrator().fit(y_true, raw)
                iso = IsotonicCalibrator().fit(y_true, raw)
                brier_platt = compute_brier_score(y_true, platt.predict_proba(raw))
                brier_iso = compute_brier_score(y_true, iso.predict_proba(raw))

                if brier_iso <= brier_platt:
                    chosen = "isotonic"
                    self.calibrator_ = iso
                else:
                    chosen = "platt"
                    self.calibrator_ = platt
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.selected_method_ = chosen
        cal_probs = self.calibrator_.predict_proba(raw)
        self.calibrated_metrics_ = {
            "brier_score": compute_brier_score(y_true, cal_probs),
            "ece": compute_ece(y_true, cal_probs),
        }
        return self

    def predict_proba(
        self,
        y_prob_or_logits: np.ndarray,
        target_prior: Optional[float] = None,
    ) -> np.ndarray:
        """Predict calibrated probabilities, with optional Bayesian prior-shift adjustment."""
        if self.calibrator_ is None:
            raise ValueError("UnifiedCalibrator has not been fitted.")

        cal_p = self.calibrator_.predict_proba(y_prob_or_logits)

        # Apply prior shift if target prior is provided
        t_prior = target_prior if target_prior is not None else self.target_prior
        if t_prior is not None and self.train_prior is not None:
            return apply_prior_shift(cal_p, train_prior=self.train_prior, target_prior=t_prior)

        return cal_p


class ModelCalibratorWrapper:
    """Wraps any scikit-learn or custom model with a UnifiedCalibrator."""

    def __init__(self, base_model: Any, calibrator: UnifiedCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(
        self,
        X: np.ndarray,
        target_prior: Optional[float] = None,
    ) -> np.ndarray:
        """Compute base model probabilities and return calibrated, prior-adjusted probabilities."""
        raw_probs = self.base_model.predict_proba(X)
        if raw_probs.ndim == 2 and raw_probs.shape[1] > 1:
            p_pos = raw_probs[:, 1]
        else:
            p_pos = raw_probs.ravel()

        cal_pos = self.calibrator.predict_proba(p_pos, target_prior=target_prior)
        return np.column_stack([1.0 - cal_pos, cal_pos])

    def save(self, path: Union[str, Path]) -> None:
        """Persist wrapper model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelCalibratorWrapper":
        """Load wrapper model from disk."""
        return joblib.load(path)
