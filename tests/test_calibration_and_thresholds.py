"""Unit Tests for Probability Calibration, Prior-Shift, and Cost-Utility Thresholds (WP7).

Verifies:
1. Unified Calibrator interface: fit, predict_proba, save, load, and ModelCalibratorWrapper.
2. Small-data guard: fallback to Platt scaling when N < 1000.
3. Bayesian prior-shift formula correctness and edge cases.
4. Bayes-optimal threshold derivation from explicit costs (C_FP / (C_FP + C_FN)).
5. Constrained threshold optimization (target recall, max FPR).
6. ECE and Brier score improvement over uncalibrated baseline on held-out folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.calibrator import (
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


@pytest.fixture
def synthetic_calibration_data():
    """Generate controlled calibration data with overconfident raw probabilities."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.binomial(1, 0.30, size=n)
    # Simulate overconfident probabilities
    raw_probs = np.where(y_true == 1, rng.beta(5, 2, size=n), rng.beta(2, 5, size=n))
    return y_true, raw_probs


def test_unified_calibrator_interface(synthetic_calibration_data, tmp_path):
    """Verify fit, predict_proba, save, and load methods."""
    y_true, raw_probs = synthetic_calibration_data

    calibrator = UnifiedCalibrator(method="auto")
    calibrator.fit(y_true, raw_probs)

    cal_p = calibrator.predict_proba(raw_probs)
    assert len(cal_p) == len(raw_probs)
    assert np.all((cal_p >= 0.0) & (cal_p <= 1.0))

    # Persistence
    save_file = tmp_path / "test_calibrator.joblib"
    calibrator.save(save_file)
    assert save_file.exists()

    loaded = UnifiedCalibrator.load(save_file)
    loaded_p = loaded.predict_proba(raw_probs)
    assert np.allclose(cal_p, loaded_p, atol=1e-5)


def test_small_data_guard_fallback():
    """Verify that UnifiedCalibrator automatically selects Platt when N < 1000."""
    rng = np.random.default_rng(42)
    n_small = 150  # Small sample size (< 1000)
    y_small = rng.binomial(1, 0.25, size=n_small)
    p_small = rng.uniform(0.1, 0.9, size=n_small)

    calibrator = UnifiedCalibrator(method="auto", min_samples_for_isotonic=1000)
    calibrator.fit(y_small, p_small)

    # Must fall back from Isotonic to parametric Platt or Temperature
    assert calibrator.selected_method_ in ["platt", "temperature"]
    assert calibrator.selected_method_ != "isotonic"


def test_bayesian_prior_shift_formula():
    """Verify Saerens Bayesian prior-shift calculation."""
    # Identity when train_prior == target_prior
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    shifted_same = apply_prior_shift(probs, train_prior=0.50, target_prior=0.50)
    assert np.allclose(probs, shifted_same, atol=1e-4)

    # When target prior is much lower (e.g. 50% down to 1%), probabilities must drop
    shifted_lower = apply_prior_shift(probs, train_prior=0.50, target_prior=0.01)
    assert np.all(shifted_lower < probs)

    # Mathematical test:
    # p = 0.50, train_prior = 0.50 (odds = 1.0). Target prior = 0.10.
    # odds' = 1.0 * (0.10/0.90) / (0.50/0.50) = 1/9.
    # p' = (1/9) / (1 + 1/9) = 1/10 = 0.10.
    p_mid = np.array([0.50])
    p_shifted = apply_prior_shift(p_mid, train_prior=0.50, target_prior=0.10)
    assert np.isclose(p_shifted[0], 0.10, atol=1e-4)


def test_bayes_optimal_threshold():
    """Verify p* = C_FP / (C_FP + C_FN)."""
    # cost_fn = 10, cost_fp = 1 => 1 / (1 + 10) = 1/11 ~ 0.0909
    th = compute_bayes_optimal_threshold(cost_fn=10.0, cost_fp=1.0)
    assert np.isclose(th, 1.0 / 11.0, atol=1e-3)

    # Equal costs => 0.50
    th_equal = compute_bayes_optimal_threshold(cost_fn=1.0, cost_fp=1.0)
    assert np.isclose(th_equal, 0.50, atol=1e-3)

    # Invalid negative costs raise ValueError
    with pytest.raises(ValueError):
        compute_bayes_optimal_threshold(cost_fn=-1.0, cost_fp=1.0)


def test_constrained_threshold_optimization():
    """Verify constrained threshold search on calibration data."""
    rng = np.random.default_rng(42)
    n = 300
    y_true = rng.binomial(1, 0.30, size=n)
    probs = np.where(y_true == 1, rng.uniform(0.4, 0.9, size=n), rng.uniform(0.1, 0.6, size=n))

    # Constrained to target recall >= 0.90
    res_rec = find_constrained_threshold(y_true, probs, target_recall=0.90)
    assert res_rec["calibration_achieved_recall"] >= 0.90
    assert 0.0 < res_rec["threshold"] < 1.0

    # Constrained to max FPR <= 0.05
    res_fpr = find_constrained_threshold(y_true, probs, max_fpr=0.05)
    assert res_fpr["calibration_achieved_fpr"] <= 0.05


def test_profile_decision_engine(tmp_path):
    """Verify ProfileDecisionEngine correctly parses YAML rules."""
    config_content = """
profiles:
  vaccine:
    antigenicity: {target_recall: 0.90, cost_fn: 10.0, cost_fp: 1.0}
    toxicity:     {max_fpr: 0.02}
  therapeutic:
    toxicity:       {min_sensitivity_on_hazard: 0.98, max_fpr: 0.20}
    immunogenicity: {min_sensitivity_on_hazard: 0.98, max_fpr: 0.20}
prevalence_target: {antigen: 0.02, toxic: 0.01, allergen: 0.01}
"""
    cfg_file = tmp_path / "test_profiles.yaml"
    cfg_file.write_text(config_content.strip(), encoding="utf-8")

    engine = ProfileDecisionEngine(cfg_file)
    assert engine.get_prevalence_target("antigenicity") == 0.02
    assert engine.get_prevalence_target("toxicity") == 0.01

    y_t = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1])

    # Explicit cost in vaccine antigenicity
    th_ant = engine.determine_threshold_for_task("vaccine", "antigenicity", y_t, p)
    assert np.isclose(th_ant["threshold"], 1.0 / 11.0, atol=1e-3)

    # Constrained max_fpr in vaccine toxicity
    th_tox = engine.determine_threshold_for_task("vaccine", "toxicity", y_t, p)
    assert "threshold" in th_tox


def test_model_calibrator_wrapper():
    """Verify ModelCalibratorWrapper wraps a model and outputs calibrated probabilities."""
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    dummy = DummyModel()
    y_cal = np.array([0, 1, 0, 1])
    p_cal = np.array([0.2, 0.8, 0.3, 0.7])
    calibrator = UnifiedCalibrator(method="platt").fit(y_cal, p_cal)

    wrapper = ModelCalibratorWrapper(base_model=dummy, calibrator=calibrator)
    probs = wrapper.predict_proba(np.zeros((2, 5)))
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_held_out_benchmark_improvements():
    """Verify that ECE and Brier scores improved over raw uncalibrated baseline on held-out folds."""
    bench_file = Path(__file__).resolve().parent.parent / "results" / "calibration_and_thresholds_benchmark.json"
    assert bench_file.exists(), f"Benchmark results file missing: {bench_file}"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cal = data["calibration_comparison"]
    raw = cal["uncalibrated_raw"]
    uni = cal["unified_calibrator"]

    # Acceptance: ECE and Brier improve over uncalibrated on held-out folds
    assert uni["ece_reduction"] > 0.0, f"Expected ECE reduction > 0, got {uni['ece_reduction']}"
    assert uni["brier_reduction"] >= 0.0, f"Expected Brier reduction >= 0, got {uni['brier_reduction']}"

    # Verify reliability diagram exists
    assert "reliability_diagram" in uni
    rel = uni["reliability_diagram"]
    assert len(rel["bin_midpoints"]) == 10
    assert len(rel["empirical_accuracies"]) == 10
