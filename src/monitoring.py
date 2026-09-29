"""Production Monitoring, Request Logging, Drift Detection, and Quality Guardrails.

Provides:
1. Append-only local request auditing (no network dependency).
2. Population Stability Index (PSI) & Kolmogorov-Smirnov (KS) distribution tests.
3. Length-bin mix shift tracking.
4. Rolling adversarial validation against the baseline training/calibration set.
5. Multi-tiered alert evaluation (GREEN, YELLOW, RED) tied to config thresholds.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MONITORING_CONFIG_PATH = BASE_DIR / "config" / "monitoring_config.json"
_CACHED_MONITORING_CONFIG = None


def load_monitoring_config() -> Dict[str, Any]:
    """Load monitoring and production safety configuration."""
    global _CACHED_MONITORING_CONFIG
    if _CACHED_MONITORING_CONFIG is not None:
        return _CACHED_MONITORING_CONFIG

    if MONITORING_CONFIG_PATH.exists():
        try:
            _CACHED_MONITORING_CONFIG = json.loads(MONITORING_CONFIG_PATH.read_text(encoding="utf-8"))
            return _CACHED_MONITORING_CONFIG
        except Exception as e:
            logger.warning(f"Could not load monitoring config: {e}")

    # Fallback default
    return {
      "metadata": {"version": "1.0.0"},
      "drift_monitoring": {
        "alert_thresholds": {
          "psi": {"investigate_yellow": 0.10, "critical_red": 0.25},
          "ks_test_pvalue": {"investigate_yellow": 0.05, "critical_red": 0.01},
          "adversarial_validation_auroc": {"investigate_yellow": 0.70, "critical_red": 0.80},
          "abstention_rate": {"investigate_yellow": 0.10, "critical_red": 0.15},
        }
      },
      "logging": {"audit_log_path": "logs/inference_audit.jsonl", "enabled": True}
    }


def log_request_audit(
    sequence_record: Dict[str, Any],
    profile_result: Dict[str, Any],
    sanitizer_edit_count: int = 0,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """Log complete inference transaction to local append-only JSONL file."""
    cfg = load_monitoring_config()
    if not cfg.get("logging", {}).get("enabled", True):
        return

    if log_path is None:
        relative_path = cfg.get("logging", {}).get("audit_log_path", "logs/inference_audit.jsonl")
        log_file = BASE_DIR / relative_path
    else:
        log_file = Path(log_path)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    seq = sequence_record.get("sequence", "")
    seq_len = len(seq)
    sanitizer_edit_rate = round(sanitizer_edit_count / max(1, seq_len), 4)

    phys = profile_result.get("physicochemical", {})
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "id": profile_result.get("id"),
        "length": seq_len,
        "sanitizer_edit_rate": sanitizer_edit_rate,
        "candidate_status": profile_result.get("candidate_status", "APPROVED"),
        "reason_codes": profile_result.get("reason_codes", []),
        "abstentions": profile_result.get("abstentions", []),
        "scores": {
            "toxicity_raw": profile_result.get("toxicity", {}).get("score"),
            "toxicity_calibrated": profile_result.get("toxicity", {}).get("calibrated_probability"),
            "allergenicity_raw": profile_result.get("allergenicity", {}).get("score"),
            "allergenicity_calibrated": profile_result.get("allergenicity", {}).get("calibrated_probability"),
            "antigenicity_raw": profile_result.get("antigenicity", {}).get("score"),
            "antigenicity_calibrated": profile_result.get("antigenicity", {}).get("calibrated_probability"),
        },
        "desirability": profile_result.get("desirability", {}).get("score"),
        "conservative_rank_score": profile_result.get("desirability", {}).get("conservative_rank_score"),
        "feature_summary": {
            "mol_weight": phys.get("mol_weight"),
            "gravy": phys.get("gravy"),
            "isoelectric_point": phys.get("isoelectric_point"),
            "charge_pH7": phys.get("charge_at_pH7"),
            "aromaticity": phys.get("aromaticity"),
        },
        "metadata": {
            "model_hashes": cfg.get("model_hashes", {}),
            "feature_extractor_version": cfg.get("metadata", {}).get("feature_extractor_version", "v1.0"),
        }
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to append to audit log {log_file}: {e}")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, eps: float = 1e-4) -> float:
    """Calculate the Population Stability Index (PSI) between baseline and production distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Quantile bins based on baseline expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(max(0.0, psi), 4))


def calculate_ks(baseline: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Run two-sample Kolmogorov-Smirnov test for continuous distribution shift."""
    if len(baseline) == 0 or len(actual) == 0:
        return 0.0, 1.0
    res = ks_2samp(baseline, actual)
    return float(round(res.statistic, 4)), float(round(res.pvalue, 6))


def calculate_length_bin_mix(lengths: np.ndarray) -> Dict[str, float]:
    """Calculate fraction of sequences across standardized biophysical length bins."""
    n = max(1, len(lengths))
    return {
        "< 15 aa": float(round(np.sum(lengths < 15) / n, 4)),
        "15-50 aa": float(round(np.sum((lengths >= 15) & (lengths <= 50)) / n, 4)),
        "50-200 aa": float(round(np.sum((lengths > 50) & (lengths <= 200)) / n, 4)),
        "200-500 aa": float(round(np.sum((lengths > 200) & (lengths <= 500)) / n, 4)),
        "> 500 aa": float(round(np.sum(lengths > 500) / n, 4)),
    }


def calculate_adversarial_validation_auroc(baseline_X: np.ndarray, production_X: np.ndarray, seed: int = 42) -> float:
    """Evaluate covariate shift by training a classifier to distinguish production sequences from baseline.

    If AUROC ~ 0.50, feature distributions are indistinguishable (nominal).
    If AUROC > 0.70, significant covariate shift exists (investigate).
    If AUROC > 0.80, severe distribution drift has occurred (critical tripwire).
    """
    n_base = len(baseline_X)
    n_prod = len(production_X)
    if n_base < 10 or n_prod < 10:
        return 0.50

    X = np.vstack([baseline_X, production_X])
    y = np.hstack([np.zeros(n_base, dtype=int), np.ones(n_prod, dtype=int)])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for tr, val in skf.split(X, y):
        clf = LogisticRegression(max_iter=300, C=0.5, random_state=seed)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[val])
        oof_preds[val] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    try:
        score = roc_auc_score(y, oof_preds)
        return float(round(score, 4))
    except Exception:
        return 0.50


def evaluate_batch_drift(
    production_records: List[Dict[str, Any]],
    baseline_dataset_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end drift diagnostic comparing production batch to baseline calibration data."""
    cfg = load_monitoring_config()
    thresholds = cfg.get("drift_monitoring", {}).get("alert_thresholds", {})

    if baseline_dataset_path is None:
        base_file = BASE_DIR / "data" / "evaluation_dataset.json"
    else:
        base_file = Path(baseline_dataset_path)

    baseline = json.loads(base_file.read_text(encoding="utf-8")) if base_file.exists() else []
    if not baseline or not production_records:
        return {"alert_level": "GREEN", "status": "Insufficient data for drift analysis"}

    base_lengths = np.array([r["length"] for r in baseline])
    prod_lengths = np.array([len(r["sequence"]) for r in production_records])

    # 1. Length shift
    length_psi = calculate_psi(base_lengths, prod_lengths)
    length_ks_stat, length_ks_pval = calculate_ks(base_lengths, prod_lengths)
    base_mix = calculate_length_bin_mix(base_lengths)
    prod_mix = calculate_length_bin_mix(prod_lengths)

    # 2. Abstention Rate in production batch
    abstentions = sum(1 for r in production_records if len(r.get("abstentions", [])) > 0 or len(r["sequence"]) < 15)
    abstention_rate = round(abstentions / max(1, len(production_records)), 4)

    # 3. Model score shifts (Toxicity, Allergenicity, Antigenicity)
    # Extract production scores if present
    prod_tox = np.array([
        r.get("toxicity", {}).get("score", r.get("toxicity_score", 0.5)) for r in production_records
    ])
    base_tox = np.array([r.get("toxicity_score", 0.5) for r in baseline])
    tox_psi = calculate_psi(base_tox, prod_tox)
    tox_ks_stat, tox_ks_pval = calculate_ks(base_tox, prod_tox)

    # 4. Feature Extraction & Adversarial Validation (AAC 20-D)
    from src.physicochem import AMINO_ACIDS, calculate_aac
    base_X = np.array([[calculate_aac(r["sequence"])[aa] for aa in AMINO_ACIDS] for r in baseline])
    prod_X = np.array([[calculate_aac(r["sequence"])[aa] for aa in AMINO_ACIDS] for r in production_records])
    adversarial_auroc = calculate_adversarial_validation_auroc(base_X, prod_X)

    # Determine Alert Level based on config
    psi_red = thresholds.get("psi", {}).get("critical_red", 0.25)
    psi_yellow = thresholds.get("psi", {}).get("investigate_yellow", 0.10)
    adv_red = thresholds.get("adversarial_validation_auroc", {}).get("critical_red", 0.80)
    adv_yellow = thresholds.get("adversarial_validation_auroc", {}).get("investigate_yellow", 0.70)
    abst_red = thresholds.get("abstention_rate", {}).get("critical_red", 0.15)
    abst_yellow = thresholds.get("abstention_rate", {}).get("investigate_yellow", 0.10)

    alerts = []
    alert_level = "GREEN"

    if length_psi >= psi_red or tox_psi >= psi_red or adversarial_auroc >= adv_red or abstention_rate >= abst_red:
        alert_level = "RED"
    elif length_psi >= psi_yellow or tox_psi >= psi_yellow or adversarial_auroc >= adv_yellow or abstention_rate >= abst_yellow:
        alert_level = "YELLOW"

    if length_psi >= psi_yellow:
        alerts.append(f"Length PSI elevated: {length_psi:.4f} (threshold: {psi_yellow})")
    if tox_psi >= psi_yellow:
        alerts.append(f"Toxicity Score PSI elevated: {tox_psi:.4f} (threshold: {psi_yellow})")
    if adversarial_auroc >= adv_yellow:
        alerts.append(f"Adversarial AUROC indicates covariate shift: {adversarial_auroc:.4f} (threshold: {adv_yellow})")
    if abstention_rate >= abst_yellow:
        alerts.append(f"High sequence abstention rate: {abstention_rate:.1%} (threshold: {abst_yellow})")

    return {
        "alert_level": alert_level,
        "sample_count": len(production_records),
        "baseline_count": len(baseline),
        "alerts_triggered": alerts,
        "metrics": {
            "length_psi": length_psi,
            "length_ks": {"statistic": length_ks_stat, "pvalue": length_ks_pval},
            "toxicity_score_psi": tox_psi,
            "toxicity_score_ks": {"statistic": tox_ks_stat, "pvalue": tox_ks_pval},
            "adversarial_validation_auroc": adversarial_auroc,
            "abstention_rate": abstention_rate,
            "production_length_bin_mix": prod_mix,
            "baseline_length_bin_mix": base_mix,
        }
    }
