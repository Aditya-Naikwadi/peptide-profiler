"""Script to compute cost-optimal operating thresholds and generate versioned config.

Computes thresholds per profile (vaccine, therapeutic) and per length bin
under realistic deployment prevalence using group-held-out calibration data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import bayesian_prior_shift

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Business Cost Matrices (Default domain placeholder costs in USD)
COST_MATRICES = {
    "vaccine": {
        "description": "Vaccine target discovery profile (Prioritizes safety and high antigenicity)",
        "deployment_prevalence": 0.02,  # 2% true prevalence in screening libraries
        "costs": {
            "toxicity": {
                "C_FN": 100000.0,  # Lethal toxicity entering animal trials
                "C_FP": 2500.0,    # Discarded viable candidate synthesis cost
            },
            "allergenicity": {
                "C_FN": 50000.0,   # Severe allergenicity in preclinical tests
                "C_FP": 2000.0,    # Discarded synthesis cost
            },
            "antigenicity": {
                "C_FN": 30000.0,   # Wasted opportunity: missed protective antigen
                "C_FP": 5000.0,    # Inactive candidate advanced to in vivo assay
            },
        },
    },
    "therapeutic": {
        "description": "Peptide therapeutic drug screening profile (Prioritizes safety and non-immunogenicity)",
        "deployment_prevalence": 0.02,
        "costs": {
            "toxicity": {
                "C_FN": 150000.0,  # Toxic therapeutic candidate in preclinical development
                "C_FP": 3000.0,    # Discarded safe peptide lead
            },
            "allergenicity": {
                "C_FN": 60000.0,   # Hypersensitivity reaction in clinic
                "C_FP": 2500.0,    # Discarded non-allergenic lead
            },
            "antigenicity": {
                "C_FN": 75000.0,   # Immunogenic lead triggers neutralizing anti-drug antibodies
                "C_FP": 3000.0,    # Discarded non-immunogenic candidate
            },
        },
    },
}


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    c_fn: float,
    c_fp: float,
    prevalence: float = 0.02,
    n_threshold_steps: int = 200,
) -> Dict[str, Any]:
    """Calculate the decision threshold that minimizes monetary expected loss per screened lead."""
    thresholds = np.linspace(0.01, 0.99, n_threshold_steps)
    min_cost = float("inf")
    best_t = 0.5
    best_stats = {}

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        tn = np.sum((y_true == 0) & (preds == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = 1.0 - tpr

        # Expected cost per candidate at deployment prevalence
        expected_cost = (fnr * prevalence * c_fn) + (fpr * (1.0 - prevalence) * c_fp)

        if expected_cost < min_cost:
            min_cost = expected_cost
            best_t = float(t)
            best_stats = {
                "threshold": float(round(best_t, 3)),
                "expected_cost_per_lead_usd": float(round(expected_cost, 2)),
                "tpr_sensitivity": float(round(tpr, 4)),
                "fpr_rate": float(round(fpr, 4)),
                "specificity": float(round(1.0 - fpr, 4)),
            }

    return best_stats


def main():
    logger.info("Computing cost-optimal thresholds across profiles and length bins...")
    dataset = json.loads(Path("data/evaluation_dataset.json").read_text(encoding="utf-8"))
    clusters = json.loads(Path("data/cluster_assignments.json").read_text(encoding="utf-8"))["assignments"]

    lengths = np.array([r["length"] for r in dataset])
    y_tox = np.array([r["is_toxic"] for r in dataset], dtype=int)
    y_alg = np.array([r["is_allergen"] for r in dataset], dtype=int)
    y_ant = np.array([r["is_antigen"] for r in dataset], dtype=int)
    cluster_list = [clusters.get(r["id"], "cluster_unknown") for r in dataset]

    # Extract features
    from src.calibration import run_group_heldout_calibration
    from src.physicochem import AMINO_ACIDS, calculate_aac, calculate_pcp_descriptors
    from src.toxicity import predict_toxicity_onnx
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    n = len(dataset)
    X_50d = np.zeros((n, 50), dtype=np.float32)
    tox_raw = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(dataset):
        seq = r["sequence"]
        aac = calculate_aac(seq)
        pcp = calculate_pcp_descriptors(seq)
        aac_vec = [aac[aa] for aa in AMINO_ACIDS]
        X_50d[i] = aac_vec + list(pcp.values())
        res = predict_toxicity_onnx(seq, threshold=0.6)
        tox_raw[i] = res["toxicity_score"] if res else 0.5

    # Grouped CV for Antigenicity & Allergenicity
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    ant_raw = np.zeros(n)
    alg_raw = np.zeros(n)

    for tr, val in sgkf.split(X_50d, y_ant, groups=cluster_list):
        clf_ant = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        clf_ant.fit(X_50d[tr], y_ant[tr])
        p = clf_ant.predict_proba(X_50d[val])
        ant_raw[val] = p[:, 1] if p.shape[1] > 1 else p[:, 0]

        clf_alg = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        clf_alg.fit(X_50d[tr], y_alg[tr])
        p2 = clf_alg.predict_proba(X_50d[val])
        alg_raw[val] = p2[:, 1] if p2.shape[1] > 1 else p2[:, 0]

    # Platt calibration
    tox_cal = run_group_heldout_calibration(y_tox, tox_raw, cluster_list)
    tox_probs = np.array(tox_cal["platt_calibrated"]["probabilities"])

    alg_cal = run_group_heldout_calibration(y_alg, alg_raw, cluster_list)
    alg_probs = np.array(alg_cal["platt_calibrated"]["probabilities"])

    ant_cal = run_group_heldout_calibration(y_ant, ant_raw, cluster_list)
    ant_probs = np.array(ant_cal["platt_calibrated"]["probabilities"])

    length_bins = [
        ("all", lambda l: np.ones(len(l), dtype=bool)),
        ("< 15 aa", lambda l: l < 15),
        ("15-50 aa", lambda l: (l >= 15) & (l <= 50)),
        ("50-200 aa", lambda l: (l > 50) & (l <= 200)),
        ("200-500 aa", lambda l: (l > 200) & (l <= 500)),
        ("> 500 aa", lambda l: l > 500),
    ]

    config_output = {
        "metadata": {
            "version": "2.0.0",
            "calibration_set_id": "swiss_prot_eval_v1_437_clusters",
            "generated_date": "2026-09-29",
            "default_deployment_prevalence": 0.02,
            "description": "Versioned screening configuration with business cost matrices and calibrated thresholds",
        },
        "profiles": {},
    }

    for profile_name, prof_cfg in COST_MATRICES.items():
        prev = prof_cfg["deployment_prevalence"]
        costs = prof_cfg["costs"]
        profile_entry = {
            "description": prof_cfg["description"],
            "deployment_prevalence": prev,
            "cost_matrix_usd": costs,
            "optimal_thresholds": {},
        }

        targets = [
            ("toxicity", y_tox, tox_probs, costs["toxicity"]["C_FN"], costs["toxicity"]["C_FP"]),
            ("allergenicity", y_alg, alg_probs, costs["allergenicity"]["C_FN"], costs["allergenicity"]["C_FP"]),
            ("antigenicity", y_ant, ant_probs, costs["antigenicity"]["C_FN"], costs["antigenicity"]["C_FP"]),
        ]

        for target_name, y, probs, c_fn, c_fp in targets:
            profile_entry["optimal_thresholds"][target_name] = {}
            for bin_name, bin_fn in length_bins:
                mask = bin_fn(lengths)
                sub_y = y[mask]
                sub_p = probs[mask]
                if len(sub_y) > 0 and np.sum(sub_y) > 0:
                    opt = find_cost_optimal_threshold(sub_y, sub_p, c_fn, c_fp, prevalence=prev)
                else:
                    # Fallback to global threshold if bin has 0 positive ground truth
                    opt = find_cost_optimal_threshold(y, probs, c_fn, c_fp, prevalence=prev)
                profile_entry["optimal_thresholds"][target_name][bin_name] = opt

        config_output["profiles"][profile_name] = profile_entry

    out_file = Path("config/screening_profiles.json")
    out_file.write_text(json.dumps(config_output, indent=2), encoding="utf-8")
    logger.info(f"Successfully computed and saved cost-optimal thresholds to {out_file}")


if __name__ == "__main__":
    main()
