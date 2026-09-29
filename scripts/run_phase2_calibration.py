"""Script to execute Phase 2: Calibration, Prior Shift, and Prevalence-Aware Evaluation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    bayesian_prior_shift,
    calculate_ppv_at_prevalence,
    evaluate_length_stratified,
    run_group_heldout_calibration,
)
from src.physicochem import AMINO_ACIDS, calculate_aac, calculate_pcp_descriptors
from src.toxicity import predict_toxicity_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


def load_dataset_and_clusters():
    records = json.loads(Path("data/evaluation_dataset.json").read_text(encoding="utf-8"))
    clusters = json.loads(Path("data/cluster_assignments.json").read_text(encoding="utf-8"))["assignments"]
    for r in records:
        r["cluster_id"] = clusters.get(r["id"], "cluster_unknown")
    return records


def extract_features(records):
    n = len(records)
    X_aac = np.zeros((n, 20), dtype=np.float32)
    X_50d = np.zeros((n, 50), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.float32)

    y_tox = np.array([r["is_toxic"] for r in records], dtype=int)
    y_alg = np.array([r["is_allergen"] for r in records], dtype=int)
    y_ant = np.array([r["is_antigen"] for r in records], dtype=int)
    clusters = [r["cluster_id"] for r in records]

    for i, r in enumerate(records):
        seq = r["sequence"]
        lengths[i] = len(seq)
        aac = calculate_aac(seq)
        pcp = calculate_pcp_descriptors(seq)
        aac_vec = [aac[aa] for aa in AMINO_ACIDS]
        X_aac[i] = aac_vec
        X_50d[i] = aac_vec + list(pcp.values())

    return X_aac, X_50d, lengths, y_tox, y_alg, y_ant, clusters


def get_grouped_cv_probabilities(X, y, clusters, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    probs = np.zeros(len(y))
    for tr, val in sgkf.split(X, y, groups=clusters):
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[val])
        probs[val] = p[:, 1] if p.shape[1] > 1 else p[:, 0]
    return probs


def main():
    logger.info("Starting Phase 2: Calibration, Prior Shift, and Prevalence Evaluation...")
    records = load_dataset_and_clusters()
    X_aac, X_50d, lengths, y_tox, y_alg, y_ant, clusters = extract_features(records)
    logger.info(f"Loaded {len(records)} records across {len(set(clusters))} homology clusters.")

    # 1. Get Out-of-Fold / Raw Probabilities
    logger.info("Generating grouped out-of-fold probabilities...")

    # For Toxicity: Evaluate upstream ToxinPred2 ONNX model raw scores
    logger.info("Extracting ToxinPred2 ONNX raw scores on evaluation dataset...")
    y_tox_prob = np.zeros(len(records))
    for i, r in enumerate(records):
        res = predict_toxicity_onnx(r["sequence"], threshold=0.6)
        y_tox_prob[i] = res["toxicity_score"] if res else 0.5

    # For Antigenicity & Allergenicity: Grouped CV probabilities on 50-D features
    logger.info("Computing grouped CV probabilities for Antigenicity & Allergenicity...")
    y_ant_prob = get_grouped_cv_probabilities(X_50d, y_ant, clusters)
    y_alg_prob = get_grouped_cv_probabilities(X_50d, y_alg, clusters)

    models_data = [
        ("toxicity", y_tox, y_tox_prob, 0.60, "ToxinPred2 ONNX RF"),
        ("antigenicity", y_ant, y_ant_prob, 0.50, "50-D AAC+PCP Grouped RF"),
        ("allergenicity", y_alg, y_alg_prob, 0.50, "50-D AAC+PCP Grouped RF"),
    ]

    results = {
        "metadata": {
            "evaluation_date": "2026-09-29",
            "dataset_size": len(records),
            "clusters_count": len(set(clusters)),
            "length_distribution": {
                "<15 aa": int(np.sum(lengths < 15)),
                "15-50 aa": int(np.sum((lengths >= 15) & (lengths <= 50))),
                "50-200 aa": int(np.sum((lengths > 50) & (lengths <= 200))),
                "200-500 aa": int(np.sum((lengths > 200) & (lengths <= 500))),
                ">500 aa": int(np.sum(lengths > 500)),
            },
        },
        "calibration": {},
        "stakeholder_prevalence_table": {},
        "length_stratified": {},
    }

    prevalences = [0.01, 0.02, 0.03]  # 1%, 2%, 3%

    for model_name, y_true, y_prob, def_thresh, desc in models_data:
        logger.info(f"--- Running Calibration & Prior Shift for {model_name} ---")

        # 1. Group-held-out calibration
        cal_res = run_group_heldout_calibration(y_true, y_prob, clusters)
        results["calibration"][model_name] = {
            "description": desc,
            "raw_brier": cal_res["raw"]["brier_score"],
            "raw_ece": cal_res["raw"]["ece"],
            "platt_brier": cal_res["platt_calibrated"]["brier_score"],
            "platt_ece": cal_res["platt_calibrated"]["ece"],
            "platt_ece_reduction_pct": round(
                ((cal_res["raw"]["ece"] - cal_res["platt_calibrated"]["ece"]) / cal_res["raw"]["ece"]) * 100, 2
            ),
            "isotonic_brier": cal_res["isotonic_calibrated"]["brier_score"],
            "isotonic_ece": cal_res["isotonic_calibrated"]["ece"],
            "reliability_curve_raw": cal_res["raw"]["reliability_curve"],
            "reliability_curve_platt": cal_res["platt_calibrated"]["reliability_curve"],
        }

        # 2. Sensitivity and Specificity at default threshold
        y_pred = (y_prob >= def_thresh).astype(int)
        tn, fp, fn, tp = (
            int(np.sum((y_true == 0) & (y_pred == 0))),
            int(np.sum((y_true == 0) & (y_pred == 1))),
            int(np.sum((y_true == 1) & (y_pred == 0))),
            int(np.sum((y_true == 1) & (y_pred == 1))),
        )
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Calculate PPV at deployment prevalences
        stakeholder_entries = []
        for prev in prevalences:
            raw_ppv = calculate_ppv_at_prevalence(sens, spec, prev)
            # True positives out of 100 flags
            true_pos_per_100 = round(raw_ppv * 100, 1)
            false_pos_per_100 = round((1.0 - raw_ppv) * 100, 1)

            stakeholder_entries.append({
                "deployment_prevalence_pct": f"{prev * 100:.0f}%",
                "sensitivity_recall": round(sens, 4),
                "specificity": round(spec, 4),
                "ppv_precision_at_prevalence": raw_ppv,
                "true_positives_per_100_flags": true_pos_per_100,
                "false_positives_per_100_flags": false_pos_per_100,
                "interpretation": f"At {prev * 100:.0f}% true prevalence, out of 100 candidates flagged by the model, {true_pos_per_100:.0f} are true positives and {false_pos_per_100:.0f} are false alarms.",
            })

        results["stakeholder_prevalence_table"][model_name] = stakeholder_entries

        # 3. Length-Stratified Evaluation
        logger.info(f"Computing length-stratified evaluation for {model_name}...")
        strat_res = evaluate_length_stratified(y_true, y_prob, lengths, threshold=def_thresh, top_k=10)
        results["length_stratified"][model_name] = strat_res

    # Save to versioned JSON
    out_file = Path("results/phase2_calibration_and_prevalence_report.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Phase 2 complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
