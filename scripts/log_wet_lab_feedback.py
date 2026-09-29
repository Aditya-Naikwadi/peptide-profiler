"""Wet-lab assay feedback logger, bias mitigation, and periodic back-testing engine."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

FEEDBACK_LOG_PATH = BASE_DIR / "data" / "wet_lab_feedback.jsonl"


def record_assay_outcome(record: Dict[str, Any], log_file: Optional[Path] = None) -> None:
    """Record an empirical assay outcome into the versioned wet-lab feedback registry."""
    target_path = log_file or FEEDBACK_LOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def sample_candidates_with_bias_mitigation(
    ranked_profiles: List[Dict[str, Any]],
    top_k_count: int = 8,
    lower_ranked_sample_rate: float = 0.20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample candidates for experimental assay synthesis, combining top leads with stratified lower-ranked sequences.

    This mitigates survivorship and selection bias, enabling continuous recalibration across the entire score range.
    """
    rng = np.random.default_rng(seed)
    survivors = [p for p in ranked_profiles if p["candidate_status"] != "EXCLUDED"]
    excluded = [p for p in ranked_profiles if p["candidate_status"] == "EXCLUDED"]

    top_leads = survivors[:top_k_count]
    remaining = survivors[top_k_count:] + excluded

    n_exploratory = max(1, int(round(len(top_leads) * lower_ranked_sample_rate)))
    selected_exploratory = []
    if remaining:
        chosen_indices = rng.choice(len(remaining), size=min(n_exploratory, len(remaining)), replace=False)
        for idx in chosen_indices:
            cand = remaining[idx].copy()
            cand["selection_sampling_stratum"] = "STRATIFIED_LOWER_RANKED"
            selected_exploratory.append(cand)

    for cand in top_leads:
        cand["selection_sampling_stratum"] = "TOP_RANKED_LEAD"

    return top_leads + selected_exploratory


def run_periodic_backtest(
    new_database_entries: List[Dict[str, Any]],
    cluster_artifact_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute back-test evaluation on newly released Swiss-Prot database entries using Phase 1 grouped protocol."""
    from src.homology import cluster_sequences_by_homology
    from src.toxicity import predict_toxicity_onnx
    from sklearn.metrics import roc_auc_score, average_precision_score

    n = len(new_database_entries)
    if n < 5:
        return {"status": "INSUFFICIENT_ENTRIES", "count": n}

    # Cluster new entries
    clusters = cluster_sequences_by_homology(new_database_entries, identity_threshold=0.35)

    y_tox = np.array([r.get("is_toxic", 0) for r in new_database_entries])
    preds = np.array([
        predict_toxicity_onnx(r["sequence"], threshold=0.6)["toxicity_score"]
        for r in new_database_entries
    ])

    auroc = roc_auc_score(y_tox, preds) if len(np.unique(y_tox)) > 1 else 0.50
    pr_auc = average_precision_score(y_tox, preds) if len(np.unique(y_tox)) > 1 else 0.0

    return {
        "new_entries_count": n,
        "new_clusters_count": len(clusters),
        "backtest_auroc": float(round(auroc, 4)),
        "backtest_pr_auc": float(round(pr_auc, 4)),
        "status": "PASS" if auroc >= 0.85 else "RETRAIN_TRIGGERED",
    }
