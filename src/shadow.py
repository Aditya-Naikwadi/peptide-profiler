"""Shadow Mode Execution, Canary Routing, and Auto-Rollback Engine.

Provides:
1. Parallel shadow execution comparing production vs candidate models.
2. Divergence metrics: Score delta distribution, Status disagreement rate, Top-k Jaccard overlap, Spearman correlation.
3. Canary routing switch with automatic tripwire rollback on safety anomalies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class ShadowComparisonEngine:
    """Engine to execute candidate models in shadow mode and monitor divergence from production."""

    def __init__(
        self,
        max_divergence_tolerance: float = 0.20,
        max_disagreement_tripwire: float = 0.10,
        log_file: Optional[Union[str, Path]] = None,
    ):
        self.max_divergence_tolerance = max_divergence_tolerance
        self.max_disagreement_tripwire = max_disagreement_tripwire
        self.log_file = Path(log_file) if log_file else BASE_DIR / "logs" / "shadow_evaluations.jsonl"
        self.is_rolled_back = False
        self.rollback_reason = None

    def compare_batches(
        self,
        production_profiles: List[Dict[str, Any]],
        candidate_profiles: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compare two batches profiled by production and candidate models respectively."""
        n = len(production_profiles)
        if n == 0 or len(candidate_profiles) != n:
            return {"error": "Mismatched or empty profile batches"}

        score_deltas = []
        status_disagreements = []
        prod_scores = []
        cand_scores = []

        prod_ranked = sorted(production_profiles, key=lambda x: x["desirability"].get("conservative_rank_score", 0.0), reverse=True)
        cand_ranked = sorted(candidate_profiles, key=lambda x: x["desirability"].get("conservative_rank_score", 0.0), reverse=True)

        prod_top_ids = set(p["id"] for p in prod_ranked[:top_k])
        cand_top_ids = set(p["id"] for p in cand_ranked[:top_k])

        # Top-k Jaccard overlap
        jaccard_overlap = len(prod_top_ids.intersection(cand_top_ids)) / max(1, len(prod_top_ids.union(cand_top_ids)))

        cand_map = {p["id"]: p for p in candidate_profiles}

        for p_prod in production_profiles:
            cid = p_prod["id"]
            p_cand = cand_map.get(cid)
            if not p_cand:
                continue

            s_prod = p_prod["desirability"].get("conservative_rank_score", 0.0)
            s_cand = p_cand["desirability"].get("conservative_rank_score", 0.0)
            delta = abs(s_prod - s_cand)

            score_deltas.append(delta)
            prod_scores.append(s_prod)
            cand_scores.append(s_cand)

            stat_prod = p_prod.get("candidate_status", "APPROVED")
            stat_cand = p_cand.get("candidate_status", "APPROVED")

            if stat_prod != stat_cand:
                status_disagreements.append({
                    "id": cid,
                    "prod_status": stat_prod,
                    "cand_status": stat_cand,
                    "prod_score": s_prod,
                    "cand_score": s_cand,
                    "delta": round(delta, 4),
                })

        disagreement_rate = len(status_disagreements) / max(1, n)
        mean_delta = float(round(np.mean(score_deltas), 4)) if score_deltas else 0.0
        max_delta = float(round(np.max(score_deltas), 4)) if score_deltas else 0.0

        if len(prod_scores) > 1 and float(np.std(prod_scores)) > 1e-6 and float(np.std(cand_scores)) > 1e-6:
            spearman_corr, _ = spearmanr(prod_scores, cand_scores)
        else:
            spearman_corr = 1.0 if np.allclose(prod_scores, cand_scores) else 0.0

        # Automatic Rollback Tripwire Check
        tripwire_triggered = False
        if disagreement_rate > self.max_disagreement_tripwire:
            tripwire_triggered = True
            self.is_rolled_back = True
            self.rollback_reason = f"Candidate disagreement rate ({disagreement_rate:.1%}) exceeded safety tripwire ({self.max_disagreement_tripwire:.1%})"
            logger.critical(f"AUTO-ROLLBACK TRIGGERED: {self.rollback_reason}")

        report = {
            "batch_size": n,
            "top_k": top_k,
            "top_k_jaccard_overlap": round(jaccard_overlap, 4),
            "spearman_rank_correlation": round(spearman_corr, 4) if not np.isnan(spearman_corr) else 1.0,
            "mean_score_delta": mean_delta,
            "max_score_delta": max_delta,
            "status_disagreement_count": len(status_disagreements),
            "status_disagreement_rate": round(disagreement_rate, 4),
            "tripwire_triggered": tripwire_triggered,
            "canary_status": "ROLLED_BACK" if tripwire_triggered else "HEALTHY",
            "disagreement_cases": status_disagreements,
        }

        # Log to file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.error(f"Failed to write shadow comparison: {e}")

        return report
