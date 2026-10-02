"""Hybrid Regulatory Allergenicity Pipeline (FAO/WHO Rule + Calibrated ML).

Combines:
1. Deterministic FAO/WHO Regulatory Stream:
   - Primary 80-mer window identity (> 35.0%, >= 29/80 identities)
   - Contiguous short match (6-aa or 8-aa)
   - Explicit short-query status ('not_applicable_short_query' for < 80 aa)
2. Probabilistic ML Stream:
   - 50-D AAC + PCP representation
   - Calibrated classification score
3. Transparent Decision Hierarchy:
   - Rule hit -> regulatory_flag: 'high_risk' immediately, regardless of ML score.
   - No rule hit -> regulatory_flag: 'no_homology_evidence', ML decides.
   - Decision basis: 'who_fao_primary', 'who_fao_short_match', or 'ml_only'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.allergen_db import ReferenceAllergenDatabase
from src.features.aac import calculate_aac_vector
from src.features.pcp import calculate_pcp_vector
from src.models.fao_who_engine import FAOWHOAllergenicityEngine

logger = logging.getLogger(__name__)


class HybridAllergenPipeline:
    """Production-grade hybrid immunological allergenicity screening system."""

    def __init__(
        self,
        ml_model: Any,
        rule_engine: FAOWHOAllergenicityEngine,
        operating_threshold: float = 0.50,
        model_name: str = "Hybrid_FAO_WHO_RandomForest_v2.0",
    ):
        self.ml_model = ml_model
        self.rule_engine = rule_engine
        self.operating_threshold = operating_threshold
        self.model_name = model_name

    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract standard 50-D AAC + PCP feature vector."""
        clean = sequence.strip().upper()
        aac = calculate_aac_vector(clean)
        pcp = calculate_pcp_vector(clean)
        return np.hstack([aac, pcp]).reshape(1, -1)

    def predict(
        self,
        sequence: str,
        threshold: Optional[float] = None,
        excluded_ids: Optional[Set[str]] = None,
        excluded_clusters: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Classify candidate peptide sequence across both regulatory and ML evidence streams.

        Positive class = Allergenic Hazard.
        """
        th = threshold if threshold is not None else self.operating_threshold
        clean_seq = sequence.strip().upper()
        q_len = len(clean_seq)

        # 1. Deterministic Regulatory Stream
        reg_res = self.rule_engine.evaluate(
            query=clean_seq,
            excluded_ids=excluded_ids,
            excluded_clusters=excluded_clusters,
        )

        # 2. Probabilistic ML Stream
        feat = self.extract_features(clean_seq)
        probs = self.ml_model.predict_proba(feat)[0]
        ml_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

        # 3. Transparent Hierarchical Synthesis
        rule_hit = reg_res["rule_hit"]
        decision_basis = reg_res["decision_basis"]
        regulatory_flag = reg_res["regulatory_flag"]

        if rule_hit:
            # Immediate high-risk override per international regulatory guidelines
            final_call = "allergen"
            is_allergen = True
            risk_tier = "high_risk"
            final_score = round(float(max(ml_prob, 0.85)), 4)
        else:
            # Absence of homology hit is never reported as non-allergen; ML model decides
            is_allergen = bool(ml_prob >= th)
            final_call = "allergen" if is_allergen else "non_allergen"
            risk_tier = "moderate_risk" if is_allergen else "low_risk"
            decision_basis = "ml_only"
            final_score = round(float(ml_prob), 4)

        return {
            "query_length": q_len,
            "final_call": {
                "classification": final_call,
                "is_allergen": is_allergen,
                "risk_tier": risk_tier,
                "decision_basis": decision_basis,
                "composite_hazard_score": final_score,
            },
            "regulatory_evidence_stream": reg_res,
            "ml_evidence_stream": {
                "model_name": self.model_name,
                "predicted_probability": round(float(ml_prob), 4),
                "threshold": round(th, 3),
                "ml_call": "allergen" if ml_prob >= th else "non_allergen",
                "feature_space": "50-D AAC + PCP (Pfeature standard)",
            },
            "provenance": reg_res["db_provenance"],
        }
