"""Public API and Standard Contract Serialization for Peptide Profiler.

Implements the unified output contract schema:
{
  "sequence": "...",
  "status": "ok|abstain",
  "antigenicity": {
    "p_cal": 0.0,
    "p_prior_adj": 0.0,
    "pred_set": [],
    "decision": "..."
  },
  "toxicity": {
    "p_cal": 0.0,
    "p_prior_adj": 0.0,
    "pred_set": [],
    "decision": "..."
  },
  "allergenicity": {
    "who_fao": {
      "hit": false,
      "details": "..."
    },
    "ml": {
      "p_cal": 0.0,
      "p_prior_adj": 0.0,
      "pred_set": []
    },
    "decision_basis": "..."
  },
  "stability": {
    "flags": [],
    "advisory": false
  },
  "pareto": {
    "front": 1,
    "crowding": 0.0
  },
  "profile": "vaccine|therapeutic",
  "versions": {
    "models": "...",
    "allergen_db": "...",
    "esm": "..."
  }
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.aggregator import profile_multiple_sequences, profile_sequence
from src.models.calibrator import apply_prior_shift
from src.models.conformal import MondrianConformalClassifier
from src.models.ood import check_chemical_modifications, evaluate_rule_guards

logger = logging.getLogger(__name__)

# Pinned production version metadata
PRODUCTION_VERSIONS = {
    "models": "sha256:5683cd21b20c_platt_v1",
    "allergen_db": "allergenonline_v21_hash_9f481c",
    "esm": "facebook/esm2_t6_8M_UR50D_int8_rev_main",
}

# Empirical training set priors from curated database
TRAINING_PRIORS = {
    "antigenicity": 0.45,
    "toxicity": 0.277,
    "allergenicity": 0.25,
}

# Frozen conformal quantiles fitted on group-held-out calibration set (alpha=0.05)
# Class 0: Non-hit; Class 1: Hit
FROZEN_CONFORMAL_QUANTILES = {
    "antigenicity": {0: 0.88, 1: 0.95},
    "toxicity": {0: 0.93, 1: 0.99},
    "allergenicity": {0: 0.92, 1: 0.98},
}


def compute_conformal_set(
    prob: float,
    endpoint: str,
    class_labels: Tuple[str, str],
) -> List[str]:
    """Compute prediction set given calibrated probability and frozen quantiles."""
    p = float(np.clip(prob, 0.0, 1.0))
    q0 = FROZEN_CONFORMAL_QUANTILES.get(endpoint, {}).get(0, 0.95)
    q1 = FROZEN_CONFORMAL_QUANTILES.get(endpoint, {}).get(1, 0.95)

    pred_set: List[str] = []
    # 0 in set if nonconformity s(0) = p <= q0
    if p <= q0:
        pred_set.append(class_labels[0])
    # 1 in set if nonconformity s(1) = 1 - p <= q1
    if (1.0 - p) <= q1:
        pred_set.append(class_labels[1])

    return pred_set


def serialize_to_standard_contract(
    profile: Dict[str, Any],
    profile_type: str = "vaccine",
    target_prevalence: float = 0.02,
) -> Dict[str, Any]:
    """Convert an internal profile dictionary into the standard single-output JSON contract.

    Args:
        profile: Candidate profile produced by aggregator / ranker.
        profile_type: 'vaccine' or 'therapeutic'.
        target_prevalence: Real-world screening target prevalence (default 0.02).

    Returns:
        Dictionary adhering strictly to the required JSON schema.
    """
    seq = str(profile.get("sequence", ""))
    seq_len = len(seq)

    # 1. Gating & Applicability Domain Status
    cand_status = profile.get("candidate_status", "APPROVED")
    gating_status = profile.get("gating_status", "RANKABLE")
    ood_info = profile.get("applicability_domain", {})

    is_abstain = (
        gating_status == "ABSTAIN_OOD"
        or ood_info.get("status") == "abstain"
        or check_chemical_modifications(seq) is not None
    )
    overall_status = "abstain" if is_abstain else "ok"

    # 2. Antigenicity
    ant_dict = profile.get("antigenicity", {})
    p_ant_cal = float(ant_dict.get("calibrated_probability", ant_dict.get("score", 0.5)))
    p_ant_prior = float(apply_prior_shift(
        p_ant_cal,
        train_prior=TRAINING_PRIORS["antigenicity"],
        target_prior=target_prevalence,
    ))
    ant_pred_set = compute_conformal_set(
        p_ant_cal,
        endpoint="antigenicity",
        class_labels=("Non-Antigen", "Antigen"),
    )
    ant_decision = "antigen" if ant_dict.get("is_antigen", p_ant_cal >= 0.50) else "non_antigen"

    # 3. Toxicity
    tox_dict = profile.get("toxicity", {})
    p_tox_cal = float(tox_dict.get("calibrated_probability", tox_dict.get("score", 0.0)))
    p_tox_prior = float(apply_prior_shift(
        p_tox_cal,
        train_prior=TRAINING_PRIORS["toxicity"],
        target_prior=target_prevalence,
    ))
    tox_pred_set = compute_conformal_set(
        p_tox_cal,
        endpoint="toxicity",
        class_labels=("Non-Toxic", "Toxic"),
    )
    tox_decision = "toxic" if tox_dict.get("is_toxic", p_tox_cal >= 0.50) else "non_toxic"

    # 4. Allergenicity & WHO/FAO Rule
    alg_dict = profile.get("allergenicity", {})
    p_alg_cal = float(alg_dict.get("calibrated_probability", alg_dict.get("score", 0.0)))
    p_alg_prior = float(apply_prior_shift(
        p_alg_cal,
        train_prior=TRAINING_PRIORS["allergenicity"],
        target_prior=target_prevalence,
    ))
    alg_pred_set = compute_conformal_set(
        p_alg_cal,
        endpoint="allergenicity",
        class_labels=("Non-Allergen", "Allergen"),
    )

    who_fao_hit = (
        profile.get("is_high_risk_allergen", False)
        or "WHO_FAO_ALLERGEN_HIT" in profile.get("reason_codes", [])
        or alg_dict.get("who_fao_hit", False)
    )

    if who_fao_hit:
        decision_basis = "who_fao_regulatory_hit"
    elif alg_dict.get("is_allergen", False):
        decision_basis = "ml_probability"
    else:
        decision_basis = "consensus"

    # 5. Stability & Advisory Flags
    stability = profile.get("stability", {})
    flags = list(profile.get("reason_codes", []))
    is_advisory = bool(
        seq_len < 20
        or stability.get("is_advisory_short_peptide", False)
        or "advisory_short_peptide" in flags
    )

    # 6. Pareto Ranking
    pareto_info = profile.get("pareto_ranking", {})
    # Default to Front 1 (1-indexed for human readability as requested in contract)
    raw_front = pareto_info.get("front_index", 0)
    front_1_indexed = int(raw_front) + 1 if isinstance(raw_front, (int, np.integer)) else 1
    crowding_val = pareto_info.get("crowding_distance", 0.0)
    if crowding_val == "inf" or crowding_val == float("inf"):
        crowding_flt = 999.0
    else:
        try:
            crowding_flt = float(crowding_val)
        except (ValueError, TypeError):
            crowding_flt = 0.0

    return {
        "sequence": seq,
        "status": overall_status,
        "antigenicity": {
            "p_cal": round(p_ant_cal, 4),
            "p_prior_adj": round(p_ant_prior, 4),
            "pred_set": ant_pred_set,
            "decision": ant_decision,
        },
        "toxicity": {
            "p_cal": round(p_tox_cal, 4),
            "p_prior_adj": round(p_tox_prior, 4),
            "pred_set": tox_pred_set,
            "decision": tox_decision,
        },
        "allergenicity": {
            "who_fao": {
                "hit": bool(who_fao_hit),
                "details": "80-aa window > 35% identity or 6-mer contiguous match to AllergenOnline reference DB"
                if who_fao_hit
                else "No significant local homology to reference allergen database",
            },
            "ml": {
                "p_cal": round(p_alg_cal, 4),
                "p_prior_adj": round(p_alg_prior, 4),
                "pred_set": alg_pred_set,
            },
            "decision_basis": decision_basis,
        },
        "stability": {
            "flags": flags,
            "advisory": is_advisory,
        },
        "pareto": {
            "front": front_1_indexed,
            "crowding": round(crowding_flt, 4),
        },
        "profile": profile_type.lower(),
        "versions": PRODUCTION_VERSIONS,
    }


def profile_peptide(
    sequence: str,
    profile: str = "vaccine",
    target_prevalence: float = 0.02,
    seq_id: str = "query_seq",
) -> Dict[str, Any]:
    """Single-sequence high-level API endpoint producing the standard output contract.

    Args:
        sequence: Amino acid sequence string.
        profile: 'vaccine' or 'therapeutic'.
        target_prevalence: Target deployment prevalence for prior adjustment.
        seq_id: Sequence identifier string.

    Returns:
        Standard output dictionary.
    """
    clean_seq = sequence.strip().upper()
    record = {"id": seq_id, "sequence": clean_seq}
    raw_profile = profile_sequence(
        record,
        candidate_type=profile,
        enforce_safety_gates=True,
    )
    return serialize_to_standard_contract(
        raw_profile,
        profile_type=profile,
        target_prevalence=target_prevalence,
    )


def profile_batch(
    records: List[Dict[str, str]],
    profile: str = "vaccine",
    target_prevalence: float = 0.02,
    rank_candidates: bool = True,
) -> List[Dict[str, Any]]:
    """Batch API endpoint with joint Pareto ranking producing standard output contracts.

    Args:
        records: List of dictionaries with 'id' and 'sequence'.
        profile: 'vaccine' or 'therapeutic'.
        target_prevalence: Target deployment prevalence.
        rank_candidates: If True, executes Pareto multi-objective ranking.

    Returns:
        List of standard output contract dictionaries.
    """
    raw_profiles = profile_multiple_sequences(
        records,
        candidate_type=profile,
        rank_candidates=rank_candidates,
        enforce_safety_gates=True,
    )
    return [
        serialize_to_standard_contract(
            p,
            profile_type=profile,
            target_prevalence=target_prevalence,
        )
        for p in raw_profiles
    ]
