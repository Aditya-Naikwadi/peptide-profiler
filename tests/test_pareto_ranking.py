"""Unit tests for Joint Candidate Pareto Ranking, Stability Gates, and Gating Hierarchy."""

import math
import numpy as np
import pytest

from src.models.pareto import (
    JointCandidateRanker,
    calculate_aggregation_score,
    check_uncertainty_aware_dominance,
    compute_crowding_distance,
    compute_derringer_suich_desirability,
    evaluate_stability_gate,
    fast_non_dominated_sort,
)


# =========================================================================
# 1. MATHEMATICAL ACCEPTANCE: MONOTONE RESCALING INVARIANCE
# =========================================================================

def test_pareto_ranking_monotone_rescaling_invariance():
    """Verify Pareto non-dominated sorting is mathematically invariant to monotone axis transformations.
    
    Non-dominated sorting relies exclusively on pairwise ordinal comparisons:
    z_k(A) >= z_k(B) <=> g(z_k(A)) >= g(z_k(B)) for any strictly increasing g.
    """
    np.random.seed(42)
    # 20 points in 3-objective space
    Z_original = np.random.uniform(0.1, 0.9, size=(20, 3))

    # Base Pareto fronts
    base_fronts = fast_non_dominated_sort(Z_original, uncertainty_margin=0.0)

    # Monotonic transformation 1: Power transformation (g(z) = z^3)
    Z_power = Z_original ** 3.0
    power_fronts = fast_non_dominated_sort(Z_power, uncertainty_margin=0.0)
    assert power_fronts == base_fronts, "Pareto fronts changed under power transformation!"

    # Monotonic transformation 2: Logarithmic transformation (g(z) = log(1 + 10*z))
    Z_log = np.log1p(10.0 * Z_original)
    log_fronts = fast_non_dominated_sort(Z_log, uncertainty_margin=0.0)
    assert log_fronts == base_fronts, "Pareto fronts changed under log transformation!"

    # Monotonic transformation 3: Affine scaling (g(z) = 150*z + 42)
    Z_affine = 150.0 * Z_original + 42.0
    affine_fronts = fast_non_dominated_sort(Z_affine, uncertainty_margin=0.0)
    assert affine_fronts == base_fronts, "Pareto fronts changed under affine scaling!"

    # Monotonic transformation 4: Mixed strictly increasing functions per column
    Z_mixed = np.empty_like(Z_original)
    Z_mixed[:, 0] = np.exp(Z_original[:, 0])         # exponential on col 0
    Z_mixed[:, 1] = Z_original[:, 1] ** 2.5          # power on col 1
    Z_mixed[:, 2] = np.sqrt(Z_original[:, 2])        # square root on col 2

    mixed_fronts = fast_non_dominated_sort(Z_mixed, uncertainty_margin=0.0)
    assert mixed_fronts == base_fronts, "Pareto fronts changed under mixed monotone transformations!"


# =========================================================================
# 2. UNCERTAINTY-AWARE DOMINANCE
# =========================================================================

def test_uncertainty_aware_dominance():
    """Verify uncertainty margin prevents spurious dominance within statistical error bounds."""
    # Point A has a tiny nominal advantage over Point B (+0.02 on axis 0, identical on others)
    z_a = np.array([0.82, 0.80, 0.80])
    z_b = np.array([0.80, 0.80, 0.80])

    # Standard dominance (margin = 0.0): A dominates B
    assert check_uncertainty_aware_dominance(z_a, z_b, uncertainty_margin=0.0) is True
    assert check_uncertainty_aware_dominance(z_b, z_a, uncertainty_margin=0.0) is False

    # Uncertainty-aware (margin = 0.01): difference 0.02 > 0.01 -> A still dominates B
    assert check_uncertainty_aware_dominance(z_a, z_b, uncertainty_margin=0.01) is True

    # Uncertainty-aware (margin = 0.03): difference 0.02 <= 0.03 -> within noise, A does NOT dominate B
    assert check_uncertainty_aware_dominance(z_a, z_b, uncertainty_margin=0.03) is False


# =========================================================================
# 3. CROWDING DISTANCE TIE-BREAKING
# =========================================================================

def test_crowding_distance_boundary_and_interior():
    """Verify boundary candidates receive infinity and interior points receive normalized spacing."""
    # Front with 4 candidates along a 2D trade-off line
    obj_matrix = np.array([
        [0.10, 0.90],
        [0.30, 0.70],
        [0.60, 0.40],
        [0.90, 0.10],
    ])
    front = [0, 1, 2, 3]

    distances = compute_crowding_distance(front, obj_matrix)

    # Min and max boundaries on both objectives must be infinite
    assert math.isinf(distances[0])
    assert math.isinf(distances[3])

    # Interior points must have finite, positive crowding distance
    assert math.isfinite(distances[1])
    assert distances[1] > 0.0
    assert math.isfinite(distances[2])
    assert distances[2] > 0.0


# =========================================================================
# 4. STABILITY GATE & ADVISORY FOR SHORT PEPTIDES
# =========================================================================

def test_stability_gate_advisory_for_short_peptides():
    """Verify instability index is advisory for peptides < 20 aa and aggregation is detected."""
    # 1. Short peptide (14 aa): must emit advisory_short_peptide
    short_seq = "ACDEFGHIKLMNPQ"
    res_short = evaluate_stability_gate(short_seq)
    assert res_short["is_advisory_short_peptide"] is True
    assert "advisory_short_peptide" in res_short["reason_codes"]
    assert len(res_short["advisory_notes"]) >= 1

    # 2. Long peptide (35 aa)
    long_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQR"
    res_long = evaluate_stability_gate(long_seq)
    assert res_long["is_advisory_short_peptide"] is False
    assert "advisory_short_peptide" not in res_long["reason_codes"]

    # 3. Aggregation-prone hydrophobic peptide
    agg_seq = "FFFFIIIILLLLVVVVWWWW"
    res_agg = evaluate_stability_gate(agg_seq)
    assert res_agg["aggregation_score"] > 0.50
    assert "aggregation_prone" in res_agg["reason_codes"]


# =========================================================================
# 5. DERRINGER-SUICH DESIRABILITY SCALARIZATION
# =========================================================================

def test_derringer_suich_scalarization():
    """Verify Derringer-Suich geometric mean scalarization and zero-collapse property."""
    # 1. Ideal candidate (High antigenicity, zero toxicity, zero allergenicity)
    ideal = compute_derringer_suich_desirability(p_antigen=0.85, p_toxic=0.05, p_allergen=0.05)
    assert ideal["derringer_suich_score"] == 1.0
    assert ideal["is_scalarization"] is True
    assert "Derringer-Suich" in ideal["scalarization_type"]

    # 2. Candidate breaching toxicity safety limit (p_tox = 0.70 > 0.60)
    toxic_breach = compute_derringer_suich_desirability(p_antigen=0.85, p_toxic=0.70, p_allergen=0.05)
    # Must strictly collapse to 0.0
    assert toxic_breach["derringer_suich_score"] == 0.0
    assert toxic_breach["individual_desirabilities"]["toxicity"] == 0.0

    # 3. Intermediate candidate
    inter = compute_derringer_suich_desirability(p_antigen=0.50, p_toxic=0.30, p_allergen=0.30)
    assert 0.0 < inter["derringer_suich_score"] < 1.0


# =========================================================================
# 6. GATING HIERARCHY EXECUTION
# =========================================================================

def test_joint_ranker_gating_hierarchy_order():
    """Verify strict gating hierarchy order: OOD -> WHO/FAO -> Hard Toxicity -> Stability."""
    ranker = JointCandidateRanker(
        candidate_type="vaccine",
        active_toxicity_cap=0.50,
        uncertainty_margin=0.02,
        enforce_hard_toxicity_cap=True,
    )

    candidates = [
        # Candidate 1: Clean lead
        {
            "id": "c_lead",
            "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
            "calibrated_antigenicity": 0.85,
            "calibrated_toxicity": 0.10,
            "calibrated_allergenicity": 0.10,
            "applicability_domain": {"status": "in_domain"},
        },
        # Candidate 2: Hard toxicity violation
        {
            "id": "c_toxic",
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "calibrated_antigenicity": 0.90,
            "calibrated_toxicity": 0.75,  # > 0.50 cap
            "calibrated_allergenicity": 0.10,
            "applicability_domain": {"status": "in_domain"},
        },
        # Candidate 3: Abstain / OOD (Chemical modification)
        {
            "id": "c_ood",
            "sequence": "cyclo(RGDfV)",
            "calibrated_antigenicity": 0.90,
            "calibrated_toxicity": 0.10,
            "calibrated_allergenicity": 0.10,
            "applicability_domain": {"status": "abstain", "reason": "out_of_applicability_domain"},
        },
    ]

    bundle = ranker.rank_candidates(candidates)

    assert bundle["total_ranked"] == 1
    assert bundle["total_excluded"] == 1
    assert bundle["total_abstained_ood"] == 1

    # Lead must be in Pareto Front 0
    ranked_lead = bundle["ranked_candidates"][0]
    assert ranked_lead["id"] == "c_lead"
    assert ranked_lead["pareto_ranking"]["front_index"] == 0
    assert ranked_lead["pareto_ranking"]["is_pareto_optimal"] is True
    assert "per_axis_probabilities" in ranked_lead
    assert "derringer_suich_scalarization" in ranked_lead
