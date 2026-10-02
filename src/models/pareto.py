"""Joint Multi-Objective Ranking and Gating Engine for Peptide Screening.

Implements:
1. Strict Gating Hierarchy (in order):
   - Step 1: Abstain / OOD -> excluded from ranking, reported in separate abstained list.
   - Step 2: WHO/FAO allergenicity rule hit -> flagged high-risk allergen.
   - Step 3: Hard toxicity cap from active cost profile -> excluded from leads.
   - Step 4: Stability & aggregation gate -> soft gate with diagnostic reason codes:
             ('unstable', 'aggregation_prone', 'advisory_short_peptide', 'extreme_net_charge').
2. Multi-Objective Pareto Ranking:
   - Objectives: (Antigenicity ^, Toxicity Risk v, Allergenicity Risk v) on calibrated probabilities.
   - Fast Non-Dominated Sorting (Deb et al., NSGA-II) for front index calculation.
   - Crowding distance calculation as within-front tie-breaker.
   - Uncertainty-aware dominance: A dominates B only if the margin exceeds statistical uncertainty.
   - Invariant to strictly monotonic transformations of the objective axes.
3. Derringer-Suich Desirability Score:
   - Optional geometric mean scalarization clearly labeled as a secondary view.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

# Normalized aggregation propensity scale (Zyggregator / AGGRESCAN derived)
# Positive = aggregation promoting; Negative = anti-aggregation / soluble
AGGREGATION_PROPENSITY_SCALE: Dict[str, float] = {
    "F": 1.5, "I": 1.3, "W": 1.2, "L": 1.1, "V": 1.0, "Y": 0.9, "M": 0.8, "C": 0.5,
    "A": 0.1, "T": 0.0, "S": -0.2, "G": -0.3, "Q": -0.3, "N": -0.4, "H": -0.5,
    "K": -1.0, "R": -1.0, "D": -1.2, "E": -1.2, "P": -1.5,
}


def calculate_aggregation_score(sequence: str) -> float:
    """Compute normalized sequence aggregation propensity score.

    Args:
        sequence: Canonical amino acid sequence string.

    Returns:
        Float score where values > 0.20 indicate elevated aggregation propensity.
    """
    clean_seq = sequence.strip().upper()
    if not clean_seq:
        return 0.0
    scores = [AGGREGATION_PROPENSITY_SCALE.get(aa, 0.0) for aa in clean_seq]
    return float(round(sum(scores) / len(clean_seq), 3))


def evaluate_stability_gate(sequence: str) -> Dict[str, Any]:
    """Evaluate biophysical stability, charge at pH 7.4, and aggregation propensity.

    The instability index is parameterized on globular proteins; for peptides
    under 20 aa, it is strictly treated as advisory with an explicit note.
    Gates are soft by default and emit diagnostic reason codes.
    """
    clean_seq = sequence.strip().upper()
    seq_len = len(clean_seq)
    analyser = ProteinAnalysis(clean_seq)

    # 1. Biophysical calculations
    instability = round(float(analyser.instability_index()), 2)
    gravy = round(float(analyser.gravy()), 3)
    charge_7_4 = round(float(analyser.charge_at_pH(7.4)), 2)
    sec_struct = analyser.secondary_structure_fraction()
    sec_dict = {
        "helix": round(float(sec_struct[0]), 3),
        "turn": round(float(sec_struct[1]), 3),
        "sheet": round(float(sec_struct[2]), 3),
    }
    agg_score = calculate_aggregation_score(clean_seq)

    # 2. Gate evaluation & reason codes
    reason_codes: List[str] = []
    advisory_notes: List[str] = []
    is_short = seq_len < 20

    if is_short:
        reason_codes.append("advisory_short_peptide")
        advisory_notes.append(
            f"Peptide length ({seq_len} aa) < 20 aa: Instability index ({instability}) "
            "is parameterized on globular proteins and is advisory only."
        )

    if instability >= 40.0:
        if not is_short:
            reason_codes.append("unstable")
        else:
            reason_codes.append("unstable_advisory")

    if agg_score > 0.20 or (sec_dict["sheet"] >= 0.35 and gravy > 0.0):
        reason_codes.append("aggregation_prone")

    if abs(charge_7_4) > 6.0:
        reason_codes.append("extreme_net_charge")

    return {
        "instability_index": instability,
        "is_stable": bool(instability < 40.0),
        "gravy": gravy,
        "charge_at_pH7_4": charge_7_4,
        "secondary_structure": sec_dict,
        "aggregation_score": agg_score,
        "is_aggregation_prone": bool("aggregation_prone" in reason_codes),
        "is_advisory_short_peptide": is_short,
        "reason_codes": reason_codes,
        "advisory_notes": advisory_notes,
    }


def compute_derringer_suich_desirability(
    p_antigen: float,
    p_toxic: float,
    p_allergen: float,
    candidate_type: str = "vaccine",
    ant_bounds: Tuple[float, float] = (0.20, 0.80),
    tox_bounds: Tuple[float, float] = (0.10, 0.60),
    alg_bounds: Tuple[float, float] = (0.10, 0.60),
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Compute Derringer-Suich (1980) multi-response desirability score (scalarization).

    Individual desirability functions d_k in [0, 1]:
    - Maximization (Antigenicity for vaccine):
      d = 0 if y <= L; ((y - L)/(T - L))^s if L < y < T; 1 if y >= T.
    - Minimization (Toxicity and Allergenicity; Antigenicity for therapeutic):
      d = 1 if y <= T; ((U - y)/(U - T))^t if T < y < U; 0 if y >= U.

    Composite Desirability D = (d1^w1 * d2^w2 * d3^w3)^(1 / sum(w)).
    """
    p_ant = float(np.clip(p_antigen, 0.0, 1.0))
    p_tox = float(np.clip(p_toxic, 0.0, 1.0))
    p_alg = float(np.clip(p_allergen, 0.0, 1.0))

    # 1. Antigenicity Desirability d_ant
    L_ant, T_ant = ant_bounds
    if candidate_type.lower() == "therapeutic":
        # Minimize immunogenicity for therapeutic
        T_ant_th, U_ant_th = 0.10, 0.50
        if p_ant <= T_ant_th:
            d_ant = 1.0
        elif p_ant >= U_ant_th:
            d_ant = 0.0
        else:
            d_ant = (U_ant_th - p_ant) / (U_ant_th - T_ant_th)
    else:
        # Maximize antigenicity for vaccine
        if p_ant <= L_ant:
            d_ant = 0.0
        elif p_ant >= T_ant:
            d_ant = 1.0
        else:
            d_ant = (p_ant - L_ant) / (T_ant - L_ant)

    # 2. Toxicity Desirability d_tox (Minimize)
    T_tox, U_tox = tox_bounds
    if p_tox <= T_tox:
        d_tox = 1.0
    elif p_tox >= U_tox:
        d_tox = 0.0
    else:
        d_tox = (U_tox - p_tox) / (U_tox - T_tox)

    # 3. Allergenicity Desirability d_alg (Minimize)
    T_alg, U_alg = alg_bounds
    if p_alg <= T_alg:
        d_alg = 1.0
    elif p_alg >= U_alg:
        d_alg = 0.0
    else:
        d_alg = (U_alg - p_alg) / (U_alg - T_alg)

    # Geometric mean composite
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3

    # If any individual desirability is zero, composite is strictly zero
    if d_ant <= 0.0 or d_tox <= 0.0 or d_alg <= 0.0:
        composite = 0.0
    else:
        composite = math.exp((w1 * math.log(d_ant) + w2 * math.log(d_tox) + w3 * math.log(d_alg)) / w_sum)

    return {
        "derringer_suich_score": round(float(composite), 4),
        "individual_desirabilities": {
            "antigenicity": round(float(d_ant), 4),
            "toxicity": round(float(d_tox), 4),
            "allergenicity": round(float(d_alg), 4),
        },
        "scalarization_type": "Derringer-Suich Geometric Mean Desirability (1980)",
        "is_scalarization": True,
    }


def check_uncertainty_aware_dominance(
    z_a: np.ndarray,
    z_b: np.ndarray,
    uncertainty_margin: Union[float, np.ndarray] = 0.0,
) -> bool:
    """Check if candidate A dominates candidate B across all maximization objectives.

    A dominates B (A > B) iff:
    1. For all objectives k: z_k(A) >= z_k(B) - margin_k
    2. For at least one objective k: z_k(A) > z_k(B) + margin_k

    When uncertainty_margin == 0.0, this reduces to standard Pareto dominance.
    """
    diff = z_a - z_b
    margin = np.asarray(uncertainty_margin, dtype=float)

    # Condition 1: A is not worse than B beyond margin on any objective
    if np.any(diff < -margin):
        return False

    # Condition 2: A is strictly better than B exceeding margin on at least one objective
    if np.any(diff > margin):
        return True

    return False


def fast_non_dominated_sort(
    objective_matrix: np.ndarray,
    uncertainty_margin: Union[float, np.ndarray] = 0.0,
) -> List[List[int]]:
    """Fast Non-Dominated Sorting algorithm (Deb et al., NSGA-II).

    Args:
        objective_matrix: Shape (N, M), where all M objectives are to be MAXIMIZED.
        uncertainty_margin: Significance margin for uncertainty-aware dominance.

    Returns:
        List of fronts, where fronts[0] is Pareto Front 0 (non-dominated), fronts[1] is Front 1, etc.
    """
    N = len(objective_matrix)
    if N == 0:
        return []

    # S[p] = set of candidates that candidate p dominates
    S: List[List[int]] = [[] for _ in range(N)]
    # n[p] = number of candidates that dominate candidate p
    n = np.zeros(N, dtype=int)

    fronts: List[List[int]] = [[]]

    for p in range(N):
        z_p = objective_matrix[p]
        for q in range(N):
            if p == q:
                continue
            z_q = objective_matrix[q]

            if check_uncertainty_aware_dominance(z_p, z_q, uncertainty_margin=uncertainty_margin):
                S[p].append(q)
            elif check_uncertainty_aware_dominance(z_q, z_p, uncertainty_margin=uncertainty_margin):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    curr_front_idx = 0
    while len(fronts[curr_front_idx]) > 0:
        next_front: List[int] = []
        for p in fronts[curr_front_idx]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        curr_front_idx += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if len(fronts) > 0 and len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def compute_crowding_distance(
    front: List[int],
    objective_matrix: np.ndarray,
) -> Dict[int, float]:
    """Compute crowding distance for candidates within a single Pareto front.

    Boundary candidates on each objective receive infinity (or 1e9).
    Interior candidates receive normalized distance between nearest neighbors.
    """
    n_pts = len(front)
    if n_pts == 0:
        return {}
    if n_pts <= 2:
        return {idx: float("inf") for idx in front}

    M = objective_matrix.shape[1]
    distances = {idx: 0.0 for idx in front}

    for m in range(M):
        # Sort front by m-th objective
        sorted_indices = sorted(front, key=lambda idx: objective_matrix[idx, m])

        # Assign infinity to boundaries
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        obj_min = objective_matrix[sorted_indices[0], m]
        obj_max = objective_matrix[sorted_indices[-1], m]
        obj_range = obj_max - obj_min

        if obj_range == 0.0:
            continue

        for i in range(1, n_pts - 1):
            if not math.isinf(distances[sorted_indices[i]]):
                prev_val = objective_matrix[sorted_indices[i - 1], m]
                next_val = objective_matrix[sorted_indices[i + 1], m]
                distances[sorted_indices[i]] += (next_val - prev_val) / obj_range

    return distances


class JointCandidateRanker:
    """Joint candidate ranking engine with strict gating and Pareto multi-objective sorting."""

    def __init__(
        self,
        candidate_type: str = "vaccine",
        active_toxicity_cap: float = 0.50,
        uncertainty_margin: float = 0.02,
        enforce_hard_toxicity_cap: bool = True,
    ):
        """Initialize ranker.

        Args:
            candidate_type: 'vaccine' (maximize antigenicity) or 'therapeutic' (minimize antigenicity).
            active_toxicity_cap: Maximum allowable calibrated toxicity probability.
            uncertainty_margin: Threshold for uncertainty-aware dominance.
            enforce_hard_toxicity_cap: If True, candidates exceeding cap are excluded from lead ranking.
        """
        self.candidate_type = candidate_type.lower()
        self.active_toxicity_cap = active_toxicity_cap
        self.uncertainty_margin = uncertainty_margin
        self.enforce_hard_toxicity_cap = enforce_hard_toxicity_cap

    def execute_gating_hierarchy(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply the 4-step gating hierarchy in strict order.

        Returns:
            Tuple of:
                1. rankable_candidates: Candidates passing hard gates entering Pareto ranking.
                2. excluded_candidates: Candidates failing hard toxicity cap.
                3. abstained_ood: Candidates abstaining due to applicability domain / OOD.
        """
        rankable: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        abstained_ood: List[Dict[str, Any]] = []

        for cand in candidates:
            # Copy to avoid mutating input
            c = dict(cand)
            seq = str(c.get("sequence", ""))
            reasons = list(c.get("reason_codes", []))

            # -------------------------------------------------------------
            # STEP 1: Abstain / OOD Gate
            # -------------------------------------------------------------
            ood_info = c.get("applicability_domain", {})
            if ood_info.get("status") == "abstain":
                c["gating_status"] = "ABSTAIN_OOD"
                c["gating_reason"] = ood_info.get("reason", "out_of_applicability_domain")
                abstained_ood.append(c)
                continue

            # -------------------------------------------------------------
            # STEP 2: WHO/FAO Allergenicity Rule Gate
            # -------------------------------------------------------------
            who_fao_hit = c.get("who_fao_rule_hit", False) or c.get("allergenicity", {}).get("who_fao_hit", False)
            if who_fao_hit:
                reasons.append("WHO_FAO_ALLERGEN_HIT")
                c["is_high_risk_allergen"] = True
            else:
                c["is_high_risk_allergen"] = False

            # -------------------------------------------------------------
            # STEP 3: Hard Toxicity Cap Gate
            # -------------------------------------------------------------
            p_tox = float(c.get("calibrated_toxicity", c.get("toxicity", {}).get("score", 0.0)))
            if p_tox >= self.active_toxicity_cap:
                reasons.append("HARD_TOXICITY_CAP_EXCEEDED")
                if self.enforce_hard_toxicity_cap:
                    c["gating_status"] = "EXCLUDED_TOXICITY"
                    c["reason_codes"] = reasons
                    excluded.append(c)
                    continue

            # -------------------------------------------------------------
            # STEP 4: Stability & Aggregation Gate (Soft diagnostic gate)
            # -------------------------------------------------------------
            stability_eval = evaluate_stability_gate(seq)
            c["stability"] = stability_eval
            for r in stability_eval["reason_codes"]:
                if r not in reasons:
                    reasons.append(r)

            c["gating_status"] = "RANKABLE"
            c["reason_codes"] = reasons
            rankable.append(c)

        return rankable, excluded, abstained_ood

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute full joint ranking across antigenicity, toxicity, and allergenicity.

        Returns:
            Dictionary containing:
                ranked_candidates: List of candidates ordered by Pareto front and crowding distance.
                excluded_candidates: Candidates excluded by hard toxicity cap.
                abstained_ood: Candidates abstained by applicability domain.
                front_summary: Summary of candidate counts per front.
        """
        rankable, excluded, abstained_ood = self.execute_gating_hierarchy(candidates)

        if not rankable:
            return {
                "ranked_candidates": [],
                "excluded_candidates": excluded,
                "abstained_ood": abstained_ood,
                "front_summary": {},
            }

        # Build Objective Matrix for Pareto Sorting
        # All M objectives must be framed as MAXIMIZATION
        obj_matrix_list = []
        for c in rankable:
            p_ant = float(c.get("calibrated_antigenicity", c.get("antigenicity", {}).get("score", 0.5)))
            p_tox = float(c.get("calibrated_toxicity", c.get("toxicity", {}).get("score", 0.0)))
            p_alg = float(c.get("calibrated_allergenicity", c.get("allergenicity", {}).get("score", 0.0)))

            # Store per-axis calibrated probabilities in candidate record
            c["per_axis_probabilities"] = {
                "antigenicity": round(p_ant, 4),
                "toxicity": round(p_tox, 4),
                "allergenicity": round(p_alg, 4),
            }

            # Optional Derringer-Suich secondary view
            ds_eval = compute_derringer_suich_desirability(
                p_antigen=p_ant,
                p_toxic=p_tox,
                p_allergen=p_alg,
                candidate_type=self.candidate_type,
            )
            c["derringer_suich_scalarization"] = ds_eval

            # Formulate Maximization Objectives
            if self.candidate_type == "therapeutic":
                # For therapeutic: Low immunogenicity, Low toxicity, Low allergenicity
                z_ant = 1.0 - p_ant
            else:
                # For vaccine: High immunogenicity, Low toxicity, Low allergenicity
                z_ant = p_ant

            z_tox = 1.0 - p_tox
            z_alg = 1.0 - p_alg

            obj_matrix_list.append([z_ant, z_tox, z_alg])

        objective_matrix = np.array(obj_matrix_list, dtype=float)

        # 1. Fast Non-Dominated Sorting with Uncertainty-Aware Dominance
        fronts = fast_non_dominated_sort(
            objective_matrix,
            uncertainty_margin=self.uncertainty_margin,
        )

        # 2. Compute Crowding Distance within each Front
        front_summary: Dict[int, int] = {}
        ordered_rankable: List[Dict[str, Any]] = []

        overall_rank = 1
        for front_idx, front_members in enumerate(fronts):
            front_summary[front_idx] = len(front_members)
            crowding_dist = compute_crowding_distance(front_members, objective_matrix)

            # Sort front members primarily by crowding distance descending (favoring diversity)
            sorted_front = sorted(
                front_members,
                key=lambda idx: (crowding_dist[idx], -rankable[idx]["per_axis_probabilities"]["toxicity"]),
                reverse=True,
            )

            for idx in sorted_front:
                cand = rankable[idx]
                cd_val = crowding_dist[idx]
                cand["pareto_ranking"] = {
                    "rank": overall_rank,
                    "front_index": front_idx,
                    "is_pareto_optimal": bool(front_idx == 0),
                    "crowding_distance": "inf" if math.isinf(cd_val) else round(float(cd_val), 4),
                    "uncertainty_margin": self.uncertainty_margin,
                }
                overall_rank += 1
                ordered_rankable.append(cand)

        return {
            "ranked_candidates": ordered_rankable,
            "excluded_candidates": excluded,
            "abstained_ood": abstained_ood,
            "front_summary": front_summary,
            "total_ranked": len(ordered_rankable),
            "total_excluded": len(excluded),
            "total_abstained_ood": len(abstained_ood),
        }
