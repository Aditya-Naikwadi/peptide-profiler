"""Deterministic FAO/WHO Regulatory Allergenicity Rule Engine.

Implements Codex Alimentarius and FAO/WHO (2001/2003) allergenicity screening rules:
1. Primary Rule: Local Smith-Waterman alignment using BLOSUM50 (gap open: 10, gap extend: 2).
   An 80-amino-acid sliding window across the local alignment triggers if identity > 35%.
   With strict ">", this requires at least 29 identical residues out of 80 (28/80 is 35.0% and does not trigger).
2. Short-Match Rule: Exact contiguous match of 6 amino acids (Codex baseline) or 8 amino acids (configurable).
3. Short-Query Rule: Queries < 80 aa return primary_rule: "not_applicable_short_query", NEVER "pass".
4. Cluster-Aware Exclusion: Supports strict evaluation without self-hits or cluster-level leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import parasail

from src.data.allergen_db import ReferenceAllergenDatabase

logger = logging.getLogger(__name__)

# Standard Alignment Configuration per FAO/WHO / FASTA36 defaults
DEFAULT_GAP_OPEN = 10
DEFAULT_GAP_EXTEND = 2
MIN_80MER_IDENTITIES = 29  # 29/80 = 36.25% > 35.0% (strict inequality)


class FAOWHOAllergenicityEngine:
    """Production-grade deterministic regulatory allergenicity rule engine."""

    def __init__(
        self,
        db: ReferenceAllergenDatabase,
        gap_open: int = DEFAULT_GAP_OPEN,
        gap_extend: int = DEFAULT_GAP_EXTEND,
        matrix_name: str = "blosum50",
        short_match_k: int = 6,
        strict_mode_8aa: bool = False,
        min_identities_threshold: int = MIN_80MER_IDENTITIES,
    ):
        """Initialize engine with versioned reference database.

        Args:
            db: ReferenceAllergenDatabase instance.
            gap_open: Penalty for opening a gap (default 10).
            gap_extend: Penalty for extending a gap (default 2).
            matrix_name: Substitution matrix (default 'blosum50').
            short_match_k: Short-match k-mer length (default 6).
            strict_mode_8aa: If True, uses 8-aa exact match instead of 6-aa.
            min_identities_threshold: Minimum identities in 80 columns (default 29).
        """
        self.db = db
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.matrix_name = matrix_name
        self.short_match_k = 8 if strict_mode_8aa else short_match_k
        self.min_identities_threshold = min_identities_threshold
        self._matrix = parasail.blosum50

    def evaluate_short_match_rule(
        self,
        query: str,
        excluded_ids: Optional[Set[str]] = None,
        excluded_clusters: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Check for exact contiguous 6-aa (or 8-aa) match against reference allergens."""
        excluded_ids = excluded_ids or set()
        excluded_clusters = excluded_clusters or set()

        k = self.short_match_k
        clean_q = query.strip().upper()
        q_len = len(clean_q)

        if q_len < k:
            return {
                "status": "not_applicable_short_query",
                "exact_kmer_size": k,
                "matched_kmer": None,
                "matched_allergen_id": None,
                "description": f"Query length {q_len} is shorter than short-match k-mer length {k}.",
            }

        kmer_index = self.db.get_exact_kmer_index(k=k)
        matched_kmer = None
        matched_allergen_id = None

        # Check each k-mer in query
        for i in range(q_len - k + 1):
            kmer = clean_q[i : i + k]
            if kmer in kmer_index:
                # Check candidate IDs against exclusion sets
                candidates = kmer_index[kmer]
                for cid in candidates:
                    if cid in excluded_ids:
                        continue
                    # Check cluster exclusion if mapped
                    cand_obj = next((a for a in self.db.allergens if a["id"] == cid), None)
                    if cand_obj and cand_obj.get("cluster_id") in excluded_clusters:
                        continue

                    matched_kmer = kmer
                    matched_allergen_id = cid
                    break

            if matched_kmer is not None:
                break

        if matched_kmer is not None:
            return {
                "status": "hit",
                "exact_kmer_size": k,
                "matched_kmer": matched_kmer,
                "matched_allergen_id": matched_allergen_id,
                "description": f"Exact match of {k} contiguous amino acids ({matched_kmer}) to allergen {matched_allergen_id}.",
            }

        return {
            "status": "no_homology_evidence",
            "exact_kmer_size": k,
            "matched_kmer": None,
            "matched_allergen_id": None,
            "description": f"No exact contiguous {k}-mer match found in reference database.",
        }

    def evaluate_primary_80mer_rule(
        self,
        query: str,
        excluded_ids: Optional[Set[str]] = None,
        excluded_clusters: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Perform Smith-Waterman local alignment and slide an 80-column window across alignments.

        Enforces:
        - If query < 80 aa, returns 'not_applicable_short_query' (never 'pass').
        - Strict identity > 35% (at least 29 / 80 identical residues).
        """
        excluded_ids = excluded_ids or set()
        excluded_clusters = excluded_clusters or set()

        clean_q = query.strip().upper()
        q_len = len(clean_q)

        # Rule 4: Short-query handling
        if q_len < 80:
            return {
                "status": "not_applicable_short_query",
                "max_80mer_identity": None,
                "max_80mer_identities": None,
                "threshold": 0.35,
                "min_identities_required": self.min_identities_threshold,
                "best_matching_allergen_id": None,
                "aligned_region": None,
                "reason": f"primary_rule: not_applicable_short_query. Query length ({q_len} aa) is under 80 aa. Primary 80-mer window cannot be constructed.",
            }

        # Filter database candidates
        target_allergens = [
            a for a in self.db.allergens
            if a["id"] not in excluded_ids and a.get("cluster_id") not in excluded_clusters
        ]

        if not target_allergens:
            return {
                "status": "no_homology_evidence",
                "max_80mer_identity": 0.0,
                "max_80mer_identities": 0,
                "threshold": 0.35,
                "min_identities_required": self.min_identities_threshold,
                "best_matching_allergen_id": None,
                "aligned_region": None,
                "reason": "All reference allergens were excluded under current evaluation filters.",
            }

        # Fast prefilter: check candidate 6-mers and 2-mers
        q_6mers = {clean_q[i : i + 6] for i in range(q_len - 5)}
        kmer_index = self.db.get_exact_kmer_index(k=6)

        candidate_allergens = []
        for a in target_allergens:
            # If sequence is >= 80 aa, candidate must either share a 6-mer or have similar length
            candidate_allergens.append(a)

        profile = parasail.profile_create_sat(clean_q, self._matrix)

        best_identities_count = 0
        best_window_identity = 0.0
        best_allergen_id = None
        best_region_info = None

        for a in candidate_allergens:
            ref_seq = a["sequence"]
            res = parasail.sw_trace_striped_profile_sat(
                profile, ref_seq, self.gap_open, self.gap_extend
            )

            tb = res.get_traceback()
            comp_str = tb.comp
            aln_len = len(comp_str)

            if aln_len < 80:
                continue

            # Slide 80-column window across local alignment
            for w_start in range(aln_len - 80 + 1):
                w_comp = comp_str[w_start : w_start + 80]
                ident_count = w_comp.count("|")

                if ident_count > best_identities_count:
                    best_identities_count = ident_count
                    best_window_identity = float(round(ident_count / 80.0, 4))
                    best_allergen_id = a["id"]

                    # Extract aligned subsegment
                    cigar = res.cigar
                    best_region_info = {
                        "query_start": cigar.beg_query + w_start,
                        "query_end": cigar.beg_query + w_start + 80,
                        "ref_start": cigar.beg_ref + w_start,
                        "ref_end": cigar.beg_ref + w_start + 80,
                        "aligned_query_window": tb.query[w_start : w_start + 80],
                        "alignment_match_window": w_comp,
                        "aligned_ref_window": tb.ref[w_start : w_start + 80],
                    }

        # Strict identity > 35% check (at least 29 / 80)
        is_hit = bool(best_identities_count >= self.min_identities_threshold)
        status_str = "hit" if is_hit else "no_homology_evidence"

        return {
            "status": status_str,
            "max_80mer_identity": best_window_identity,
            "max_80mer_identities": best_identities_count,
            "threshold": 0.35,
            "min_identities_required": self.min_identities_threshold,
            "best_matching_allergen_id": best_allergen_id,
            "aligned_region": best_region_info,
            "reason": (
                f"Maximum 80-column window identity is {best_window_identity:.1%} "
                f"({best_identities_count}/80 identities), {'exceeding' if is_hit else 'not exceeding'} the strict >35% threshold."
            ),
        }

    def evaluate(
        self,
        query: str,
        excluded_ids: Optional[Set[str]] = None,
        excluded_clusters: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Execute full FAO/WHO regulatory rule hierarchy on query sequence."""
        clean_q = query.strip().upper()
        q_len = len(clean_q)

        short_match_res = self.evaluate_short_match_rule(
            clean_q, excluded_ids=excluded_ids, excluded_clusters=excluded_clusters
        )
        primary_res = self.evaluate_primary_80mer_rule(
            clean_q, excluded_ids=excluded_ids, excluded_clusters=excluded_clusters
        )

        # Decision hierarchy
        primary_hit = bool(primary_res.get("status") == "hit")
        short_hit = bool(short_match_res.get("status") == "hit")
        any_rule_hit = primary_hit or short_hit

        if primary_hit:
            regulatory_flag = "high_risk"
            decision_basis = "who_fao_primary"
        elif short_hit:
            regulatory_flag = "high_risk"
            decision_basis = "who_fao_short_match"
        else:
            regulatory_flag = "no_homology_evidence"
            decision_basis = "ml_only"

        return {
            "regulatory_flag": regulatory_flag,
            "decision_basis": decision_basis,
            "rule_hit": any_rule_hit,
            "primary_rule": primary_res,
            "short_match_rule": short_match_res,
            "parameters": {
                "alignment_algorithm": "Smith-Waterman (parasail)",
                "matrix": self.matrix_name,
                "gap_open": self.gap_open,
                "gap_extend": self.gap_extend,
                "primary_rule_threshold": "identities > 35.0% (>= 29/80)",
                "short_match_k": self.short_match_k,
            },
            "db_provenance": self.db.get_provenance_stamp(),
        }
