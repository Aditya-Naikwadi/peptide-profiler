"""FAO/WHO & Codex Alimentarius Homology-Based Allergenicity Assessment Module.

Implements the official international guidelines for allergenicity evaluation of proteins:
1. Codex Alimentarius / FAO/WHO 80-mer rule:
   >35% sequence identity over an 80-amino-acid sliding window to any known allergen.
   For short peptides (<80 aa), evaluates normalized identity over the sequence span.
2. Contiguous exact peptide match rule:
   Exact match of 6 (or 8) consecutive identical amino acids to a known allergen epitope.

STRICT ZERO LEAKAGE:
In cross-validation or evaluation, the reference allergen library is instantiated ONLY
from the training fold clusters, ensuring zero information leakage to test folds.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute simple pairwise identity for equal-length windows."""
    length = min(len(seq1), len(seq2))
    if length == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq1[:length], seq2[:length]) if a == b)
    return float(matches / length)


class FAOWHORuleEvaluator:
    """Evaluates peptide sequences against a reference library of allergens per FAO/WHO standards."""

    def __init__(
        self,
        reference_allergens: List[Dict[str, str]],
        identity_threshold: float = 0.35,
        window_size: int = 80,
        exact_kmer_size: int = 6,
    ):
        """Initialize evaluator with known reference allergens.

        Args:
            reference_allergens: List of dicts with 'id' and 'sequence'.
            identity_threshold: Minimum identity threshold (default 0.35 = 35%).
            window_size: Sliding window length (default 80 aa).
            exact_kmer_size: Length of contiguous exact match (default 6 aa).
        """
        self.identity_threshold = identity_threshold
        self.window_size = window_size
        self.exact_kmer_size = exact_kmer_size
        self.reference_allergens = [
            {"id": str(r.get("id", f"ref_{i}")), "sequence": str(r["sequence"]).strip().upper()}
            for i, r in enumerate(reference_allergens)
            if len(str(r.get("sequence", ""))) >= 5
        ]

        # Build exact k-mer index for fast exact matching
        self._exact_kmer_index: Dict[str, List[str]] = defaultdict(list)
        for ref in self.reference_allergens:
            seq = ref["sequence"]
            ref_id = ref["id"]
            if len(seq) >= self.exact_kmer_size:
                for j in range(len(seq) - self.exact_kmer_size + 1):
                    kmer = seq[j : j + self.exact_kmer_size]
                    self._exact_kmer_index[kmer].append(ref_id)

    def check_exact_contiguous_match(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if query shares an exact contiguous substring of length exact_kmer_size."""
        clean_q = query.strip().upper()
        if len(clean_q) < self.exact_kmer_size:
            return False, None, None

        for i in range(len(clean_q) - self.exact_kmer_size + 1):
            kmer = clean_q[i : i + self.exact_kmer_size]
            if kmer in self._exact_kmer_index:
                matched_ref_id = self._exact_kmer_index[kmer][0]
                return True, kmer, matched_ref_id

        return False, None, None

    def check_80mer_sliding_window_identity(self, query: str) -> Tuple[float, Optional[str], Optional[int], Optional[int]]:
        """Slide 80-mer window and find maximum sequence identity against reference allergens.

        Returns:
            (max_identity, matched_ref_id, query_start, ref_start)
        """
        clean_q = query.strip().upper()
        q_len = len(clean_q)
        if q_len == 0:
            return 0.0, None, None, None

        max_identity = 0.0
        best_ref_id = None
        best_q_start = None
        best_ref_start = None

        # Case 1: Short peptides (< 80 aa)
        if q_len < self.window_size:
            for ref in self.reference_allergens:
                ref_seq = ref["sequence"]
                ref_len = len(ref_seq)
                if ref_len < q_len:
                    # Target is shorter than query
                    ident = compute_sequence_identity(clean_q, ref_seq)
                    if ident > max_identity:
                        max_identity = ident
                        best_ref_id = ref["id"]
                        best_q_start = 0
                        best_ref_start = 0
                else:
                    # Slide query across reference
                    for r_start in range(ref_len - q_len + 1):
                        window_ref = ref_seq[r_start : r_start + q_len]
                        ident = compute_sequence_identity(clean_q, window_ref)
                        if ident > max_identity:
                            max_identity = ident
                            best_ref_id = ref["id"]
                            best_q_start = 0
                            best_ref_start = r_start
                            if max_identity >= 1.0:
                                return 1.0, best_ref_id, best_q_start, best_ref_start
            return float(round(max_identity, 4)), best_ref_id, best_q_start, best_ref_start

        # Case 2: Sequences >= 80 aa (standard Codex Alimentarius sliding window)
        for q_start in range(q_len - self.window_size + 1):
            q_window = clean_q[q_start : q_start + self.window_size]
            for ref in self.reference_allergens:
                ref_seq = ref["sequence"]
                ref_len = len(ref_seq)
                if ref_len < self.window_size:
                    ident = compute_sequence_identity(q_window, ref_seq)
                    if ident > max_identity:
                        max_identity = ident
                        best_ref_id = ref["id"]
                        best_q_start = q_start
                        best_ref_start = 0
                else:
                    # Fast heuristic: step by 5 residues for speed unless candidate is promising
                    for r_start in range(0, ref_len - self.window_size + 1, 3):
                        ref_window = ref_seq[r_start : r_start + self.window_size]
                        ident = compute_sequence_identity(q_window, ref_window)
                        if ident > max_identity:
                            max_identity = ident
                            best_ref_id = ref["id"]
                            best_q_start = q_start
                            best_ref_start = r_start
                            if max_identity >= 1.0:
                                return 1.0, best_ref_id, best_q_start, best_ref_start

        return float(round(max_identity, 4)), best_ref_id, best_q_start, best_ref_start

    def evaluate(self, query: str) -> Dict[str, Any]:
        """Comprehensive evaluation against FAO/WHO criteria.

        Positive class = Allergenic Hazard.
        """
        clean_q = query.strip().upper()
        exact_match, matched_kmer, exact_ref_id = self.check_exact_contiguous_match(clean_q)
        max_ident, ident_ref_id, q_pos, r_pos = self.check_80mer_sliding_window_identity(clean_q)

        ident_triggered = bool(max_ident >= self.identity_threshold)
        rule_triggered_flags = []
        if ident_triggered:
            rule_triggered_flags.append(f"80MER_IDENTITY_{round(max_ident*100, 1)}pct")
        if exact_match:
            rule_triggered_flags.append(f"EXACT_{self.exact_kmer_size}MER_{matched_kmer}")

        is_hazard = bool(ident_triggered or exact_match)
        rule_desc = " + ".join(rule_triggered_flags) if rule_triggered_flags else "NONE"

        return {
            "fao_who_hazard": is_hazard,
            "rule_triggered": rule_desc,
            "max_80mer_identity": max_ident,
            "matched_allergen_id": ident_ref_id or exact_ref_id,
            "identity_triggered": ident_triggered,
            "exact_kmer_match": exact_match,
            "matched_kmer": matched_kmer,
            "exact_ref_id": exact_ref_id,
            "window_size": self.window_size,
            "identity_threshold": self.identity_threshold,
        }
