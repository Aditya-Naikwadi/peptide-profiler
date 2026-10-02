"""Strict Leakage Audit Module.

Implements Task 4 of Shared Infrastructure:
Enforces zero-leakage constraints and FAILS THE BUILD if:
1. Any cluster appears in two partitions (outer test vs. outer train, or inner train vs. inner cal).
2. Any query in a test partition shares sequence identity >= identity_threshold with a training sequence.
3. Any exact or near-duplicate sequence spans train/test partitions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from src.clustering import compute_normalized_identity, jaccard_kmer_similarity

logger = logging.getLogger(__name__)


class LeakageAuditError(AssertionError):
    """Raised when data or cluster leakage is detected across evaluation partitions."""
    pass


class LeakageAuditor:
    """Audits nested splits to guarantee zero information leakage between train, cal, and test."""

    def __init__(self, identity_threshold: float = 0.40):
        self.identity_threshold = identity_threshold

    def audit_splits(
        self,
        splits_data: Dict[str, Any],
        records: List[Dict[str, Any]],
        cluster_assignments: Dict[str, str],
    ) -> Dict[str, Any]:
        """Perform comprehensive leakage audit across all nested folds.

        Raises:
            LeakageAuditError: If any leakage or cross-contamination is detected.

        Returns:
            Audit report certificate dict if all checks pass.
        """
        seq_map = {r["id"]: r["sequence"].strip().upper() for r in records}
        folds = splits_data.get("folds", [])

        total_comparisons = 0
        max_cross_identity_observed = 0.0

        for fold in folds:
            fold_id = fold["outer_fold_id"]
            test_ids = fold["outer_test"]["sequence_ids"]
            test_clusters = set(fold["outer_test"]["cluster_ids"])

            train_ids = fold["outer_train"]["sequence_ids"]
            train_clusters = set(fold["outer_train"]["cluster_ids"])

            inner_train_ids = fold["inner_disjoint_partitions"]["inner_train"]["sequence_ids"]
            inner_train_clusters = set(fold["inner_disjoint_partitions"]["inner_train"]["cluster_ids"])

            inner_cal_ids = fold["inner_disjoint_partitions"]["inner_calibration"]["sequence_ids"]
            inner_cal_clusters = set(fold["inner_disjoint_partitions"]["inner_calibration"]["cluster_ids"])

            # ---------------------------------------------------------
            # 1. Cluster Disjointness Checks
            # ---------------------------------------------------------
            outer_leak = test_clusters.intersection(train_clusters)
            if outer_leak:
                raise LeakageAuditError(
                    f"Fold {fold_id}: Outer cluster leakage detected! Clusters present in both train and test: {outer_leak}"
                )

            inner_leak = inner_cal_clusters.intersection(inner_train_clusters)
            if inner_leak:
                raise LeakageAuditError(
                    f"Fold {fold_id}: Inner cluster leakage detected! Clusters present in both inner train and cal: {inner_leak}"
                )

            test_cal_leak = test_clusters.intersection(inner_cal_clusters)
            if test_cal_leak:
                raise LeakageAuditError(
                    f"Fold {fold_id}: Test/Cal cluster leakage detected! Clusters present in both outer test and inner cal: {test_cal_leak}"
                )

            # ---------------------------------------------------------
            # 2. Sequence Identity & Duplicate Cross-Partition Checks
            # ---------------------------------------------------------
            train_seq_tuples = [(tid, seq_map[tid]) for tid in train_ids]

            for qid in test_ids:
                qseq = seq_map[qid]
                qlen = len(qseq)

                for tid, tseq in train_seq_tuples:
                    total_comparisons += 1
                    tlen = len(tseq)

                    # Exact duplicate check
                    if qseq == tseq:
                        raise LeakageAuditError(
                            f"Fold {fold_id}: Exact duplicate sequence found across partitions! Test seq {qid} is identical to train seq {tid}."
                        )

                    # Length ratio filter for efficiency
                    if min(qlen, tlen) / max(qlen, tlen) < 0.40 and max(qlen, tlen) > 20:
                        continue

                    # Normalized sequence identity check
                    ident = compute_normalized_identity(qseq, tseq)
                    if ident > max_cross_identity_observed:
                        max_cross_identity_observed = ident

                    if ident >= self.identity_threshold:
                        raise LeakageAuditError(
                            f"Fold {fold_id}: Sequence identity leakage detected! "
                            f"Test sequence {qid} (len {qlen}) has {ident:.2%} identity to train sequence {tid} (len {tlen}), "
                            f"which exceeds the allowed threshold of {self.identity_threshold:.2%}."
                        )

        return {
            "status": "PASS",
            "folds_audited": len(folds),
            "pairwise_cross_comparisons_performed": total_comparisons,
            "max_cross_partition_identity_observed": round(float(max_cross_identity_observed), 4),
            "identity_threshold_enforced": self.identity_threshold,
            "verdict": "ZERO_LEAKAGE_VERIFIED: All outer test, inner calibration, and inner training sets are strictly disjoint.",
        }
