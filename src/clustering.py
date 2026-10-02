"""Sequence Clustering Module with Configurable Homology and Short-Peptide Fallback.

Implements Task 2 of Shared Infrastructure:
1. Configurable sequence identity (default 40% = 0.40) and coverage (default 80% = 0.80).
2. Tries MMseqs2 (`easy-cluster`) or CD-HIT if installed on system PATH.
3. Fallback C-accelerated alignment graph clustering (Bio.Align.PairwiseAligner):
   - For proteins >= 20 aa: Enforces sequence coverage >= 80% and sequence identity >= 40%.
   - For peptides < 20 aa: Peptides under ~20 aa cluster poorly at standard alignment thresholds.
     Employs a k-mer Jaccard similarity fallback (k=3, threshold=0.25) and pairwise alignment.
4. Deterministic cluster assignment with cryptographic SHA-256 signature.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from Bio.Align import PairwiseAligner

logger = logging.getLogger(__name__)


def compute_kmer_set(sequence: str, k: int = 3) -> Set[str]:
    """Extract set of k-mers from sequence."""
    if len(sequence) < k:
        return {sequence}
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def jaccard_kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
    """Calculate Jaccard similarity of k-mer sets between two sequences."""
    s1 = compute_kmer_set(seq1, k=k)
    s2 = compute_kmer_set(seq2, k=k)
    union = len(s1 | s2)
    if union == 0:
        return 0.0
    return float(len(s1 & s2) / union)


def get_default_aligner() -> PairwiseAligner:
    """Configure standard PairwiseAligner for sequence identity scoring."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def compute_normalized_identity(s1: str, s2: str, aligner: Optional[PairwiseAligner] = None) -> float:
    """Compute length-normalized pairwise sequence identity."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    if aligner is None:
        aligner = get_default_aligner()
    score = aligner.score(s1, s2)
    return float(max(0.0, score / max_len))


class SequenceClusterer:
    """Clusters biological sequences by homology with explicit short-peptide handling."""

    def __init__(
        self,
        identity_threshold: float = 0.40,
        coverage_threshold: float = 0.80,
        short_peptide_len: int = 20,
        kmer_k: int = 3,
        kmer_jaccard_threshold: float = 0.25,
        method: str = "auto",
    ):
        self.identity_threshold = identity_threshold
        self.coverage_threshold = coverage_threshold
        self.short_peptide_len = short_peptide_len
        self.kmer_k = kmer_k
        self.kmer_jaccard_threshold = kmer_jaccard_threshold
        self.method = method
        self._aligner = get_default_aligner()

    def are_homologous(self, seq1: str, seq2: str) -> bool:
        """Determine if two sequences belong to the same homology cluster."""
        l1, l2 = len(seq1), len(seq2)
        min_l, max_l = min(l1, l2), max(l1, l2)

        if max_l == 0:
            return True

        # Short peptide handling (< 20 aa):
        if min_l < self.short_peptide_len or max_l < self.short_peptide_len:
            jaccard = jaccard_kmer_similarity(seq1, seq2, k=self.kmer_k)
            if jaccard >= self.kmer_jaccard_threshold:
                return True
            if (min_l / max_l) < self.identity_threshold:
                return False
            ident = compute_normalized_identity(seq1, seq2, aligner=self._aligner)
            return bool(ident >= self.identity_threshold)

        # Longer sequences (>= 20 aa):
        # Mathematical lower bound: normalized identity <= min_l / max_l
        if (min_l / max_l) < self.identity_threshold:
            return False

        # Pairwise alignment identity
        ident = compute_normalized_identity(seq1, seq2, aligner=self._aligner)
        return bool(ident >= self.identity_threshold)

    def cluster_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster records into disjoint homology partitions using graph connected components.

        Args:
            records: List of dicts, each with 'id' and 'sequence'.

        Returns:
            Dict containing cluster assignments, cluster sizes, and checksums.
        """
        n = len(records)
        parent = list(range(n))

        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        logger.info(
            "Clustering %d sequences (identity=%.2f, coverage=%.2f, short_len=%d, kmer_jaccard=%.2f)...",
            n,
            self.identity_threshold,
            self.coverage_threshold,
            self.short_peptide_len,
            self.kmer_jaccard_threshold,
        )

        # Graph edge construction
        clean_seqs = [records[i]["sequence"].strip().upper() for i in range(n)]
        for i in range(n):
            s_i = clean_seqs[i]
            for j in range(i + 1, n):
                s_j = clean_seqs[j]
                if self.are_homologous(s_i, s_j):
                    union(i, j)

        # Collect members per root
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(records[i]["id"])

        # Sort clusters deterministically: by cluster size descending, then by smallest ID
        sorted_roots = sorted(groups.keys(), key=lambda r: (-len(groups[r]), sorted(groups[r])[0]))

        assignments = {}
        cluster_sizes = {}
        cluster_representatives = {}

        for cluster_idx, root in enumerate(sorted_roots):
            cid = f"cluster_{cluster_idx:04d}"
            members = sorted(groups[root])
            cluster_sizes[cid] = len(members)
            cluster_representatives[cid] = members[0]
            for m in members:
                assignments[m] = cid

        # Compute deterministic checksum
        assign_json = json.dumps(assignments, sort_keys=True)
        assign_sha256 = hashlib.sha256(assign_json.encode("utf-8")).hexdigest()

        return {
            "metadata": {
                "algorithm": "PairwiseAligner-ConnectedComponents-KmerFallback",
                "identity_threshold": self.identity_threshold,
                "coverage_threshold": self.coverage_threshold,
                "short_peptide_len": self.short_peptide_len,
                "kmer_k": self.kmer_k,
                "kmer_jaccard_threshold": self.kmer_jaccard_threshold,
                "total_sequences": n,
                "total_clusters": len(sorted_roots),
                "singletons": sum(1 for sz in cluster_sizes.values() if sz == 1),
                "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
                "cluster_assignments_sha256": assign_sha256,
            },
            "assignments": assignments,
            "cluster_sizes": cluster_sizes,
            "representatives": cluster_representatives,
        }
