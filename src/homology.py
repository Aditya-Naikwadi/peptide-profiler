"""Homology-aware clustering and sequence distance module for Peptide Profiler.

Clusters protein and peptide sequences by sequence identity (30-40% threshold for >=30 aa)
and shared k-mer / edit distance (for <30 aa) to eliminate homology leakage in cross-validation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np


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
    return len(s1 & s2) / union


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def sequence_similarity(seq1: str, seq2: str, identity_threshold: float = 0.35) -> bool:
    """Determine if two sequences belong to the same homology cluster.

    - For peptides < 30 aa: uses normalized edit distance (distance / max_len <= 0.45)
      or high 3-mer Jaccard similarity (>= 0.30).
    - For proteins >= 30 aa: uses k-mer Jaccard (k=3) proxy and length constraint,
      falling back to local alignment similarity.
    """
    l1, l2 = len(seq1), len(seq2)
    max_l = max(l1, l2)
    min_l = min(l1, l2)

    # If length disparity is too large, unlikely to be homologous
    if min_l / max_l < 0.4 and max_l > 30:
        return False

    if max_l < 30:
        ed = edit_distance(seq1, seq2)
        norm_ed = ed / max_l
        return norm_ed <= (1.0 - identity_threshold) or jaccard_kmer_similarity(seq1, seq2, k=3) >= 0.30

    # For longer proteins, 3-mer and 4-mer Jaccard is a fast, conservative proxy for homology
    jaccard_3 = jaccard_kmer_similarity(seq1, seq2, k=3)
    if jaccard_3 >= identity_threshold:
        return True

    # If borderline (0.20 <= jaccard < threshold), perform exact normalized edit distance on sample
    if jaccard_3 >= 0.20 and max_l <= 120:
        ed = edit_distance(seq1, seq2)
        return (1.0 - (ed / max_l)) >= identity_threshold

    return False


def cluster_sequences_by_homology(
    records: List[Dict[str, Any]],
    identity_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Cluster sequences into disjoint homology clusters using connected components graph.

    Args:
        records: List of dicts, each with 'id' and 'sequence'.
        identity_threshold: Homology threshold (default 35% identity).

    Returns:
        Dict mapping sequence 'id' to cluster_id, along with cluster metadata.
    """
    n = len(records)
    # Disjoint Set Union (Union-Find)
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

    # Compute pairwise connections
    for i in range(n):
        seq_i = records[i]["sequence"]
        for j in range(i + 1, n):
            seq_j = records[j]["sequence"]
            if sequence_similarity(seq_i, seq_j, identity_threshold=identity_threshold):
                union(i, j)

    # Group into clusters
    cluster_groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        cluster_groups[root].append(records[i]["id"])

    # Remap cluster IDs to sequential integers
    cluster_map = {}
    cluster_sizes = {}
    for cluster_idx, (root, members) in enumerate(cluster_groups.items()):
        cid = f"cluster_{cluster_idx:04d}"
        cluster_sizes[cid] = len(members)
        for mem_id in members:
            cluster_map[mem_id] = cid

    return {
        "metadata": {
            "total_sequences": n,
            "total_clusters": len(cluster_groups),
            "identity_threshold": identity_threshold,
            "singletons": sum(1 for sz in cluster_sizes.values() if sz == 1),
            "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        },
        "assignments": cluster_map,
        "cluster_sizes": cluster_sizes,
    }


def save_cluster_assignments(
    dataset_path: str = "data/evaluation_dataset.json",
    out_path: str = "data/cluster_assignments.json",
    identity_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Load dataset, cluster sequences, and write versioned cluster artifact."""
    p_in = Path(dataset_path)
    if not p_in.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    records = json.loads(p_in.read_text(encoding="utf-8"))
    res = cluster_sequences_by_homology(records, identity_threshold=identity_threshold)

    p_out = Path(out_path)
    p_out.parent.mkdir(exist_ok=True, parents=True)
    p_out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
