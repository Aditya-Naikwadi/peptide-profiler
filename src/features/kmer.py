"""K-mer composition feature extractor."""

from __future__ import annotations

from typing import Dict, List, Set
import numpy as np

KMER_VERSION = "kmer_v2.0"


def generate_all_kmers(alphabet: List[str], k: int) -> List[str]:
    """Generate all possible k-mers from given alphabet."""
    if k == 1:
        return alphabet
    sub = generate_all_kmers(alphabet, k - 1)
    return [a + s for a in alphabet for s in sub]


def calculate_kmer_features(
    sequence: str,
    k: int = 2,
    as_frequency: bool = True,
) -> np.ndarray:
    """Calculate k-mer composition frequency vector.

    For k=1: 20 dimensions (AAC equivalent).
    For k=2: 400 dimensions (DPC equivalent).
    """
    seq = sequence.strip().upper()
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    all_kmers = generate_all_kmers(alphabet, k)
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}

    counts = np.zeros(len(all_kmers), dtype=np.float32)
    n = len(seq)
    total_kmers = n - k + 1

    if total_kmers <= 0:
        return counts

    for i in range(total_kmers):
        km = seq[i : i + k]
        if km in kmer_to_idx:
            counts[kmer_to_idx[km]] += 1.0

    if as_frequency and total_kmers > 0:
        counts = counts / total_kmers

    return counts
