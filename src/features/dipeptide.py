"""400-D Dipeptide Composition (DPC) feature extractor."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

DPC_VERSION = "dpc_v2.0"
STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDE_PAIRS: List[str] = [a + b for a in STANDARD_AMINO_ACIDS for b in STANDARD_AMINO_ACIDS]


def calculate_dpc_dict(sequence: str, as_percentage: bool = True) -> Dict[str, float]:
    """Calculate 400-D dipeptide composition as dictionary."""
    seq = sequence.strip().upper()
    total = len(seq) - 1
    if total <= 0:
        return {dp: 0.0 for dp in DIPEPTIDE_PAIRS}

    # Count occurrences
    counts = {dp: 0 for dp in DIPEPTIDE_PAIRS}
    for i in range(total):
        pair = seq[i : i + 2]
        if pair in counts:
            counts[pair] += 1

    multiplier = 100.0 if as_percentage else 1.0
    return {dp: (counts[dp] / total) * multiplier for dp in DIPEPTIDE_PAIRS}


def calculate_dpc_vector(sequence: str, as_percentage: bool = True) -> np.ndarray:
    """Calculate 400-D DPC numpy array in standardized alphabetical order."""
    d = calculate_dpc_dict(sequence, as_percentage=as_percentage)
    return np.array([d[dp] for dp in DIPEPTIDE_PAIRS], dtype=np.float32)
