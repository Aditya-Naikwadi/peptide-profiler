"""20-D Amino Acid Composition (AAC) feature extractor."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

AAC_VERSION = "aac_v2.0"
STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def calculate_aac_dict(sequence: str, as_percentage: bool = True) -> Dict[str, float]:
    """Calculate amino acid composition as dictionary."""
    seq = sequence.strip().upper()
    total = len(seq)
    if total == 0:
        return {aa: 0.0 for aa in STANDARD_AMINO_ACIDS}

    multiplier = 100.0 if as_percentage else 1.0
    return {aa: (seq.count(aa) / total) * multiplier for aa in STANDARD_AMINO_ACIDS}


def calculate_aac_vector(sequence: str, as_percentage: bool = True) -> np.ndarray:
    """Calculate 20-D AAC numpy array in alphabetical standard order."""
    d = calculate_aac_dict(sequence, as_percentage=as_percentage)
    return np.array([d[aa] for aa in STANDARD_AMINO_ACIDS], dtype=np.float32)
