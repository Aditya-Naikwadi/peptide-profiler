"""Auto-Cross Covariance (ACC) feature transformation.

Implementation of the exact Auto-Cross Covariance formulation used in VaxiJen v3.0
(Dimitrov et al., Vaccines 2020, 8:709) and AllerTOP.

Transformations:
- Auto-Covariance (AC): Measures correlation of the same physicochemical property along the sequence at lag l.
  AC(j, l) = (1 / (N - l)) * sum_{i=1}^{N - l} (z_{j, i} - z_bar_j) * (z_{j, i+l} - z_bar_j)

- Cross-Covariance (CC): Measures correlation of different properties j and k along the sequence at lag l.
  CC(j, k, l) = (1 / (N - l)) * sum_{i=1}^{N - l} (z_{j, i} - z_bar_j) * (z_{k, i+l} - z_bar_k)

Residue Scales:
- Uses Sandberg et al. (1998) 5 Z-scales:
  z1: Lipophilicity / Hydrophobicity
  z2: Steric properties / Molecular weight / Size
  z3: Polarizability / Polarity
  z4: Net charge / Isoelectric behavior
  z5: Electronegativity / H-bonding ability
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

ACC_VERSION = "acc_v2.0"

# Sandberg et al. (1998) 5 Z-scales for the 20 canonical amino acids
Z_SCALES_5D: Dict[str, List[float]] = {
    "A": [0.07, -1.73, 0.09, -1.86, -0.25],
    "C": [0.71, -0.97, 4.13, 2.02, -0.78],
    "D": [3.64, 1.13, -1.07, 0.07, 1.48],
    "E": [3.08, 0.39, -0.07, 0.88, 1.94],
    "F": [-4.92, 1.30, 0.45, 0.54, -0.18],
    "G": [2.23, -5.36, 0.30, -3.93, 0.22],
    "H": [2.41, 1.74, 1.11, -0.18, -0.11],
    "I": [-4.44, -1.68, -1.03, -0.98, -0.27],
    "K": [2.84, 1.41, -3.14, 0.21, 1.54],
    "L": [-4.19, -1.03, -0.98, -0.55, -0.17],
    "M": [-2.49, -0.27, -0.41, -0.19, -0.41],
    "N": [3.22, 1.45, 0.84, 0.07, 0.22],
    "P": [-1.22, 0.88, 2.23, -1.51, -1.41],
    "Q": [2.18, 0.53, -1.14, 1.48, 0.65],
    "R": [2.88, 2.52, -3.44, -1.13, -0.48],
    "S": [1.96, -1.63, 0.57, -1.72, 0.12],
    "T": [0.92, -2.09, -1.40, -1.67, -0.27],
    "V": [-2.69, -2.53, -1.29, -1.33, -0.32],
    "W": [-4.75, 3.65, 0.85, -0.32, -0.32],
    "Y": [-1.39, 2.32, 0.01, -0.44, 0.02],
}


def calculate_acc_features(
    sequence: str,
    max_lag: int = 5,
    include_cross: bool = True,
) -> np.ndarray:
    """Calculate Auto-Cross Covariance (ACC) descriptors over 5 Z-scales.

    Args:
        sequence: Amino acid sequence.
        max_lag: Maximum lag distance L (default 5, per VaxiJen v3.0 standard).
        include_cross: If True, includes cross-covariance terms (CC).

    Returns:
        1D numpy array of ACC descriptors.
        Dimension = (5 * max_lag) if auto-only, or (25 * max_lag) with cross terms.
    """
    seq = sequence.strip().upper()
    n = len(seq)
    n_properties = 5

    # If sequence is shorter than lag + 1, effective lag is bounded
    eff_lag = min(max_lag, max(1, n - 1)) if n > 1 else 1

    # Total dimensions expected for max_lag
    total_dims = (n_properties * max_lag) + (n_properties * (n_properties - 1) * max_lag if include_cross else 0)
    out_vector = np.zeros(total_dims, dtype=np.float32)

    if n <= 1:
        return out_vector

    # Build sequence matrix of shape (n, 5)
    mat = np.zeros((n, n_properties), dtype=np.float64)
    for i, aa in enumerate(seq):
        mat[i] = Z_SCALES_5D.get(aa, [0.0] * n_properties)

    # Mean center each property across the sequence
    means = np.mean(mat, axis=0)
    centered = mat - means

    idx = 0
    # 1. Auto-covariance (AC) for each property j and lag l in 1..max_lag
    for j in range(n_properties):
        for l in range(1, max_lag + 1):
            if l < n:
                prod = centered[: n - l, j] * centered[l:, j]
                out_vector[idx] = float(np.sum(prod) / (n - l))
            else:
                out_vector[idx] = 0.0
            idx += 1

    # 2. Cross-covariance (CC) for each pair (j, k), j != k, and lag l in 1..max_lag
    if include_cross:
        for j in range(n_properties):
            for k in range(n_properties):
                if j == k:
                    continue
                for l in range(1, max_lag + 1):
                    if l < n:
                        prod = centered[: n - l, j] * centered[l:, k]
                        out_vector[idx] = float(np.sum(prod) / (n - l))
                    else:
                        out_vector[idx] = 0.0
                    idx += 1

    return out_vector
