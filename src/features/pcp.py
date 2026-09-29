"""30-D Physicochemical Properties (Pfeature standard) feature extractor.

Matches the exact definition used across Peptide Profiler:
- 25 group fractions in [0, 1]
- 5 average Z-scale values (Z1 to Z5)
Total: 30 dimensions.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

PCP_VERSION = "pcp_v2.0"

# Pfeature Residue Groupings
RES_PC = set("KRH")
RES_NC = set("DE")
RES_NE = set("ACFGILMNPQSTVWY")
RES_PO = set("CDEHKNQRSTWY")
RES_NP = set("AFGILMPV")
RES_AL = set("ILVA")
RES_CY = set("P")
RES_AR = set("FHWY")
RES_AC = set("DE")
RES_BS = set("KRH")
RES_NE_PH = set("ACFGILMNPQSTVWY")
RES_HB = set("NQSTYCW HDEKR".replace(" ", ""))
RES_HL = set("DEKRHNQ")
RES_NT = set("ACGPSTWY")
RES_HX = set("FILMV")
RES_SM = set("ACDGNPSTV")
RES_LR = set("EFHIKLMQRWY")
RES_TN = set("ACGS")
RES_SS_HE = set("EALMQKRH")
RES_SS_ST = set("VITYFWC")
RES_SS_CO = set("GNPDS")
RES_SA_BU = set("ACFILMVW")
RES_SA_EX = set("DEHKNQR")
RES_SA_IN = set("GPSTY")

# Sandberg 5 Z-scales
Z_SCALES = {
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
    "M": [-2.49, -0.27, -0.41, 0.04, -0.22],
    "N": [3.22, 1.45, 0.84, 0.81, 0.99],
    "P": [-1.22, 0.88, 2.23, -1.51, -1.01],
    "Q": [2.18, 0.53, -1.14, 0.79, 1.53],
    "R": [2.88, 2.52, -3.44, -0.20, 1.13],
    "S": [1.96, -1.63, 0.57, -1.60, 0.27],
    "T": [0.92, -2.09, -1.40, -1.16, -0.17],
    "V": [-2.69, -2.53, -1.29, -1.49, -0.39],
    "W": [-4.75, 3.65, 0.85, 0.65, -0.19],
    "Y": [-1.39, 2.32, 0.01, 0.38, -0.83],
}


def calculate_pcp_dict(sequence: str) -> Dict[str, float]:
    """Calculate 30 Physicochemical Property descriptors matching Pfeature definitions."""
    seq = sequence.strip().upper()
    length = len(seq)
    if length == 0:
        return {}

    def frac(subset: set) -> float:
        return round(sum(1 for aa in seq if aa in subset) / length, 3)

    # Average Z-scales over the sequence
    z_sums = [0.0, 0.0, 0.0, 0.0, 0.0]
    for aa in seq:
        z = Z_SCALES.get(aa, [0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(5):
            z_sums[i] += z[i]
    z_avgs = [round(z_s / length, 3) for z_s in z_sums]

    return {
        "PCP_PC": frac(RES_PC),
        "PCP_NC": frac(RES_NC),
        "PCP_NE": frac(RES_NE),
        "PCP_PO": frac(RES_PO),
        "PCP_NP": frac(RES_NP),
        "PCP_AL": frac(RES_AL),
        "PCP_CY": frac(RES_CY),
        "PCP_AR": frac(RES_AR),
        "PCP_AC": frac(RES_AC),
        "PCP_BS": frac(RES_BS),
        "PCP_NE_pH": frac(RES_NE_PH),
        "PCP_HB": frac(RES_HB),
        "PCP_HL": frac(RES_HL),
        "PCP_NT": frac(RES_NT),
        "PCP_HX": frac(RES_HX),
        "PCP_SC": 0.0,
        "PCP_SS_HE": frac(RES_SS_HE),
        "PCP_SS_ST": frac(RES_SS_ST),
        "PCP_SS_CO": frac(RES_SS_CO),
        "PCP_SA_BU": frac(RES_SA_BU),
        "PCP_SA_EX": frac(RES_SA_EX),
        "PCP_SA_IN": frac(RES_SA_IN),
        "PCP_TN": frac(RES_TN),
        "PCP_SM": frac(RES_SM),
        "PCP_LR": frac(RES_LR),
        "PCP_Z1": z_avgs[0],
        "PCP_Z2": z_avgs[1],
        "PCP_Z3": z_avgs[2],
        "PCP_Z4": z_avgs[3],
        "PCP_Z5": z_avgs[4],
    }


def calculate_pcp_vector(sequence: str) -> np.ndarray:
    """Calculate 30-D PCP numpy vector."""
    d = calculate_pcp_dict(sequence)
    return np.array(list(d.values()), dtype=np.float32)
