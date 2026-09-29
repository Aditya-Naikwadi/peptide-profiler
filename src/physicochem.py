"""Physicochemical characterization module using Biopython ProtParam and Pfeature."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Standard 20 amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Physicochemical property residue groupings (Pfeature definitions)
# 1. Positively charged (K, R, H)
RES_PC = set("KRH")
# 2. Negatively charged (D, E)
RES_NC = set("DE")
# 3. Neutral (A, C, F, G, I, L, M, N, P, Q, S, T, V, W, Y)
RES_NE = set("ACFGILMNPQSTVWY")
# 4. Polar (C, D, E, H, K, N, Q, R, S, T, W, Y)
RES_PO = set("CDEHKNQRSTWY")
# 5. Non-polar (A, F, G, I, L, M, P, V)
RES_NP = set("AFGILMPV")
# 6. Aliphatic (I, L, V, A)
RES_AL = set("ILVA")
# 7. Cyclic (P)
RES_CY = set("P")
# 8. Aromatic (F, H, W, Y)
RES_AR = set("FHWY")
# 9. Acidic (D, E)
RES_AC = set("DE")
# 10. Basic (K, R, H)
RES_BS = set("KRH")
# 11. Neutral at pH 7 (A, C, F, G, I, L, M, N, P, Q, S, T, V, W, Y)
RES_NE_PH = set("ACFGILMNPQSTVWY")
# 12. Hydrogen bond donor/acceptor (N, Q, S, T, Y, C, W, H, D, E, K, R)
RES_HB = set("NQSTYCW HDEKR".replace(" ", ""))
# 13. Hydrophilic (D, E, K, R, H, N, Q)
RES_HL = set("DEKRHNQ")
# 14. Neutral hydropathy (A, C, G, P, S, T, W, Y)
RES_NT = set("ACGPSTWY")
# 15. Hydrophobic (F, I, L, M, V)
RES_HX = set("FILMV")
# 16. Small (A, C, D, G, N, P, S, T, V)
RES_SM = set("ACDGNPSTV")
# 17. Large (E, F, H, I, K, L, M, Q, R, W, Y)
RES_LR = set("EFHIKLMQRWY")
# 18. Tiny (A, C, G, S)
RES_TN = set("ACGS")
# Secondary structure propensities
# Helix formers (E, A, L, M, Q, K, R, H)
RES_SS_HE = set("EALMQKRH")
# Strand formers (V, I, T, Y, F, W, C)
RES_SS_ST = set("VITYFWC")
# Coil / Turn formers (G, N, P, D, S)
RES_SS_CO = set("GNPDS")
# Solvent accessibility
# Buried (A, C, F, I, L, M, V, W)
RES_SA_BU = set("ACFILMVW")
# Exposed (D, E, H, K, N, Q, R)
RES_SA_EX = set("DEHKNQR")
# Intermediate (G, P, S, T, Y)
RES_SA_IN = set("GPSTY")

# Sandberg et al. 5 Z-scales for amino acids (Lipophilicity, Steric bulk, Polarity, etc.)
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


def calculate_aac(sequence: str) -> Dict[str, float]:
    """Calculate Amino Acid Composition (AAC) percentages for 20 standard amino acids."""
    length = len(sequence)
    if length == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: round((sequence.count(aa) / length) * 100.0, 3) for aa in AMINO_ACIDS}


def calculate_pcp_descriptors(sequence: str) -> Dict[str, float]:
    """Calculate 30 Physicochemical Property (PCP) descriptors matching Pfeature definitions."""
    length = len(sequence)
    if length == 0:
        return {}

    def frac(subset: set) -> float:
        return round(sum(1 for aa in sequence if aa in subset) / length, 3)

    # Calculate average Z-scales over the sequence
    z_sums = [0.0, 0.0, 0.0, 0.0, 0.0]
    for aa in sequence:
        z = Z_SCALES.get(aa, [0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(5):
            z_sums[i] += z[i]
    z_avgs = [round(z_s / length, 3) for z_s in z_sums]

    pcp = {
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
        "PCP_SC": 0.0,  # Specific charges
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
    return pcp


def calculate_autocorrelation(sequence: str, max_lag: int = 3) -> Dict[str, float]:
    """Calculate Moreau-Broto autocorrelation descriptors based on hydrophobicity and charge."""
    # Normalized Kyte-Doolittle hydrophobicity scale
    kd = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
    }
    n = len(sequence)
    acr_dict: Dict[str, float] = {}

    for lag in range(1, max_lag + 1):
        if n > lag:
            val = sum(kd.get(sequence[i], 0.0) * kd.get(sequence[i + lag], 0.0) for i in range(n - lag))
            acr_dict[f"ACR_lag_{lag}"] = round(val / (n - lag), 3)
        else:
            acr_dict[f"ACR_lag_{lag}"] = 0.0

    return acr_dict


def run_physicochemical(sequence: str) -> Dict[str, Any]:
    """Calculate comprehensive physicochemical properties for an amino acid sequence.

    Uses Biopython ProtParam for fundamental physical constants and Pfeature
    algorithms for AAC, PCP, and ACR vector descriptors.

    Args:
        sequence: Validated uppercase amino acid sequence string.

    Returns:
        Dictionary adhering to the project interface:
            - mol_weight: Molecular weight in Daltons (g/mol)
            - gravy: Grand average of hydropathicity (positive = hydrophobic)
            - instability_index: Instability index (< 40 is stable)
            - is_stable: Boolean classification based on instability index
            - isoelectric_point: Theoretical isoelectric point (pI)
            - charge_at_pH7: Net electrical charge at physiological pH 7.0
            - aromaticity: Relative frequency of aromatic residues (F, W, Y)
            - secondary_structure: Fraction of helix, turn, and sheet
            - aac: Dict of 20 amino acid composition percentages
            - pcp_dict: Dict of 30 Pfeature physicochemical property descriptors
            - pcp_vector: List of 30 numerical values for ML pipelines
            - acr_dict: Autocorrelation descriptors
    """
    cleaned = sequence.strip().upper()
    analyser = ProteinAnalysis(cleaned)

    # ProtParam calculations
    mol_weight = round(analyser.molecular_weight(), 2)
    gravy = round(analyser.gravy(), 3)
    instability = round(analyser.instability_index(), 2)
    pI = round(analyser.isoelectric_point(), 2)
    charge_7 = round(analyser.charge_at_pH(7.0), 2)
    aromaticity = round(analyser.aromaticity(), 3)

    # Helix, Turn, Sheet fractions
    sec_struct = analyser.secondary_structure_fraction()
    sec_struct_dict = {
        "helix": round(sec_struct[0], 3),
        "turn": round(sec_struct[1], 3),
        "sheet": round(sec_struct[2], 3),
    }

    # Pfeature composition & descriptors
    aac = calculate_aac(cleaned)
    pcp = calculate_pcp_descriptors(cleaned)
    pcp_vector = list(pcp.values())
    acr = calculate_autocorrelation(cleaned, max_lag=3)

    return {
        "mol_weight": mol_weight,
        "gravy": gravy,
        "instability_index": instability,
        "is_stable": bool(instability < 40.0),
        "isoelectric_point": pI,
        "charge_at_pH7": charge_7,
        "aromaticity": aromaticity,
        "secondary_structure": sec_struct_dict,
        "aac": aac,
        "pcp_dict": pcp,
        "pcp_vector": pcp_vector,
        "acr_dict": acr,
    }
