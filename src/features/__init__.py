"""Feature extraction module for Peptide Profiler.

Includes versioned extractors:
- AAC: 20-D Amino Acid Composition
- DPC: 400-D Dipeptide Composition
- K-mer: K-mer composition
- ACC: Auto-Cross Covariance with documented lags (VaxiJen v3.0 standard)
- PCP: 30-D Physicochemical Properties (Pfeature standard)
"""

from __future__ import annotations

from src.features.aac import calculate_aac_vector
from src.features.acc import calculate_acc_features
from src.features.dipeptide import calculate_dpc_vector
from src.features.kmer import calculate_kmer_features
from src.features.pcp import calculate_pcp_vector

FEATURE_EXTRACTOR_VERSION = "2.0.0"

__all__ = [
    "FEATURE_EXTRACTOR_VERSION",
    "calculate_aac_vector",
    "calculate_dpc_vector",
    "calculate_kmer_features",
    "calculate_acc_features",
    "calculate_pcp_vector",
]
