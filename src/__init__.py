"""Peptide Profiler Package.

A production-grade, offline immunological and safety characterization pipeline
for vaccine design and peptide-therapeutic discovery.
"""

from __future__ import annotations

from src.aggregator import profile_multiple_sequences, profile_sequence
from src.allergenicity import run_allergenicity
from src.antigenicity import run_antigenicity
from src.api import profile_batch, profile_peptide, serialize_to_standard_contract
from src.parser import parse_fasta
from src.physicochem import run_physicochemical
from src.toxicity import run_toxicity

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "profile_peptide",
    "profile_batch",
    "serialize_to_standard_contract",
    "profile_sequence",
    "profile_multiple_sequences",
    "run_antigenicity",
    "run_toxicity",
    "run_allergenicity",
    "run_physicochemical",
    "parse_fasta",
]
