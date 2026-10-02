"""Dataset Inventory, Provenance Tracking, and Deduplication Infrastructure.

Implements Task 1 of Shared Infrastructure:
1. Inventories all datasets (antigenicity, toxicity, allergenicity).
2. Records sources, sizes, class balances, length distributions, and licenses from DATA_GOVERNANCE.
3. Deduplicates exact and near-exact sequences within and across datasets with removal audit logs.
4. Generates cryptographic SHA-256 checksums for versioned immutability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from src.data.deduplication import deduplicate_records

logger = logging.getLogger(__name__)

LENGTH_BINS: List[Tuple[str, int, int]] = [
    ("5-9", 5, 9),
    ("10-14", 10, 14),
    ("15-24", 15, 24),
    ("25-49", 25, 49),
    ("50+", 50, 100000),
]


def assign_length_bin(length: int) -> str:
    """Assign sequence length to standardized length bin."""
    for name, low, high in LENGTH_BINS:
        if low <= length <= high:
            return name
    return "50+"


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_length_distribution(lengths: List[int]) -> Dict[str, Any]:
    """Compute length statistics and length-bin breakdown."""
    arr = np.array(lengths)
    bin_counts = Counter(assign_length_bin(l) for l in lengths)

    return {
        "count": len(lengths),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(round(np.mean(arr), 2)),
        "std": float(round(np.std(arr), 2)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "bins": {
            b_name: {
                "count": bin_counts.get(b_name, 0),
                "fraction": float(round(bin_counts.get(b_name, 0) / len(lengths), 4)),
            }
            for b_name, _, _ in LENGTH_BINS
        },
    }


def inventory_and_deduplicate(
    raw_dataset_path: Path,
    sources_config_path: Path,
    near_dup_threshold: float = 0.98,
) -> Dict[str, Any]:
    """Perform comprehensive inventory and deduplication.

    Args:
        raw_dataset_path: Path to evaluation_dataset.json.
        sources_config_path: Path to config/sources.yaml.
        near_dup_threshold: Similarity threshold for near-duplicate removal (default 0.98).

    Returns:
        Dictionary containing inventory, class balance, length metrics, removal log, and hashes.
    """
    with open(raw_dataset_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    with open(sources_config_path, "r", encoding="utf-8") as f:
        sources_meta = yaml.safe_load(f)

    raw_count = len(raw_records)
    raw_lengths = [len(r["sequence"]) for r in raw_records]

    # Deduplicate exact and near-exact sequences
    deduped_records, removal_logs = deduplicate_records(
        raw_records,
        near_duplicate_threshold=near_dup_threshold,
        resolve_conflicts=True,
    )
    clean_count = len(deduped_records)
    clean_lengths = [len(r["sequence"]) for r in deduped_records]

    # Class balance across endpoints
    def get_balance(key: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos = sum(1 for r in records if r.get(key, 0) == 1)
        neg = len(records) - pos
        return {
            "positive_count": pos,
            "negative_count": neg,
            "positive_prevalence": float(round(pos / len(records), 4)),
        }

    raw_antigen_bal = get_balance("is_antigen", raw_records)
    raw_toxin_bal = get_balance("is_toxic", raw_records)
    raw_allergen_bal = get_balance("is_allergen", raw_records)

    clean_antigen_bal = get_balance("is_antigen", deduped_records)
    clean_toxin_bal = get_balance("is_toxic", deduped_records)
    clean_allergen_bal = get_balance("is_allergen", deduped_records)

    # Compute content hashes
    raw_json_str = json.dumps(raw_records, sort_keys=True)
    clean_json_str = json.dumps(deduped_records, sort_keys=True)
    raw_hash = compute_sha256(raw_json_str)
    clean_hash = compute_sha256(clean_json_str)

    # Removal log details
    removals_summary = [
        {
            "removed_id": entry.seq_id,
            "retained_duplicate_id": entry.duplicate_of_id,
            "reason": entry.removal_reason,
            "sequence_length": entry.sequence_length,
            "similarity": round(entry.similarity, 4),
        }
        for entry in removal_logs
    ]

    inventory = {
        "metadata": {
            "governance_version": sources_meta.get("version", "1.1.0"),
            "governance_policy": sources_meta.get("governance_policy", "DATA_GOVERNANCE.md"),
            "raw_dataset_sha256": raw_hash,
            "clean_dataset_sha256": clean_hash,
            "near_duplicate_threshold": near_dup_threshold,
        },
        "sources": sources_meta.get("sources", {}),
        "sizes": {
            "raw_sequences_count": raw_count,
            "clean_sequences_count": clean_count,
            "removed_duplicates_count": len(removals_summary),
        },
        "class_balance": {
            "antigenicity": {
                "raw": raw_antigen_bal,
                "clean": clean_antigen_bal,
            },
            "toxicity": {
                "raw": raw_toxin_bal,
                "clean": clean_toxin_bal,
            },
            "allergenicity": {
                "raw": raw_allergen_bal,
                "clean": clean_allergen_bal,
            },
        },
        "length_distributions": {
            "raw": compute_length_distribution(raw_lengths),
            "clean": compute_length_distribution(clean_lengths),
        },
        "removals_log": removals_summary,
    }

    return inventory, deduped_records
