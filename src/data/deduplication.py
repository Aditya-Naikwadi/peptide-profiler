"""Deduplication and conflicting label resolution for sequence datasets.

Identifies:
1. Exact duplicate sequences across and within sources.
2. Near-duplicate sequences (similarity >= identity_threshold, default 0.98).
3. Conflicting labels between duplicates, resolving via evidence hierarchy or dropping ambiguous cases.
Logs all removals with explicit reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Evidence strength hierarchy for conflicting label arbitration
EVIDENCE_STRENGTH: Dict[str, int] = {
    "experimental_challenge_protection": 100,
    "experimental_assay": 90,
    "curated_experimental": 80,
    "peer_reviewed_clinical_ige": 75,
    "curated_literature": 50,
    "curated_swissprot": 50,
    "curated_non_immunogen": 40,
    "curated_viral": 40,
    "predicted": 10,
    "unknown": 0,
}


@dataclass
class RemovalLogEntry:
    """Record of a sequence removed during curation."""
    seq_id: str
    duplicate_of_id: str
    removal_reason: str
    sequence_length: int
    similarity: float


def compute_pairwise_identity(seq1: str, seq2: str) -> float:
    """Compute simple global alignment identity for near-identical sequences."""
    l1, l2 = len(seq1), len(seq2)
    if l1 == 0 or l2 == 0:
        return 0.0
    if abs(l1 - l2) / max(l1, l2) > 0.10:
        return 0.0

    # Fast prefix/suffix match and count identical positions over overlap
    min_len = min(l1, l2)
    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    return matches / max(l1, l2)


def deduplicate_records(
    records: List[Dict[str, Any]],
    near_duplicate_threshold: float = 0.98,
    resolve_conflicts: bool = True,
) -> Tuple[List[Dict[str, Any]], List[RemovalLogEntry]]:
    """Deduplicate records, resolve label conflicts, and generate detailed removal logs.

    Args:
        records: List of sequence record dictionaries.
        near_duplicate_threshold: Pairwise identity threshold for near-duplicates (default 0.98).
        resolve_conflicts: Whether to arbitrate label conflicts by evidence hierarchy.

    Returns:
        Tuple of (clean_records, removal_logs).
    """
    cleaned_records: List[Dict[str, Any]] = []
    removal_logs: List[RemovalLogEntry] = []

    # Map sequence string to existing record index in cleaned_records
    exact_seq_map: Dict[str, int] = {}

    for rec in records:
        seq = rec["sequence"]
        seq_id = str(rec["id"])
        label = rec.get("label", rec.get("is_toxic", 0))
        evidence = rec.get("label_evidence", "curated_literature")
        evidence_score = EVIDENCE_STRENGTH.get(evidence, 10)

        # 1. Check Exact Duplication
        if seq in exact_seq_map:
            existing_idx = exact_seq_map[seq]
            existing_rec = cleaned_records[existing_idx]
            existing_id = str(existing_rec["id"])
            existing_label = existing_rec.get("label", existing_rec.get("is_toxic", 0))
            existing_evidence = existing_rec.get("label_evidence", "curated_literature")
            existing_score = EVIDENCE_STRENGTH.get(existing_evidence, 10)

            if label == existing_label:
                # Labels agree: Keep the one with higher evidence or earlier deposit
                if evidence_score > existing_score:
                    # Replace existing with new
                    removal_logs.append(RemovalLogEntry(
                        seq_id=existing_id,
                        duplicate_of_id=seq_id,
                        removal_reason="EXACT_DUPLICATE_SUPERSEDED_BY_HIGHER_EVIDENCE",
                        sequence_length=len(seq),
                        similarity=1.0,
                    ))
                    cleaned_records[existing_idx] = rec
                else:
                    removal_logs.append(RemovalLogEntry(
                        seq_id=seq_id,
                        duplicate_of_id=existing_id,
                        removal_reason="EXACT_DUPLICATE_RETAINED_PRIMARY",
                        sequence_length=len(seq),
                        similarity=1.0,
                    ))
                continue
            else:
                # Conflicting labels for identical sequence!
                if resolve_conflicts and evidence_score != existing_score:
                    # Higher evidence wins
                    if evidence_score > existing_score:
                        removal_logs.append(RemovalLogEntry(
                            seq_id=existing_id,
                            duplicate_of_id=seq_id,
                            removal_reason=f"CONFLICTING_LABEL_DISCARDED (Score {existing_score} < {evidence_score})",
                            sequence_length=len(seq),
                            similarity=1.0,
                        ))
                        cleaned_records[existing_idx] = rec
                    else:
                        removal_logs.append(RemovalLogEntry(
                            seq_id=seq_id,
                            duplicate_of_id=existing_id,
                            removal_reason=f"CONFLICTING_LABEL_DISCARDED (Score {evidence_score} < {existing_score})",
                            sequence_length=len(seq),
                            similarity=1.0,
                        ))
                    continue
                else:
                    # Equal evidence or unresolvable conflict: drop both to protect training data integrity
                    removal_logs.append(RemovalLogEntry(
                        seq_id=seq_id,
                        duplicate_of_id=existing_id,
                        removal_reason="UNRESOLVED_CONFLICTING_LABELS_DROPPED",
                        sequence_length=len(seq),
                        similarity=1.0,
                    ))
                    # Mark existing record as dropped
                    existing_rec["_drop_conflict"] = True
                    continue

        # 2. Check Near-Duplicates
        is_near_dup = False
        for existing_idx, existing_rec in enumerate(cleaned_records):
            if existing_rec.get("_drop_conflict"):
                continue
            existing_seq = existing_rec["sequence"]
            sim = compute_pairwise_identity(seq, existing_seq)
            if sim >= near_duplicate_threshold:
                existing_id = str(existing_rec["id"])
                removal_logs.append(RemovalLogEntry(
                    seq_id=seq_id,
                    duplicate_of_id=existing_id,
                    removal_reason=f"NEAR_DUPLICATE (identity {sim:.1%} >= {near_duplicate_threshold:.1%})",
                    sequence_length=len(seq),
                    similarity=round(sim, 4),
                ))
                is_near_dup = True
                break

        if not is_near_dup:
            exact_seq_map[seq] = len(cleaned_records)
            cleaned_records.append(rec)

    # Filter out any records that were marked dropped due to unresolvable conflicts
    final_records = [r for r in cleaned_records if not r.pop("_drop_conflict", False)]
    return final_records, removal_logs
