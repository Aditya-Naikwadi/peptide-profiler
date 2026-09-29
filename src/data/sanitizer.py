"""Sequence sanitizer with detailed residue alteration logging and quality gating.

Enforces:
1. Only the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY).
2. Explicit logging of every altered residue (position, original, converted).
3. Quality flag when altered fraction exceeds max_altered_fraction (Decision D6).
4. Minimum length filtering with documented drop reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")

# Deterministic ambiguous residue mapping
AMBIGUOUS_MAP: Dict[str, str] = {
    "B": "N",  # Asx -> Asparagine
    "Z": "Q",  # Glx -> Glutamine
    "J": "L",  # Xle -> Leucine
    "U": "C",  # Sec (Selenocysteine) -> Cysteine
    "O": "K",  # Pyl (Pyrrolysine) -> Lysine
}


@dataclass
class AlterationRecord:
    """Record of a single residue alteration during sanitization."""
    position: int
    original: str
    converted: Optional[str]
    action: str  # 'mapped', 'stripped', 'case_converted'


@dataclass
class SanitizationResult:
    """Result of sequence sanitization including alteration statistics and flags."""
    original_sequence: str
    sanitized_sequence: str
    original_length: int
    sanitized_length: int
    alterations: List[AlterationRecord] = field(default_factory=list)
    altered_count: int = 0
    altered_fraction: float = 0.0
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    is_dropped: bool = False
    drop_reason: Optional[str] = None


def sanitize_sequence(
    raw_sequence: str,
    max_altered_fraction: float = 0.05,
    min_length: int = 4,
    strip_unknown: bool = True,
) -> SanitizationResult:
    """Sanitize a raw amino acid sequence with full alteration provenance.

    Args:
        raw_sequence: Input sequence string.
        max_altered_fraction: Configurable threshold above which sequence is flagged (default 0.05).
        min_length: Minimum allowed length; shorter sequences are marked dropped (default 4).
        strip_unknown: If True, unmapped characters (e.g. X, *, numbers) are stripped.

    Returns:
        SanitizationResult containing clean sequence and alteration logs.
    """
    cleaned_input = raw_sequence.strip()
    original_length = len(cleaned_input)

    if original_length == 0:
        return SanitizationResult(
            original_sequence=raw_sequence,
            sanitized_sequence="",
            original_length=0,
            sanitized_length=0,
            is_dropped=True,
            drop_reason="EMPTY_SEQUENCE",
        )

    sanitized_chars: List[str] = []
    alterations: List[AlterationRecord] = []

    for idx, char in enumerate(cleaned_input):
        upper_char = char.upper()

        if char != upper_char and upper_char in STANDARD_AMINO_ACIDS:
            alterations.append(AlterationRecord(position=idx, original=char, converted=upper_char, action="case_converted"))
            sanitized_chars.append(upper_char)
        elif upper_char in STANDARD_AMINO_ACIDS:
            sanitized_chars.append(upper_char)
        elif upper_char in AMBIGUOUS_MAP:
            converted = AMBIGUOUS_MAP[upper_char]
            alterations.append(AlterationRecord(position=idx, original=char, converted=converted, action="mapped"))
            sanitized_chars.append(converted)
        else:
            # Unrecognized/non-amino acid character (e.g. X, *, digits)
            if strip_unknown:
                alterations.append(AlterationRecord(position=idx, original=char, converted=None, action="stripped"))
            else:
                alterations.append(AlterationRecord(position=idx, original=char, converted="X", action="retained_as_unknown"))
                sanitized_chars.append("X")

    sanitized_seq = "".join(sanitized_chars)
    sanitized_length = len(sanitized_seq)

    # Exclude trivial case conversions from alteration fraction calculation
    substantive_alterations = [a for a in alterations if a.action in ("mapped", "stripped")]
    altered_count = len(substantive_alterations)
    altered_fraction = (altered_count / original_length) if original_length > 0 else 0.0

    is_flagged = altered_fraction > max_altered_fraction
    flag_reason = f"HIGH_ALTERATION_FRACTION: {altered_fraction:.1%} > {max_altered_fraction:.1%}" if is_flagged else None

    is_dropped = False
    drop_reason = None

    if sanitized_length < min_length:
        is_dropped = True
        drop_reason = f"FRAGMENT_BELOW_MIN_LENGTH: {sanitized_length} < {min_length}"

    return SanitizationResult(
        original_sequence=raw_sequence,
        sanitized_sequence=sanitized_seq,
        original_length=original_length,
        sanitized_length=sanitized_length,
        alterations=alterations,
        altered_count=altered_count,
        altered_fraction=round(altered_fraction, 4),
        is_flagged=is_flagged,
        flag_reason=flag_reason,
        is_dropped=is_dropped,
        drop_reason=drop_reason,
    )
