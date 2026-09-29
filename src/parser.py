"""FASTA parsing and sequence validation module for Peptide Profiler."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

# Standard 20 proteinogenic amino acids
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class SequenceValidationError(ValueError):
    """Raised when an input sequence fails validation."""
    pass


def clean_sequence(seq: str, sanitize: bool = False) -> str:
    """Clean and validate an amino acid sequence string.

    Args:
        seq: Raw amino acid sequence string.
        sanitize: If True, replaces non-standard characters with closest standard or strips them.
                 If False, raises SequenceValidationError on invalid characters.

    Returns:
        Cleaned, uppercase amino acid sequence string.

    Raises:
        SequenceValidationError: If sequence is empty or contains invalid characters.
    """
    cleaned = re.sub(r"\s+", "", seq).upper()
    if not cleaned:
        raise SequenceValidationError("Sequence cannot be empty.")

    invalid_chars = set(cleaned) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        if sanitize:
            # Common conversions: B -> D/N, Z -> E/Q, U -> C (selenocysteine), X -> A (neutral alanine)
            replacements = {"B": "D", "Z": "E", "U": "C", "X": "A", "J": "L", "O": "K"}
            sanitized = []
            for char in cleaned:
                if char in STANDARD_AMINO_ACIDS:
                    sanitized.append(char)
                elif char in replacements:
                    sanitized.append(replacements[char])
                else:
                    # Drop unknown symbols
                    pass
            cleaned = "".join(sanitized)
            if not cleaned:
                raise SequenceValidationError(
                    f"Sequence contains only invalid characters: {invalid_chars}"
                )
        else:
            raise SequenceValidationError(
                f"Invalid amino acid character(s) found: {sorted(list(invalid_chars))}. "
                f"Only standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) are accepted. "
                f"Enable sanitize=True to automatically convert/strip ambiguous residues."
            )

    return cleaned


def parse_fasta(
    source: Union[str, Path, io.StringIO, io.TextIOBase],
    is_content: bool = False,
    sanitize: bool = False,
    min_length: int = 4,
) -> List[Dict[str, Union[str, int]]]:
    """Parse a FASTA file or string content into structured sequence records.

    Args:
        source: File path, Path object, or raw FASTA string/stream.
        is_content: If True, source is treated as raw FASTA text string rather than file path.
        sanitize: Whether to sanitize non-standard amino acid residues.
        min_length: Minimum sequence length required (default 4).

    Returns:
        List of dictionaries with keys:
            - 'id': Sequence identifier (first word after '>')
            - 'description': Full header line text
            - 'sequence': Cleaned amino acid sequence
            - 'length': Length of sequence

    Raises:
        SequenceValidationError: If parsing fails or sequences are invalid.
        FileNotFoundError: If source file path does not exist.
    """
    records: List[Dict[str, Union[str, int]]] = []

    if isinstance(source, (io.StringIO, io.TextIOBase)):
        content = source.read()
    elif is_content or (isinstance(source, str) and (source.startswith(">") or "\n" in source)):
        content = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {path}")
        content = path.read_text(encoding="utf-8")

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise SequenceValidationError("FASTA content is empty.")

    current_id: Optional[str] = None
    current_desc: Optional[str] = None
    current_seq_parts: List[str] = []

    def commit_record():
        nonlocal current_id, current_desc, current_seq_parts
        if current_id is not None:
            raw_seq = "".join(current_seq_parts)
            try:
                seq = clean_sequence(raw_seq, sanitize=sanitize)
            except SequenceValidationError as e:
                raise SequenceValidationError(f"Error in record '{current_id}': {e}") from e

            if len(seq) < min_length:
                raise SequenceValidationError(
                    f"Record '{current_id}' sequence length ({len(seq)}) is below "
                    f"minimum requirement ({min_length})."
                )

            records.append({
                "id": current_id,
                "description": current_desc or current_id,
                "sequence": seq,
                "length": len(seq),
            })
            current_id = None
            current_desc = None
            current_seq_parts = []

    # Handle plain single sequence with no header
    if not lines[0].startswith(">"):
        raw_seq = "".join(lines)
        seq = clean_sequence(raw_seq, sanitize=sanitize)
        return [{
            "id": "seq_1",
            "description": "seq_1",
            "sequence": seq,
            "length": len(seq),
        }]

    for line in lines:
        if line.startswith(">"):
            commit_record()
            header = line[1:].strip()
            parts = header.split(maxsplit=1)
            current_id = parts[0] if parts else f"seq_{len(records) + 1}"
            current_desc = header
        else:
            current_seq_parts.append(line)

    commit_record()

    if not records:
        raise SequenceValidationError("No valid sequences found in FASTA input.")

    return records


def write_fasta(records: List[Dict[str, Union[str, int]]], output_path: Union[str, Path]) -> Path:
    """Write records to a FASTA file.

    Args:
        records: List of sequence record dicts containing 'id' and 'sequence'.
        output_path: Path to output file.

    Returns:
        Path to written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            seq_id = rec.get("id", "seq")
            seq = rec["sequence"]
            f.write(f">{seq_id}\n{seq}\n")
    return out
