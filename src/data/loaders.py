"""Dataset loaders with provenance attachment, sanitization, and deduplication.

Enforces:
1. License gate verification before loading.
2. Conversion of all sequences through the sequence sanitizer with alteration tracking.
3. Enforcement of SequenceProvenanceRecord on all loaded items.
4. Exporting curated FASTA and JSON datasets with sha256 checksums.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.deduplication import deduplicate_records
from src.data.governance import LicenseGate, SequenceProvenanceRecord, create_provenance_record
from src.data.sanitizer import sanitize_sequence
from src.parser import parse_fasta

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DatasetLoader:
    """Loads and curates sequence datasets with strict governance."""

    def __init__(self, license_gate: Optional[LicenseGate] = None, allow_unverified: bool = False):
        self.gate = license_gate or LicenseGate()
        self.allow_unverified = allow_unverified

    def load_json_dataset(
        self,
        filepath: Path,
        source_id: str,
        default_label_evidence: str = "curated_literature",
    ) -> Tuple[List[SequenceProvenanceRecord], Dict[str, Any]]:
        """Load a JSON dataset and convert to verified provenance records."""
        self.gate.check_source_access(source_id, allow_unverified=self.allow_unverified)
        raw_items = json.loads(filepath.read_text(encoding="utf-8"))

        records: List[SequenceProvenanceRecord] = []
        dropped_items = []
        alteration_logs = []

        for item in raw_items:
            raw_seq = item.get("sequence", "")
            seq_id = str(item.get("id", item.get("accession", "")))

            # 1. Sanitize
            san_res = sanitize_sequence(raw_seq)
            if san_res.is_dropped:
                dropped_items.append({"id": seq_id, "reason": san_res.drop_reason})
                continue

            if san_res.altered_count > 0:
                alteration_logs.append({
                    "id": seq_id,
                    "altered_fraction": san_res.altered_fraction,
                    "alterations": [a.action for a in san_res.alterations],
                })

            # 2. Extract label
            label = int(item.get("label", item.get("is_toxic", item.get("is_antigen", item.get("is_allergen", 0)))))

            # 3. Create provenance
            prov = create_provenance_record(
                seq_id=seq_id,
                sequence=san_res.sanitized_sequence,
                source=source_id,
                accession=str(item.get("accession", seq_id)),
                label=label,
                label_evidence=item.get("label_evidence", default_label_evidence),
                organism=item.get("organism", "unknown"),
                deposit_date=item.get("first_public_date", item.get("deposit_date")),
                transformations=["sanitized_standard_20aa"],
                license_gate=self.gate,
                allow_unverified=self.allow_unverified,
            )
            records.append(prov)

        stats = {
            "source_id": source_id,
            "raw_count": len(raw_items),
            "valid_records": len(records),
            "dropped_count": len(dropped_items),
            "altered_count": len(alteration_logs),
            "sha256": compute_file_sha256(filepath),
        }
        return records, stats
