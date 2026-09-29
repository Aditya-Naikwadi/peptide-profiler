"""Data governance, provenance tracking, and license gate enforcement.

Enforces:
1. Strict Provenance Schema for all ingested peptide sequences.
2. License Gate: blocks unverified or restricted datasets unless explicitly
   authorized by owner approval. Prevents committing/redistributing restricted data.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
SOURCES_CONFIG_PATH = CONFIG_DIR / "sources.yaml"


class LicenseGateError(PermissionError):
    """Raised when an unapproved or unverified dataset source fails the license gate."""
    pass


class ProvenanceValidationError(ValueError):
    """Raised when a sequence record fails provenance schema validation."""
    pass


@dataclass
class SequenceProvenanceRecord:
    """Provenance record attached to every ingested peptide/protein sequence."""
    seq_id: str
    sequence: str
    source: str
    accession: str
    label: int  # 0 or 1
    label_evidence: str
    license: str
    ingest_date: str
    organism: str = "unknown"
    deposit_date: Optional[str] = None
    transformations: List[str] = field(default_factory=list)
    removal_reason: Optional[str] = None

    def validate(self) -> None:
        """Validate record completeness and constraints."""
        if not self.seq_id or not isinstance(self.seq_id, str):
            raise ProvenanceValidationError("seq_id must be a non-empty string.")
        if not self.sequence or not isinstance(self.sequence, str):
            raise ProvenanceValidationError("sequence must be a non-empty string.")
        if not self.source:
            raise ProvenanceValidationError("source must be specified.")
        if self.label not in (0, 1):
            raise ProvenanceValidationError(f"label must be 0 or 1, got {self.label}.")
        if not self.license:
            raise ProvenanceValidationError("license must be specified.")
        if not self.label_evidence:
            raise ProvenanceValidationError("label_evidence must be specified.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        self.validate()
        return asdict(self)


class LicenseGate:
    """Enforces data governance policies and blocks unauthorized sources."""

    def __init__(self, sources_config_path: Optional[Path] = None, approved_sources: Optional[Set[str]] = None):
        self.config_path = sources_config_path or SOURCES_CONFIG_PATH
        self.registry = self._load_registry()
        # Explicit owner-approved source IDs
        self.approved_sources: Set[str] = approved_sources or set()

    def _load_registry(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            return data.get("sources", {}) if data else {}
        except Exception as e:
            logger.warning(f"Could not load sources config from {self.config_path}: {e}")
            return {}

    def is_source_allowed(self, source_id: str, allow_unverified: bool = False) -> bool:
        """Check if a source is permitted under license policy."""
        # Explicit owner approval always grants local research permission
        if source_id in self.approved_sources:
            return True

        src_info = self.registry.get(source_id)
        if not src_info:
            logger.warning(f"Source '{source_id}' is not in the source registry.")
            return False

        status = src_info.get("status", "UNVERIFIED")
        if status in ("VERIFIED", "VERIFIED_RESEARCH_ONLY"):
            return True

        if status == "UNVERIFIED":
            return bool(allow_unverified)

        return False

    def check_source_access(self, source_id: str, allow_unverified: bool = False) -> None:
        """Assert access permission or raise LicenseGateError."""
        if not self.is_source_allowed(source_id, allow_unverified=allow_unverified):
            src_info = self.registry.get(source_id, {})
            license_text = src_info.get("license", "Unknown")
            raise LicenseGateError(
                f"License Gate Blocked Source '{source_id}'. "
                f"License status is '{src_info.get('status', 'UNVERIFIED')}' (License: '{license_text}'). "
                f"Data marked UNVERIFIED requires explicit owner approval before use and must NEVER be "
                f"redistributed or committed to git (see DATA_GOVERNANCE.md)."
            )

    def can_redistribute_source(self, source_id: str) -> bool:
        """Check if source data is permitted for redistribution/committing to git."""
        src_info = self.registry.get(source_id, {})
        return bool(src_info.get("redistribution_allowed", False))


def create_provenance_record(
    seq_id: str,
    sequence: str,
    source: str,
    accession: str,
    label: int,
    label_evidence: str,
    organism: str = "unknown",
    deposit_date: Optional[str] = None,
    transformations: Optional[List[str]] = None,
    removal_reason: Optional[str] = None,
    license_gate: Optional[LicenseGate] = None,
    allow_unverified: bool = False,
) -> SequenceProvenanceRecord:
    """Factory function that creates and validates a provenance record through the License Gate."""
    gate = license_gate or LicenseGate()
    gate.check_source_access(source, allow_unverified=allow_unverified)

    src_info = gate.registry.get(source, {})
    license_str = src_info.get("license", "UNVERIFIED")

    today_str = datetime.date.today().isoformat()
    record = SequenceProvenanceRecord(
        seq_id=seq_id,
        sequence=sequence,
        source=source,
        accession=accession,
        label=label,
        label_evidence=label_evidence,
        license=license_str,
        ingest_date=today_str,
        organism=organism,
        deposit_date=deposit_date,
        transformations=transformations or [],
        removal_reason=removal_reason,
    )
    record.validate()
    return record
