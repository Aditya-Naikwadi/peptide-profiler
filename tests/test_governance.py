"""Unit tests for data governance, provenance tracking, and license gate enforcement."""

import pytest
from pathlib import Path

from src.data.governance import (
    LicenseGate,
    LicenseGateError,
    ProvenanceValidationError,
    SequenceProvenanceRecord,
    create_provenance_record,
)


def test_provenance_record_validation_success():
    """Verify that a compliant provenance record validates and converts to dictionary."""
    rec = SequenceProvenanceRecord(
        seq_id="TEST_001",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        source="uniprot_swissprot",
        accession="P12345",
        label=1,
        label_evidence="curated_literature",
        license="CC BY 4.0",
        ingest_date="2026-09-29",
        organism="bacterial",
        deposit_date="2015-01-01",
        transformations=["uppercase", "sanitized"],
    )
    rec.validate()
    d = rec.to_dict()
    assert d["seq_id"] == "TEST_001"
    assert d["label"] == 1
    assert d["license"] == "CC BY 4.0"
    assert d["organism"] == "bacterial"


def test_provenance_record_validation_invalid_label():
    """Verify that non-binary labels raise ProvenanceValidationError."""
    rec = SequenceProvenanceRecord(
        seq_id="TEST_002",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        source="uniprot_swissprot",
        accession="P12345",
        label=5,  # Invalid: must be 0 or 1
        label_evidence="curated_literature",
        license="CC BY 4.0",
        ingest_date="2026-09-29",
    )
    with pytest.raises(ProvenanceValidationError):
        rec.validate()


def test_provenance_record_validation_empty_fields():
    """Verify that missing required fields raise ProvenanceValidationError."""
    rec = SequenceProvenanceRecord(
        seq_id="",
        sequence="ACDEF",
        source="uniprot_swissprot",
        accession="P12345",
        label=0,
        label_evidence="curated",
        license="CC BY 4.0",
        ingest_date="2026-09-29",
    )
    with pytest.raises(ProvenanceValidationError):
        rec.validate()


def test_license_gate_verified_sources():
    """Verify that verified sources pass the license gate automatically."""
    gate = LicenseGate()
    # UniProt, VaxiGen Tumor, COMPARE are verified
    assert gate.is_source_allowed("uniprot_swissprot") is True
    assert gate.is_source_allowed("vaxigen_tumor") is True
    assert gate.is_source_allowed("compare_database") is True

    # Should not raise exception
    gate.check_source_access("uniprot_swissprot")


def test_license_gate_unverified_sources_blocked():
    """Verify that unverified sources raise LicenseGateError unless explicitly approved."""
    gate = LicenseGate()
    # VaxiJen bacterial, ToxinPred, AllergenOnline are UNVERIFIED
    assert gate.is_source_allowed("vaxijen_bacterial") is False
    assert gate.is_source_allowed("toxinpred_datasets") is False

    with pytest.raises(LicenseGateError) as exc_info:
        gate.check_source_access("vaxijen_bacterial")
    assert "License Gate Blocked Source" in str(exc_info.value)
    assert "UNVERIFIED" in str(exc_info.value)


def test_license_gate_owner_explicit_approval():
    """Verify that explicit owner approval allows source access for local research."""
    gate = LicenseGate(approved_sources={"vaxijen_bacterial"})
    assert gate.is_source_allowed("vaxijen_bacterial") is True
    # Should not raise
    gate.check_source_access("vaxijen_bacterial")


def test_license_gate_redistribution_checks():
    """Verify redistribution flag is enforced according to source license."""
    gate = LicenseGate()
    assert gate.can_redistribute_source("uniprot_swissprot") is True
    assert gate.can_redistribute_source("vaxigen_tumor") is True
    # Unverified or restricted sources cannot be redistributed
    assert gate.can_redistribute_source("vaxijen_bacterial") is False
    assert gate.can_redistribute_source("iedb") is False
    assert gate.can_redistribute_source("protegen") is False


def test_create_provenance_record_integration():
    """Verify factory creation through license gate."""
    rec = create_provenance_record(
        seq_id="SWISS_001",
        sequence="MKTLLILAVVAAALA",
        source="uniprot_swissprot",
        accession="Q8N135",
        label=0,
        label_evidence="curated_literature",
        organism="human",
    )
    assert rec.seq_id == "SWISS_001"
    assert rec.license == "CC BY 4.0"
    assert rec.label == 0

    # Unverified source without approval raises LicenseGateError
    with pytest.raises(LicenseGateError):
        create_provenance_record(
            seq_id="VAX_001",
            sequence="MKTLLILAVVAAALA",
            source="vaxijen_bacterial",
            accession="IMM_001",
            label=1,
            label_evidence="curated_experimental",
        )
