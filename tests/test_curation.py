"""Unit and integration tests for sequence sanitization, deduplication, and negative construction."""

import pytest
from pathlib import Path

from src.data.deduplication import deduplicate_records
from src.data.loaders import DatasetLoader, compute_file_sha256
from src.data.negatives import build_length_matched_negatives, check_annotation_validity
from src.data.sanitizer import sanitize_sequence


def test_sanitizer_clean_sequence():
    """Verify standard sequence passes without alterations."""
    seq = "MKTLLILAVVAAALA"
    res = sanitize_sequence(seq)
    assert res.sanitized_sequence == seq
    assert res.altered_count == 0
    assert res.is_flagged is False
    assert res.is_dropped is False


def test_sanitizer_ambiguous_mapping_and_logging():
    """Verify ambiguous residues (B, Z, U) are correctly converted and logged."""
    seq = "MKBTLLZLAU"
    res = sanitize_sequence(seq)
    # B -> N, Z -> Q, U -> C
    assert res.sanitized_sequence == "MKNTLLQLAC"
    assert res.altered_count == 3
    actions = [a.action for a in res.alterations]
    assert all(act == "mapped" for act in actions)


def test_sanitizer_high_alteration_flagging():
    """Verify sequences with > 5% non-standard residues are flagged."""
    # 2 non-standard residues in 10-residue sequence = 20% alteration
    seq = "MKBBLLLLLL"
    res = sanitize_sequence(seq, max_altered_fraction=0.05)
    assert res.altered_fraction == 0.20
    assert res.is_flagged is True
    assert "HIGH_ALTERATION_FRACTION" in res.flag_reason


def test_sanitizer_fragment_dropping():
    """Verify fragments below min_length are marked as dropped."""
    seq = "MKT"
    res = sanitize_sequence(seq, min_length=5)
    assert res.is_dropped is True
    assert "FRAGMENT_BELOW_MIN_LENGTH" in res.drop_reason


def test_deduplication_exact_duplicates():
    """Verify exact duplicates are merged and logged."""
    records = [
        {"id": "rec1", "sequence": "ACDEFGHIKLMNPQRSTVWY", "label": 1, "label_evidence": "curated_literature"},
        {"id": "rec2", "sequence": "ACDEFGHIKLMNPQRSTVWY", "label": 1, "label_evidence": "curated_literature"},
    ]
    cleaned, logs = deduplicate_records(records)
    assert len(cleaned) == 1
    assert len(logs) == 1
    assert logs[0].removal_reason == "EXACT_DUPLICATE_RETAINED_PRIMARY"


def test_deduplication_conflict_arbitration():
    """Verify higher evidence level arbitrates conflicting labels."""
    records = [
        {"id": "low_ev", "sequence": "ACDEFGHIKLMNPQRSTVWY", "label": 0, "label_evidence": "predicted"},
        {"id": "high_ev", "sequence": "ACDEFGHIKLMNPQRSTVWY", "label": 1, "label_evidence": "experimental_challenge_protection"},
    ]
    cleaned, logs = deduplicate_records(records, resolve_conflicts=True)
    assert len(cleaned) == 1
    assert cleaned[0]["id"] == "high_ev"
    assert cleaned[0]["label"] == 1


def test_negatives_annotation_filtering():
    """Verify sequences with forbidden keywords are filtered."""
    assert check_annotation_validity("Ribosomal protein L12", ["Ribosome"]) is True
    assert check_annotation_validity("Lethal pore-forming bee venom toxin", ["Toxin"]) is False
    assert check_annotation_validity("Major birch pollen allergen Bet v 1", ["Allergen"]) is False


def test_build_length_matched_negatives():
    """Verify length-matched negative construction with homology rejection."""
    positives = [
        {"id": "pos_1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},  # len 20
        {"id": "pos_2", "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ"},  # len 26
    ]
    candidates = [
        {"id": "neg_homolog", "sequence": "ACDEFGHIKLMNPQRSTVWA", "description": "Hypothetical protein"},  # len 20, homolog (>90%)
        {"id": "neg_good_1", "sequence": "WWWWYYYYFFFFLLLLIIII", "description": "Metabolic enzyme"},  # len 20, non-homologous
        {"id": "neg_good_2", "sequence": "AAGGAAGGAAGGAAGGAAGGAAGGAA", "description": "Ribosomal protein"},  # len 26, non-homologous
        {"id": "neg_toxin", "sequence": "WWWWYYYYFFFFLLLLIIII", "description": "Snake venom neurotoxin"},  # Forbidden keyword
    ]

    selected, metrics = build_length_matched_negatives(positives, candidates, homology_cutoff=0.30)
    assert len(selected) == 2
    selected_ids = {s["id"] for s in selected}
    assert "neg_good_1" in selected_ids
    assert "neg_good_2" in selected_ids
    assert "neg_homolog" not in selected_ids
    assert "neg_toxin" not in selected_ids
    assert metrics["dropped_by_annotation"] >= 1
    assert metrics["dropped_by_homology"] >= 1


def test_dataset_loader_integration():
    """Verify loading evaluation_dataset.json through DatasetLoader with provenance."""
    eval_file = Path("data/evaluation_dataset.json")
    if eval_file.exists():
        loader = DatasetLoader()
        records, stats = loader.load_json_dataset(eval_file, source_id="uniprot_swissprot")
        assert stats["valid_records"] > 0
        assert len(stats["sha256"]) == 64
        # Verify first record is full provenance record
        r0 = records[0]
        assert r0.source == "uniprot_swissprot"
        assert r0.license == "CC BY 4.0"
