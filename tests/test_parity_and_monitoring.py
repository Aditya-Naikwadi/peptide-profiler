"""Tests for Phase 5: Train/Serve Parity, Production Monitoring, Shadow Mode, and Report Stamping."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.monitoring import (
    calculate_psi,
    calculate_ks,
    calculate_length_bin_mix,
    log_request_audit,
    evaluate_batch_drift,
)
from src.shadow import ShadowComparisonEngine
from src.aggregator import export_reports, profile_sequence, to_dataframe
from src.physicochem import AMINO_ACIDS, calculate_aac, calculate_pcp_descriptors


# ==========================================
# 1. TRAIN / SERVE PARITY & GOLDEN VECTORS
# ==========================================

def test_train_serve_parity_golden_vector():
    """Verify that feature extraction on the golden control sequence strictly matches frozen values."""
    golden_seq = "GIGAVLKVLTTGLPALISWIKRKRQQ"  # Melittin
    aac = calculate_aac(golden_seq)
    pcp = calculate_pcp_descriptors(golden_seq)
    vector = [aac[aa] for aa in AMINO_ACIDS] + list(pcp.values())

    assert len(vector) == 50, f"Expected 50-D feature vector, got {len(vector)}"
    # Frozen reference sum and specific moments
    assert np.isclose(round(sum(vector), 3), 104.952, atol=1e-2)
    assert np.isclose(vector[0], 7.692, atol=1e-2)  # Ala AAC


def test_pinned_dependency_versions():
    """Verify essential bioinformatics dependencies match minimum pinned operational baselines."""
    import Bio
    version_parts = [int(p) for p in Bio.__version__.split(".") if p.isdigit()]
    assert version_parts[0] >= 1 and version_parts[1] >= 80, f"Biopython must be >= 1.80, got {Bio.__version__}"


# ==========================================
# 2. LOCAL AUDIT REQUEST LOGGING
# ==========================================

def test_local_request_logging(tmp_path):
    """Verify append-only local inference auditing with zero network dependency."""
    log_file = tmp_path / "test_audit.jsonl"
    seq_rec = {"id": "seq_test", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    prof = profile_sequence(seq_rec, candidate_type="vaccine")

    log_request_audit(seq_rec, prof, sanitizer_edit_count=0, log_path=log_file)

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    assert entry["id"] == "seq_test"
    assert entry["length"] == 20
    assert "scores" in entry
    assert "candidate_status" in entry
    assert "metadata" in entry
    assert "model_hashes" in entry["metadata"]


# ==========================================
# 3. DRIFT MONITORING (PSI & KS)
# ==========================================

def test_psi_identical_vs_drifted():
    """Verify PSI calculation yields near 0.0 for identical distributions and > 0.25 for shifted ones."""
    rng = np.random.default_rng(42)
    base = rng.normal(100, 15, size=200)
    identical = rng.normal(100, 15, size=200)
    drifted = rng.normal(150, 25, size=200)

    psi_same = calculate_psi(base, identical)
    psi_drift = calculate_psi(base, drifted)

    assert psi_same < 0.10, f"Expected nominal PSI < 0.10, got {psi_same}"
    assert psi_drift > 0.25, f"Expected critical PSI > 0.25, got {psi_drift}"


def test_ks_distribution_test():
    """Verify KS two-sample test detects continuous distribution shift."""
    rng = np.random.default_rng(42)
    s1 = rng.uniform(0.0, 1.0, size=150)
    s2 = rng.uniform(0.5, 1.5, size=150)

    stat, pval = calculate_ks(s1, s2)
    assert pval < 0.01, f"Expected significant shift p < 0.01, got {pval}"


# ==========================================
# 4. SHADOW MODE & AUTO-ROLLBACK TRIPWIRE
# ==========================================

def test_shadow_mode_and_auto_rollback(tmp_path):
    """Verify shadow comparison detects status disagreements and triggers auto-rollback on tripwire."""
    shadow_log = tmp_path / "shadow.jsonl"
    engine = ShadowComparisonEngine(max_disagreement_tripwire=0.10, log_file=shadow_log)

    # 10 test candidates
    prod_batch = [
        {"id": f"c_{i}", "candidate_status": "APPROVED", "desirability": {"conservative_rank_score": 0.80}}
        for i in range(10)
    ]
    # Candidate model disagrees on 3 out of 10 (30% disagreement rate > 10% tripwire)
    cand_batch = [
        {"id": f"c_{i}", "candidate_status": "EXCLUDED" if i < 3 else "APPROVED", "desirability": {"conservative_rank_score": 0.20 if i < 3 else 0.80}}
        for i in range(10)
    ]

    report = engine.compare_batches(prod_batch, cand_batch, top_k=3)

    assert report["status_disagreement_count"] == 3
    assert report["status_disagreement_rate"] == 0.30
    assert report["tripwire_triggered"] is True
    assert report["canary_status"] == "ROLLED_BACK"
    assert engine.is_rolled_back is True


# ==========================================
# 5. WET-LAB BIAS MITIGATION SAMPLING
# ==========================================

def test_wet_lab_bias_mitigation_sampling():
    """Verify candidate selection for assay synthesis includes stratified lower-ranked controls."""
    from scripts.log_wet_lab_feedback import sample_candidates_with_bias_mitigation

    mock_profiles = [
        {"id": f"lead_{i}", "candidate_status": "APPROVED", "desirability": {"conservative_rank_score": 0.9 - i * 0.05}}
        for i in range(15)
    ] + [
        {"id": f"excl_{j}", "candidate_status": "EXCLUDED", "desirability": {"conservative_rank_score": 0.0}}
        for j in range(5)
    ]

    sampled = sample_candidates_with_bias_mitigation(mock_profiles, top_k_count=5, lower_ranked_sample_rate=0.40)
    strata = [s.get("selection_sampling_stratum") for s in sampled]

    assert "TOP_RANKED_LEAD" in strata
    assert "STRATIFIED_LOWER_RANKED" in strata
    assert len(sampled) == 7  # 5 top + 2 exploratory


# ==========================================
# 6. REPORT PROVENANCE STAMPING
# ==========================================

def test_report_provenance_stamping(tmp_path):
    """Verify reports are stamped with model hashes, calibration set ID, and cluster version."""
    records = [{"id": "p_test", "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ"}]
    profiles = [profile_sequence(records[0], candidate_type="vaccine")]
    out_prefix = tmp_path / "stamped_report"

    generated = export_reports(profiles, out_prefix, formats=["csv", "json", "html"])

    # 1. JSON
    json_data = json.loads(generated["json"].read_text(encoding="utf-8"))
    assert "provenance" in json_data[0]
    prov = json_data[0]["provenance"]
    assert "model_sha256_hashes" in prov
    assert prov["calibration_set_id"] == "swiss_prot_eval_v1_437_clusters"

    # 2. CSV
    df = to_dataframe(profiles)
    assert "Model_Tox_Hash" in df.columns
    assert "Calibration_Set_ID" in df.columns

    # 3. HTML
    html_text = generated["html"].read_text(encoding="utf-8")
    assert "swiss_prot_eval_v1_437_clusters" in html_text
    assert "Provenance & Audit Stamp" in html_text
