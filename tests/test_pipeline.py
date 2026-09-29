"""Comprehensive unit and integration test suite for the Peptide Characterization Pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.aggregator import (
    calculate_desirability,
    export_reports,
    predict_bcell_epitopes,
    profile_multiple_sequences,
    profile_sequence,
    to_dataframe,
)
from src.allergenicity import run_allergenicity
from src.antigenicity import run_antigenicity
from src.cli import main as cli_main
from src.parser import SequenceValidationError, clean_sequence, parse_fasta, write_fasta
from src.physicochem import (
    calculate_aac,
    calculate_autocorrelation,
    calculate_pcp_descriptors,
    run_physicochemical,
)
from src.toxicity import run_toxicity


# ==========================================
# 1. PARSER TESTS
# ==========================================

def test_clean_sequence_valid():
    raw = "  mkTl LiLa  "
    cleaned = clean_sequence(raw)
    assert cleaned == "MKTLLILA"


def test_clean_sequence_invalid():
    with pytest.raises(SequenceValidationError):
        clean_sequence("MKT123")


def test_clean_sequence_sanitize():
    cleaned = clean_sequence("MKTXBL", sanitize=True)
    assert "X" not in cleaned
    assert "B" not in cleaned
    assert len(cleaned) == 6


def test_parse_fasta_multiline():
    fasta_str = """>seq1 Test sequence
MKTLLILAVV
AAALA
>seq2 Second sequence
ALWKTLLKKVLKAAAKA
"""
    records = parse_fasta(fasta_str, is_content=True)
    assert len(records) == 2
    assert records[0]["id"] == "seq1"
    assert records[0]["sequence"] == "MKTLLILAVVAAALA"
    assert records[1]["id"] == "seq2"
    assert records[1]["length"] == 17


def test_parse_fasta_file_io(tmp_path):
    fa_file = tmp_path / "test.fa"
    records_in = [{"id": "p1", "sequence": "ACDEFGHIKLMNPQRSTVWY"}]
    write_fasta(records_in, fa_file)
    assert fa_file.exists()

    records_out = parse_fasta(fa_file)
    assert len(records_out) == 1
    assert records_out[0]["id"] == "p1"
    assert records_out[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"


# ==========================================
# 2. PHYSICOCHEMICAL MODULE TESTS
# ==========================================

def test_run_physicochemical():
    seq = "ALWKTLLKKVLKAAAKA"
    res = run_physicochemical(seq)

    assert "mol_weight" in res
    assert res["mol_weight"] > 1000.0
    assert "gravy" in res
    assert "instability_index" in res
    assert "is_stable" in res
    assert "pcp_vector" in res
    assert len(res["pcp_vector"]) == 30
    assert "aac" in res
    assert len(res["aac"]) == 20
    assert sum(res["aac"].values()) == pytest.approx(100.0, rel=1e-1)


def test_pcp_and_autocorrelation():
    seq = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    pcp = calculate_pcp_descriptors(seq)
    assert len(pcp) == 30
    assert "PCP_PC" in pcp  # Positive charge
    assert "PCP_NP" in pcp  # Nonpolar
    assert pcp["PCP_PC"] > 0.1  # Melittin is positively charged

    acr = calculate_autocorrelation(seq)
    assert "ACR_lag_1" in acr


# ==========================================
# ==========================================
# 3. TOXICITY MODULE SMOKE TESTS (ToxinPred2 ONNX)
# NOTE: These tests use canonical reference controls (Melittin, Exendin-4)
# that were present in upstream training sets. They serve strictly as
# smoke tests to verify pipeline execution without crashing, NOT as
# evidence of out-of-distribution generalization.
# ==========================================

def test_toxicity_prediction_melittin():
    """Smoke test: verify pipeline execution and score extraction on positive control Melittin."""
    # Known toxic bee venom peptide (archetypal positive training control)
    melittin = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    res = run_toxicity(melittin, threshold=0.6)

    assert "toxicity_score" in res
    assert 0.0 <= res["toxicity_score"] <= 1.0
    assert "is_toxic" in res
    assert res["is_toxic"] is True
    assert res["toxicity_score"] >= 0.6


def test_toxicity_prediction_exendin():
    """Smoke test: verify pipeline execution on known negative control Exendin-4."""
    # Known safe therapeutic peptide
    exendin = "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS"
    res = run_toxicity(exendin, threshold=0.6)

    assert res["is_toxic"] is False
    assert res["toxicity_score"] < 0.6


# ==========================================
# 4. ANTIGENICITY MODULE SMOKE TESTS
# NOTE: These tests use canonical reference controls (OspA, GAPDH)
# to verify execution flow. They are smoke tests, NOT generalizable validation.
# ==========================================

def test_antigenicity_prediction():
    """Smoke test: verify pipeline execution on archetypal bacterial antigen OspA."""
    # OspA outer surface protein (canonical positive training control)
    ospa = "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK"
    res = run_antigenicity(ospa, threshold=0.5)

    assert "antigenicity_score" in res
    assert 0.0 <= res["antigenicity_score"] <= 1.0
    assert res["is_antigen"] is True
    assert res["antigenicity_score"] >= 0.5


def test_antigenicity_non_antigen():
    """Smoke test: verify pipeline execution on cytoplasmic control GAPDH fragment."""
    # Housekeeping cytoplasmic protein GAPDH fragment
    gapdh = "MVKVGVNGFGRIGRLVTRAAFNSGKDVI"
    res = run_antigenicity(gapdh, threshold=0.5)

    assert res["is_antigen"] is False
    assert res["antigenicity_score"] < 0.5


# ==========================================
# 5. ALLERGENICITY MODULE SMOKE TESTS
# NOTE: These tests use archetypal allergen controls (Bet v 1).
# They are smoke tests, NOT statistical validation of allergenicity.
# ==========================================

def test_allergenicity_prediction():
    """Smoke test: verify pipeline execution on archetypal birch pollen allergen Bet v 1."""
    # Bet v 1 birch allergen (canonical positive control)
    bet_v_1 = "GVFNYETETTSVIPAARLFKAFILDGDNLFPKVAPQAISSVENIEGNGGPGTIKKISFPEGFPFKYVKDRVDEVDHTNFKYNYSVIEGGPIGDTLEKISNEIKIVATPDGGSILKISNKYHTKGDHEVKAEQVKASKEMGETLLRAVESYLLAHSDAYN"
    res = run_allergenicity(bet_v_1, threshold=0.5)

    assert "allergenicity_score" in res
    assert 0.0 <= res["allergenicity_score"] <= 1.0
    assert res["is_allergen"] is True


def test_allergenicity_non_allergen():
    """Smoke test: verify pipeline execution on negative control GAPDH fragment."""
    gapdh = "MVKVGVNGFGRIGRLVTRAAFNSGKDVI"
    res = run_allergenicity(gapdh, threshold=0.5)

    assert res["is_allergen"] is False


# ==========================================
# 6. AGGREGATOR & DESIRABILITY TESTS
# ==========================================

def test_aggregator_schema():
    rec = {"id": "test_pep", "sequence": "MKTLLILAVVAAALA"}
    profile = profile_sequence(rec)

    # Validate exact format from project specification
    assert profile["id"] == "test_pep"
    assert profile["sequence"] == "MKTLLILAVVAAALA"
    assert "antigenicity" in profile and "score" in profile["antigenicity"] and "is_antigen" in profile["antigenicity"]
    assert "allergenicity" in profile and "score" in profile["allergenicity"] and "is_allergen" in profile["allergenicity"]
    assert "toxicity" in profile and "score" in profile["toxicity"] and "is_toxic" in profile["toxicity"]
    assert "physicochemical" in profile
    assert "mol_weight" in profile["physicochemical"]
    assert "gravy" in profile["physicochemical"]
    assert "instability_index" in profile["physicochemical"]
    assert "desirability" in profile


def test_desirability_ranking():
    records = [
        {"id": "antigen", "sequence": "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK"},
        {"id": "toxin", "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ"},
    ]
    profiles = profile_multiple_sequences(records, candidate_type="vaccine", rank_candidates=True)
    assert len(profiles) == 2
    assert profiles[0]["desirability"]["rank"] == 1
    assert profiles[1]["desirability"]["rank"] == 2
    assert profiles[0]["desirability"]["score"] >= profiles[1]["desirability"]["score"]


def test_bcell_epitope_prediction():
    seq = "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK"
    epitopes = predict_bcell_epitopes(seq, window_size=7, threshold=1.0)
    assert isinstance(epitopes, list)
    if epitopes:
        ep = epitopes[0]
        assert "start" in ep
        assert "end" in ep
        assert "sequence" in ep
        assert "mean_score" in ep


def test_export_reports(tmp_path):
    records = [{"id": "test1", "sequence": "ALWKTLLKKVLKAAAKA"}]
    profiles = profile_multiple_sequences(records)
    out_prefix = tmp_path / "report"

    generated = export_reports(profiles, out_prefix, formats=["csv", "json", "html"])
    assert "csv" in generated and generated["csv"].exists()
    assert "json" in generated and generated["json"].exists()
    assert "html" in generated and generated["html"].exists()

    # Verify JSON content
    data = json.loads(generated["json"].read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["id"] == "test1"


# ==========================================
# 7. CLI INTEGRATION TEST
# ==========================================

def test_cli_execution(tmp_path):
    fa_file = tmp_path / "input.fa"
    fa_file.write_text(">p1\nALWKTLLKKVLKAAAKA\n>p2\nGIGAVLKVLTTGLPALISWIKRKRQQ\n", encoding="utf-8")
    out_file = tmp_path / "out.csv"

    cli_main(["-i", str(fa_file), "-o", str(out_file), "--format", "all"])

    assert out_file.exists()
    json_file = tmp_path / "out.json"
    html_file = tmp_path / "out.html"
    assert json_file.exists()
    assert html_file.exists()


# ==========================================
# 8. PHASE 1 HOMOLOGY & LEAKAGE VALIDITY TESTS
# ==========================================

from src.homology import cluster_sequences_by_homology, sequence_similarity


def test_homology_clustering_grouping():
    """Verify that homologous sequences (>35% identity) are grouped into the same cluster."""
    records = [
        {"id": "p1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        # 1-residue substitution homolog of p1
        {"id": "p1_homolog", "sequence": "ACDEFGHIKLMNPQRSTVWA"},
        # Completely distinct sequence
        {"id": "p2_distant", "sequence": "WWWWYYYYFFFFLLLLIIII"},
    ]
    res = cluster_sequences_by_homology(records, identity_threshold=0.35)
    assignments = res["assignments"]

    assert assignments["p1"] == assignments["p1_homolog"], "Homologs must share the same cluster ID"
    assert assignments["p1"] != assignments["p2_distant"], "Distant sequence must be in a separate cluster"


def test_sequence_similarity_short_peptides():
    """Verify sequence similarity logic on short peptides (<30 aa)."""
    # 2 identical except for 1 residue (high identity)
    seq_a = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    seq_b = "GIGAVLKVLTTGLPALISWIKRKRQA"
    assert sequence_similarity(seq_a, seq_b, identity_threshold=0.35) is True

    # Poly-glycine vs poly-tryptophan (zero similarity)
    seq_c = "GGGGGGGGGGGGGGGGGGGG"
    seq_d = "WWWWWWWWWWWWWWWWWWWW"
    assert sequence_similarity(seq_c, seq_d, identity_threshold=0.35) is False


# ==========================================
# 9. PHASE 2 CALIBRATION & PREVALENCE TESTS
# ==========================================

from src.calibration import bayesian_prior_shift, calculate_ppv_at_prevalence, compute_ece
import numpy as np


def test_bayesian_prior_shift():
    """Verify that Bayesian prior shift correctly downweights balanced predictions to rare prevalence."""
    # Balanced probability 0.50 should drop drastically at 1% deployment prevalence
    raw_probs = np.array([0.50, 0.90, 0.10])
    shifted_01 = bayesian_prior_shift(raw_probs, train_prior=0.50, deploy_prior=0.01)

    assert shifted_01[0] < 0.02, "Balanced probability (0.50) must shift to ~0.01 at 1% prevalence"
    assert shifted_01[1] < raw_probs[1], "High confidence must be adjusted down"
    assert shifted_01[2] < raw_probs[2], "Low confidence must be adjusted down"


def test_ppv_at_prevalence():
    """Verify Positive Predictive Value (PPV) calculation under rare prevalence."""
    # 95% sensitivity, 80% specificity at 1% prevalence
    ppv = calculate_ppv_at_prevalence(sensitivity=0.95, specificity=0.80, prevalence=0.01)
    # Expected: (0.95 * 0.01) / (0.95 * 0.01 + 0.20 * 0.99) = 0.0095 / (0.0095 + 0.198) = 0.0458
    assert 0.04 <= ppv <= 0.05, f"Expected PPV ~4.58%, got {ppv}"


def test_hard_safety_gates_and_reason_codes():
    """Verify Phase 4 hard safety gates exclude confirmed toxins and assign reason codes."""
    from src.aggregator import profile_sequence

    # Melittin is a known lethal pore-forming toxin
    melittin = {
        "id": "Melittin_Control",
        "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    }
    prof = profile_sequence(melittin, candidate_type="vaccine", enforce_safety_gates=True)
    assert prof["candidate_status"] == "EXCLUDED", "Melittin must be EXCLUDED by hard safety gate"
    assert "SAFETY_VIOLATION_TOXICITY" in prof["reason_codes"]
    assert prof["desirability"]["score"] == 0.0
    assert prof["desirability"]["conservative_rank_score"] == 0.0


def test_conservative_ranking_and_abstention():
    """Verify conservative bound penalizes residual risk and flags short peptide abstentions."""
    from src.aggregator import profile_sequence

    # Short peptide (< 15 aa)
    short_seq = {
        "id": "Short_Peptide_Control",
        "sequence": "CFWKTC",
    }
    prof = profile_sequence(short_seq, candidate_type="therapeutic", enforce_safety_gates=False)
    assert "LOW_CONFIDENCE_SHORT_PEPTIDE" in prof["abstentions"]
    assert prof["desirability"]["conservative_rank_score"] <= prof["desirability"]["score"]



