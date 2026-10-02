"""Unit Tests for Deterministic FAO/WHO Regulatory Rule Engine and Hybrid Pipeline.

Verifies:
1. Primary 80-mer rule requires strict >35% identity (at least 29/80 identities, 28/80 does not trigger).
2. Short queries (<80 aa) return 'not_applicable_short_query', never 'pass'.
3. Short-match rule (6-aa and 8-aa) triggers accurately.
4. Hybrid hierarchy correctly applies high_risk override on rule hit and delegates to ML on no_homology_evidence.
5. Provenance hash and database license are embedded in outputs.
6. Exclusion of self-hits and cluster-homologs functions strictly.
"""

from __future__ import annotations

import pytest

from src.data.allergen_db import ReferenceAllergenDatabase
from src.models.fao_who_engine import FAOWHOAllergenicityEngine
from src.models.hybrid_allergen_pipeline import HybridAllergenPipeline


@pytest.fixture
def mock_allergen_db():
    allergens = [
        {
            "id": "REF_ALLERGEN_1",
            "description": "Major birch pollen allergen Bet v 1",
            "sequence": "GVFNYETETTSVIPAARLFKAFILDGDNLFPKVAPQAISSVENIEGNGGPGTIKKISFPEGFPFKYVKDRVDEVDHTNFKYNYSVIEGGPIGDTLEKISNEIKIVATPDGGSILKISNKYHTKGDHEVKAEQVKASKEMGETLLRAVESYLLAHSDAYN",
            "cluster_id": "cluster_0001",
        },
        {
            "id": "REF_ALLERGEN_2",
            "description": "Peanut allergen Ara h 1 fragment",
            "sequence": "RQQPEENACESELEIKQIKEDDGEKDEEEEEDEEEDEEDEEEDEDEEDEEEDEDEEDEEE",
            "cluster_id": "cluster_0002",
        },
    ]
    return ReferenceAllergenDatabase(allergens=allergens, db_name="TestDB", db_version="1.0-test")


def test_primary_80mer_threshold_strict_inequality():
    """Verify 28/80 (35.0%) does not trigger, while 29/80 (36.25%) triggers."""
    # Create controlled 80-mer database allergen
    db_80 = ReferenceAllergenDatabase(allergens=[
        {"id": "CONTROL_80", "sequence": ("A" * 28) + ("V" * 52), "cluster_id": "c1"}
    ])
    engine = FAOWHOAllergenicityEngine(db=db_80)

    # 28 identities out of 80: strict >35% MUST NOT trigger (28/80 = 35.0%)
    query_28 = ("A" * 28) + ("I" * 52)
    res_28 = engine.evaluate_primary_80mer_rule(query_28)
    assert res_28["max_80mer_identities"] == 28
    assert res_28["status"] == "no_homology_evidence"

    # 29 identities out of 80: strict >35% MUST trigger (29/80 = 36.25%)
    query_29 = ("A" * 28) + "V" + ("I" * 51)
    res_29 = engine.evaluate_primary_80mer_rule(query_29)
    assert res_29["max_80mer_identities"] == 29
    assert res_29["status"] == "hit"


def test_short_query_handling_explicit_status(mock_allergen_db):
    """Short queries (<80 aa) must return not_applicable_short_query, never 'pass'."""
    engine = FAOWHOAllergenicityEngine(db=mock_allergen_db)
    short_seq = "GVFNYETETTSVIPAARLFKAFILDGDNLFPKVAPQAISSVENIEGNGGPGTIKKIS"  # 57 aa (< 80)

    res = engine.evaluate_primary_80mer_rule(short_seq)
    assert res["status"] == "not_applicable_short_query"
    assert "not_applicable_short_query" in res["reason"]
    assert res["max_80mer_identity"] is None


def test_short_match_rule_6aa_and_8aa(mock_allergen_db):
    """Verify contiguous 6-aa and 8-aa match detection."""
    # 6-aa match from REF_ALLERGEN_1: "GVFNYE"
    engine_6aa = FAOWHOAllergenicityEngine(db=mock_allergen_db, short_match_k=6)
    query_6 = "MKTLLILAGVFNYEAAALAS"
    res_6 = engine_6aa.evaluate_short_match_rule(query_6)
    assert res_6["status"] == "hit"
    assert res_6["matched_kmer"] == "GVFNYE"
    assert res_6["matched_allergen_id"] == "REF_ALLERGEN_1"

    # 8-aa stricter mode: "GVFNYEAA" does not match ("GVFNYETE" is reference)
    engine_8aa = FAOWHOAllergenicityEngine(db=mock_allergen_db, strict_mode_8aa=True)
    res_8 = engine_8aa.evaluate_short_match_rule(query_6)
    assert res_8["status"] == "no_homology_evidence"

    # Exact 8-aa match: "GVFNYETE"
    query_8 = "MKTLLILAGVFNYETEAAALAS"
    res_8_hit = engine_8aa.evaluate_short_match_rule(query_8)
    assert res_8_hit["status"] == "hit"
    assert res_8_hit["matched_kmer"] == "GVFNYETE"


def test_self_hit_and_cluster_exclusion(mock_allergen_db):
    """Verify that excluding IDs and clusters prevents hits."""
    engine = FAOWHOAllergenicityEngine(db=mock_allergen_db)
    query = mock_allergen_db.allergens[0]["sequence"]

    # Without exclusion -> hit
    res_no_ex = engine.evaluate(query)
    assert res_no_ex["regulatory_flag"] == "high_risk"

    # With self ID exclusion
    res_id_ex = engine.evaluate(query, excluded_ids={"REF_ALLERGEN_1"})
    # Should not hit REF_ALLERGEN_1
    assert res_id_ex["short_match_rule"]["matched_allergen_id"] != "REF_ALLERGEN_1"

    # With cluster exclusion
    res_cl_ex = engine.evaluate(query, excluded_clusters={"cluster_0001"})
    assert res_cl_ex["short_match_rule"]["matched_allergen_id"] != "REF_ALLERGEN_1"


def test_hybrid_pipeline_hierarchy(mock_allergen_db):
    """Test hybrid pipeline decision basis and regulatory override."""
    class DummyMLModel:
        def predict_proba(self, X):
            return [[0.8, 0.2]]  # Predicts low allergen probability (0.2)

    engine = FAOWHOAllergenicityEngine(db=mock_allergen_db)
    pipeline = HybridAllergenPipeline(ml_model=DummyMLModel(), rule_engine=engine, operating_threshold=0.5)

    # 1. Sequence with rule hit: should override ML to 'high_risk'
    query_hit = mock_allergen_db.allergens[0]["sequence"]
    out_hit = pipeline.predict(query_hit)
    assert out_hit["final_call"]["risk_tier"] == "high_risk"
    assert out_hit["final_call"]["is_allergen"] is True
    assert out_hit["final_call"]["decision_basis"] in ["who_fao_primary", "who_fao_short_match"]
    assert out_hit["regulatory_evidence_stream"]["regulatory_flag"] == "high_risk"

    # 2. Sequence without rule hit: should defer to ML
    query_no_hit = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    out_no_hit = pipeline.predict(query_no_hit, excluded_ids={"REF_ALLERGEN_1", "REF_ALLERGEN_2"})
    assert out_no_hit["regulatory_evidence_stream"]["regulatory_flag"] == "no_homology_evidence"
    assert out_no_hit["final_call"]["decision_basis"] == "ml_only"
    assert out_no_hit["final_call"]["classification"] == "non_allergen"
    assert "provenance" in out_no_hit
    assert "db_sha256" in out_no_hit["provenance"]


def test_benchmark_acceptance_criteria_satisfied():
    """Verify generated benchmark results strictly satisfy all task acceptance criteria."""
    import json
    from pathlib import Path
    bench_file = Path(__file__).resolve().parent.parent / "results" / "hybrid_allergenicity_benchmark.json"
    assert bench_file.exists(), f"Benchmark results file missing: {bench_file}"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Reference tool parity
    val = data["metadata"]["validation_parity"]
    assert val["status"] == "PASS"
    assert val["validation_cases_count"] >= 50
    assert val["agreement_rate"] == 1.0

    # 2. Fixed specificity delta >= 0.0
    f95 = data["performance_comparison"]["fixed_95pct_specificity_comparison"]
    assert f95["exact_recall_delta"] >= 0.0, f"Expected non-negative recall delta, got {f95['exact_recall_delta']}"
    assert f95["target_specificity"] == 0.95

    # 3. Short queries handled explicitly
    short_bins = ["5-9", "10-14", "15-24", "25-49"]
    for b in short_bins:
        bin_data = data["length_bin_breakdown"][b]
        # In these bins, all sequences are < 80 aa, so all must be marked not_applicable_short_query
        assert bin_data["primary_not_applicable_count"] == bin_data["sample_count"]

