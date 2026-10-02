"""CI Regression, Leakage Audit, Golden Dataset, and Latency Verification Suite.

Validates:
1. Leakage Audit: Zero cluster overlap between training, calibration, and test partitions.
2. Golden Dataset Regression: Validates exact output schema and deterministic values on reference peptides.
3. Latency Check: Asserts average per-peptide CPU inference latency remains within production budget (<100 ms).
4. CLI End-to-End Contract: Verifies CLI produces the exact required single-output schema.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.api import profile_batch, profile_peptide, serialize_to_standard_contract
from src.cli import main as cli_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reference Golden Regression Dataset
GOLDEN_PEPTIDES = [
    {
        "id": "melittin_ref",
        "name": "Melittin (Honeybee venom)",
        "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "expected_toxic": True,
        "expected_advisory": False,
    },
    {
        "id": "ubiquitin_nterm",
        "name": "Ubiquitin N-term 26aa",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENV",
        "expected_toxic": False,
        "expected_advisory": False,
    },
    {
        "id": "short_peptide_12aa",
        "name": "Short Test Lead 12aa",
        "sequence": "ALWKTLLKKVLK",
        "expected_advisory": True,  # Short peptide <20 aa advisory flag
    },
]


def test_nested_splits_leakage_audit():
    """Verify that zero homology clusters or sequence IDs leak between CV partitions."""
    splits_file = PROJECT_ROOT / "data" / "nested_group_splits.json"
    cluster_file = PROJECT_ROOT / "data" / "cluster_assignments_40pct.json"
    records_file = PROJECT_ROOT / "data" / "curated_deduplicated_dataset.json"

    assert splits_file.exists(), "nested_group_splits.json must exist"
    assert cluster_file.exists(), "cluster_assignments_40pct.json must exist"
    assert records_file.exists(), "curated_deduplicated_dataset.json must exist"

    splits_data = json.loads(splits_file.read_text(encoding="utf-8"))
    clusters_map = json.loads(cluster_file.read_text(encoding="utf-8"))["assignments"]
    records = json.loads(records_file.read_text(encoding="utf-8"))

    for fold in splits_data["folds"]:
        fold_id = fold["outer_fold_id"]
        outer_train_ids = set(fold["outer_train"]["sequence_ids"])
        outer_test_ids = set(fold["outer_test"]["sequence_ids"])

        # 1. Exact sequence ID disjointness
        overlap_ids = outer_train_ids.intersection(outer_test_ids)
        assert len(overlap_ids) == 0, f"Fold {fold_id}: Sequence leakage! Overlap IDs: {overlap_ids}"

        # 2. Cluster level disjointness in outer CV
        train_clusters = set(fold["outer_train"]["cluster_ids"])
        test_clusters = set(fold["outer_test"]["cluster_ids"])
        assert train_clusters.isdisjoint(test_clusters), f"Fold {fold_id}: Homology cluster leakage! {train_clusters & test_clusters}"

        # 3. Inner partition disjointness
        inner_train_ids = set(fold["inner_disjoint_partitions"]["inner_train"]["sequence_ids"])
        inner_cal_ids = set(fold["inner_disjoint_partitions"]["inner_calibration"]["sequence_ids"])

        inner_overlap = inner_train_ids.intersection(inner_cal_ids)
        assert len(inner_overlap) == 0, f"Fold {fold_id}: Inner calibration leakage! Overlap: {inner_overlap}"

        inner_train_clusters = set(fold["inner_disjoint_partitions"]["inner_train"]["cluster_ids"])
        inner_cal_clusters = set(fold["inner_disjoint_partitions"]["inner_calibration"]["cluster_ids"])
        assert inner_train_clusters.isdisjoint(inner_cal_clusters), f"Fold {fold_id}: Inner cluster leakage! {inner_train_clusters & inner_cal_clusters}"

    # 4. Strict LeakageAuditor cross-partition identity verification
    from src.audit import LeakageAuditor
    auditor = LeakageAuditor(identity_threshold=0.40)
    cert = auditor.audit_splits(splits_data, records, clusters_map)
    assert cert["status"] == "PASS", f"Leakage auditor failed: {cert}"


def test_golden_dataset_schema_and_integrity():
    """Verify golden dataset outputs adhere strictly to the specified contract schema."""
    for ref in GOLDEN_PEPTIDES:
        result = profile_peptide(
            sequence=ref["sequence"],
            profile="vaccine",
            target_prevalence=0.02,
            seq_id=ref["id"],
        )

        # Root-level contract schema verification
        expected_root_keys = {
            "sequence", "status", "antigenicity", "toxicity",
            "allergenicity", "stability", "pareto", "profile", "versions"
        }
        assert set(result.keys()) == expected_root_keys, f"Schema mismatch for {ref['id']}: {set(result.keys())}"

        assert result["sequence"] == ref["sequence"]
        assert result["status"] in ["ok", "abstain"]
        assert result["profile"] == "vaccine"

        # Antigenicity sub-schema
        ant = result["antigenicity"]
        assert set(ant.keys()) == {"p_cal", "p_prior_adj", "pred_set", "decision"}
        assert 0.0 <= ant["p_cal"] <= 1.0
        assert 0.0 <= ant["p_prior_adj"] <= 1.0
        assert isinstance(ant["pred_set"], list)
        assert ant["decision"] in ["antigen", "non_antigen"]

        # Toxicity sub-schema
        tox = result["toxicity"]
        assert set(tox.keys()) == {"p_cal", "p_prior_adj", "pred_set", "decision"}
        assert 0.0 <= tox["p_cal"] <= 1.0
        assert 0.0 <= tox["p_prior_adj"] <= 1.0
        assert isinstance(tox["pred_set"], list)
        assert tox["decision"] in ["toxic", "non_toxic"]

        # Allergenicity sub-schema
        alg = result["allergenicity"]
        assert set(alg.keys()) == {"who_fao", "ml", "decision_basis"}
        assert isinstance(alg["who_fao"]["hit"], bool)
        assert isinstance(alg["who_fao"]["details"], str)
        assert set(alg["ml"].keys()) == {"p_cal", "p_prior_adj", "pred_set"}
        assert alg["decision_basis"] in ["who_fao_regulatory_hit", "ml_probability", "consensus"]

        # Stability sub-schema
        stab = result["stability"]
        assert set(stab.keys()) == {"flags", "advisory"}
        assert isinstance(stab["flags"], list)
        assert isinstance(stab["advisory"], bool)
        assert stab["advisory"] == ref["expected_advisory"]

        # Pareto sub-schema
        pareto = result["pareto"]
        assert set(pareto.keys()) == {"front", "crowding"}
        assert isinstance(pareto["front"], int) and pareto["front"] >= 1
        assert isinstance(pareto["crowding"], float)

        # Versions metadata
        versions = result["versions"]
        assert set(versions.keys()) == {"models", "allergen_db", "esm"}


def test_batch_pareto_ranking_contract():
    """Verify batch profiling with multi-objective Pareto ranking satisfies the contract."""
    records = [
        {"id": ref["id"], "sequence": ref["sequence"]}
        for ref in GOLDEN_PEPTIDES
    ]
    batch_results = profile_batch(
        records=records,
        profile="therapeutic",
        target_prevalence=0.01,
        rank_candidates=True,
    )

    assert len(batch_results) == len(records)
    for res in batch_results:
        assert res["profile"] == "therapeutic"
        assert res["pareto"]["front"] >= 1
        assert "versions" in res


def test_cpu_latency_budget():
    """Verify average CPU profiling latency remains strictly within production budget (<100 ms per peptide)."""
    test_seq = "ALWKTLLKKVLKAAAKA"

    # Warmup
    _ = profile_peptide(test_seq, profile="vaccine")

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = profile_peptide(test_seq, profile="vaccine")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))
    # Production budget: under 100ms per peptide on CPU
    assert mean_latency_ms < 100.0, f"Latency check failed: {mean_latency_ms:.2f} ms > 100.0 ms"


def test_cli_standard_contract_execution(tmp_path):
    """Verify CLI produces the single output schema when invoked with --standard-contract."""
    out_file = tmp_path / "cli_golden.json"
    cli_main([
        "-i", "MQIFVKTLTGKTITLEVEPSDTIENV",
        "-o", str(out_file),
        "--profile", "therapeutic",
        "--target-prevalence", "0.01",
        "--standard-contract",
        "--format", "json",
    ])

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))

    # Single peptide should serialize as the single dictionary contract
    assert isinstance(data, dict)
    assert data["sequence"] == "MQIFVKTLTGKTITLEVEPSDTIENV"
    assert data["profile"] == "therapeutic"
    assert "antigenicity" in data
    assert "toxicity" in data
    assert "allergenicity" in data
    assert "stability" in data
    assert "pareto" in data
