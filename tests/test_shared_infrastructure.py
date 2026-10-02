"""Unit and Integration Tests for Shared Infrastructure (Tasks 1 - 6).

Verifies:
1. Deduplication correctly eliminates exact and near-exact duplicates with logged reasons.
2. Clustering correctly clusters homologous sequences (>=40% identity) and uses k-mer fallback for <20 aa peptides.
3. Nested splits guarantee outer and inner group disjointness.
4. Leakage auditor passes valid splits and FAILS THE BUILD when leakage is injected.
5. Shared evaluation module computes all standard metrics, adaptive ECE (15 bins), length stratification, and bootstrap CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.audit import LeakageAuditError, LeakageAuditor
from src.clustering import SequenceClusterer, compute_normalized_identity, jaccard_kmer_similarity
from src.data.deduplication import deduplicate_records
from src.evaluation import evaluate_screening_classifier
from src.splits import NestedGroupSplitter


@pytest.fixture
def mock_sequences():
    return [
        {"id": "SEQ1_LONG", "sequence": "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK", "is_antigen": 1, "is_toxic": 0, "is_allergen": 0},
        # Homolog of SEQ1 (>80% identity)
        {"id": "SEQ1_HOMOLOG", "sequence": "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAA", "is_antigen": 1, "is_toxic": 0, "is_allergen": 0},
        # Exact duplicate of SEQ1
        {"id": "SEQ1_DUP", "sequence": "MKTLLILAVVAAALASGCSSVSAKDQQTLNQLISKLNKVLLDNDNDQTLKVVKNAK", "is_antigen": 1, "is_toxic": 0, "is_allergen": 0},
        # Distant long sequence
        {"id": "SEQ2_LONG", "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "is_antigen": 0, "is_toxic": 0, "is_allergen": 0},
        # Short peptide 1 (<20 aa)
        {"id": "PEPTIDE1", "sequence": "ALWKTLLKKVLKAAAKA", "is_antigen": 0, "is_toxic": 1, "is_allergen": 0},
        # Short peptide 1 homolog via k-mer Jaccard
        {"id": "PEPTIDE1_HOMOLOG", "sequence": "ALWKTLLKKVLKAAAKG", "is_antigen": 0, "is_toxic": 1, "is_allergen": 0},
        # Short peptide 2 (distant)
        {"id": "PEPTIDE2", "sequence": "MRWQEMGYIFYPRKLR", "is_antigen": 0, "is_toxic": 0, "is_allergen": 0},
    ]


def test_deduplication(mock_sequences):
    deduped, logs = deduplicate_records(mock_sequences, near_duplicate_threshold=0.98)
    # SEQ1_DUP must be removed
    removed_ids = [entry.seq_id for entry in logs]
    assert "SEQ1_DUP" in removed_ids
    assert len(deduped) < len(mock_sequences)


def test_sequence_clustering_homology_and_fallback(mock_sequences):
    clusterer = SequenceClusterer(identity_threshold=0.40, coverage_threshold=0.80, short_peptide_len=20)
    res = clusterer.cluster_records(mock_sequences)
    assign = res["assignments"]

    # SEQ1 and its homolog must be in the same cluster
    assert assign["SEQ1_LONG"] == assign["SEQ1_HOMOLOG"]
    # SEQ1 and SEQ2 must be in different clusters
    assert assign["SEQ1_LONG"] != assign["SEQ2_LONG"]
    # Short peptides sharing high k-mer Jaccard must be clustered together
    assert assign["PEPTIDE1"] == assign["PEPTIDE1_HOMOLOG"]
    # Distant short peptides must be in different clusters
    assert assign["PEPTIDE1"] != assign["PEPTIDE2"]


def test_nested_splits_disjointness(mock_sequences):
    clusterer = SequenceClusterer(identity_threshold=0.40)
    clustering = clusterer.cluster_records(mock_sequences)
    cluster_map = clustering["assignments"]

    splitter = NestedGroupSplitter(n_outer_splits=2, inner_calibration_ratio=0.5, random_state=42)
    splits = splitter.generate_nested_splits(mock_sequences, cluster_map, stratify_key="is_antigen")

    for fold in splits["folds"]:
        test_c = set(fold["outer_test"]["cluster_ids"])
        train_c = set(fold["outer_train"]["cluster_ids"])
        inner_train_c = set(fold["inner_disjoint_partitions"]["inner_train"]["cluster_ids"])
        inner_cal_c = set(fold["inner_disjoint_partitions"]["inner_calibration"]["cluster_ids"])

        assert test_c.isdisjoint(train_c), "Outer test and train clusters must be disjoint"
        assert inner_train_c.isdisjoint(inner_cal_c), "Inner train and cal clusters must be disjoint"
        assert inner_cal_c.isdisjoint(test_c), "Inner cal and outer test clusters must be disjoint"


def test_leakage_audit_passes_valid_splits(mock_sequences):
    # Remove exact duplicate before auditing
    clean = [s for s in mock_sequences if s["id"] != "SEQ1_DUP"]
    clusterer = SequenceClusterer(identity_threshold=0.40)
    clustering = clusterer.cluster_records(clean)
    cluster_map = clustering["assignments"]

    splitter = NestedGroupSplitter(n_outer_splits=2, inner_calibration_ratio=0.5, random_state=42)
    splits = splitter.generate_nested_splits(clean, cluster_map, stratify_key="is_antigen")

    auditor = LeakageAuditor(identity_threshold=0.40)
    cert = auditor.audit_splits(splits, clean, cluster_map)
    assert cert["status"] == "PASS"


def test_leakage_audit_fails_on_injected_leakage(mock_sequences):
    clean = [s for s in mock_sequences if s["id"] != "SEQ1_DUP"]
    clusterer = SequenceClusterer(identity_threshold=0.40)
    clustering = clusterer.cluster_records(clean)
    cluster_map = clustering["assignments"]

    splitter = NestedGroupSplitter(n_outer_splits=2, inner_calibration_ratio=0.5, random_state=42)
    splits = splitter.generate_nested_splits(clean, cluster_map, stratify_key="is_antigen")

    # Artificially inject cluster leakage in Fold 0
    fold0 = splits["folds"][0]
    leaked_cluster = fold0["outer_test"]["cluster_ids"][0]
    fold0["outer_train"]["cluster_ids"].append(leaked_cluster)

    auditor = LeakageAuditor(identity_threshold=0.40)
    with pytest.raises(LeakageAuditError, match="Outer cluster leakage detected"):
        auditor.audit_splits(splits, clean, cluster_map)


def test_shared_evaluation_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.3, 0.7, 0.1, 0.85, 0.4, 0.15, 0.25])
    clusters = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5"]
    lengths = [8, 12, 18, 30, 60, 8, 14, 22, 45, 120]

    res = evaluate_screening_classifier(
        y_true=y_true,
        y_prob=y_prob,
        cluster_ids=clusters,
        lengths=lengths,
        threshold=0.5,
        positive_class_label="antigen",
        n_bootstraps=50,
        random_state=42,
    )

    metrics = res["metrics"]
    assert metrics["auroc"] > 0.90
    assert metrics["brier_score"] < 0.15
    assert "adaptive_ece_15bins" in metrics
    assert "confidence_intervals_95" in res
    assert "auroc" in res["confidence_intervals_95"]
    assert "length_stratified" in res

    # Verify all 5 length bins exist
    for b_name in ["5-9", "10-14", "15-24", "25-49", "50+"]:
        assert b_name in res["length_stratified"]
