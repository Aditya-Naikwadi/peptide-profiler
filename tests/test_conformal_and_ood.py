"""Tests for WP9.3: Conformal Prediction & Applicability Domain (OOD) Guardrails."""

import json
import numpy as np
import pytest

from src.models.conformal import MondrianConformalClassifier
from src.models.ood import (
    ApplicabilityDomainDetector,
    check_chemical_modifications,
    compute_sequence_entropy,
    evaluate_rule_guards,
)


# =========================================================================
# 1. CONFORMAL PREDICTION (MONDRIAN & STATUS CONTRACTS)
# =========================================================================

def test_mondrian_conformal_fit_and_finite_sample_quantiles():
    """Verify Mondrian conformal computes valid class-conditional quantiles."""
    np.random.seed(42)
    y_cal = np.array([0] * 80 + [1] * 20)
    # Well-separated calibrated probabilities
    prob_cal = np.concatenate([
        np.random.beta(1, 5, size=80),  # non-toxic (class 0, low prob)
        np.random.beta(5, 1, size=20),  # toxic (class 1, high prob)
    ])

    clf = MondrianConformalClassifier(alpha=0.05, class_labels=("Non-Toxic", "Toxic"))
    clf.fit(y_cal, prob_cal)

    assert clf.is_fitted
    assert 0 in clf.quantiles
    assert 1 in clf.quantiles
    # Quantiles should be strictly between 0 and 1
    assert 0.0 < clf.quantiles[0] <= 1.0
    assert 0.0 < clf.quantiles[1] <= 1.0


def test_mondrian_prediction_set_statuses():
    """Verify exact status mapping for singleton, uncertain ({0, 1}), and empty ({}) sets."""
    clf = MondrianConformalClassifier(alpha=0.05, class_labels=("Non-Toxic", "Toxic"))
    # Manually configure quantiles for deterministic status testing:
    # class 0 included if prob <= 0.40
    # class 1 included if (1 - prob) <= 0.40 => prob >= 0.60
    clf.quantiles = {0: 0.40, 1: 0.40}
    clf.cal_counts = {0: 50, 1: 50}
    clf.is_fitted = True

    # Case 1: Confident Non-Toxic (prob = 0.10 <= 0.40, 1 - 0.10 = 0.90 > 0.40)
    res_0 = clf.predict_set(0.10)
    assert res_0["status"] == "classified"
    assert res_0["prediction_set"] == ["Non-Toxic"]
    assert res_0["is_uncertain"] is False

    # Case 2: Confident Toxic (prob = 0.90 > 0.40, 1 - 0.90 = 0.10 <= 0.40)
    res_1 = clf.predict_set(0.90)
    assert res_1["status"] == "classified"
    assert res_1["prediction_set"] == ["Toxic"]
    assert res_1["is_uncertain"] is False

    # Case 3: Empty set anomaly (prob = 0.50 > 0.40 and 1 - 0.50 = 0.50 > 0.40)
    res_empty = clf.predict_set(0.50)
    assert res_empty["status"] == "abstain"
    assert res_empty["reason"] == "conformal_empty_set_anomaly"
    assert res_empty["prediction_set"] == []
    assert res_empty["is_uncertain"] is True

    # Case 4: Uncertain two-class set (broad quantiles)
    clf.quantiles = {0: 0.70, 1: 0.70}
    res_both = clf.predict_set(0.50)
    assert res_both["status"] == "uncertain"
    assert res_both["reason"] == "conformal_prediction_set_uncertain"
    assert res_both["prediction_set"] == ["Non-Toxic", "Toxic"]
    assert res_both["is_uncertain"] is True


def test_mondrian_protects_minority_class_coverage():
    """Verify that Mondrian conditioning guarantees target coverage on minority class."""
    np.random.seed(42)
    # Severe 9:1 class imbalance
    y_cal = np.array([0] * 180 + [1] * 20)
    prob_cal = np.concatenate([
        np.random.uniform(0.0, 0.4, size=180),
        np.random.uniform(0.3, 0.9, size=20),
    ])

    clf = MondrianConformalClassifier(alpha=0.10, class_labels=("Non-Toxic", "Toxic"))
    clf.fit(y_cal, prob_cal)

    # Test on synthetic validation set with same distribution
    y_test = np.array([0] * 450 + [1] * 50)
    prob_test = np.concatenate([
        np.random.uniform(0.0, 0.4, size=450),
        np.random.uniform(0.3, 0.9, size=50),
    ])

    eval_res = clf.evaluate_coverage(y_test, prob_test)
    assert eval_res["target_coverage"] == 0.90
    assert eval_res["class_0_coverage"] >= 0.85
    assert eval_res["class_1_coverage"] >= 0.80  # Minority class protected


# =========================================================================
# 2. APPLICABILITY DOMAIN & OOD DISTANCE METRICS
# =========================================================================

def test_mahalanobis_tied_shrinkage_covariance():
    """Verify class-conditional Mahalanobis distance increases with distance from manifold."""
    np.random.seed(42)
    # 2D features for two well-separated classes
    X_0 = np.random.normal(loc=[-2.0, -2.0], scale=0.5, size=(50, 2))
    X_1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(50, 2))
    X_train = np.vstack([X_0, X_1])
    y_train = np.array([0] * 50 + [1] * 50)

    detector = ApplicabilityDomainDetector(percentile=95.0, knn_k=3, space_name="test_2d")
    detector.fit(X_train, y_train, X_cal=X_train)

    assert detector.is_fitted
    assert detector.tied_precision is not None

    # Points near centroids should have low distance
    near_0 = detector.compute_mahalanobis(np.array([[-2.0, -2.0]]))
    near_1 = detector.compute_mahalanobis(np.array([[2.0, 2.0]]))
    far_ood = detector.compute_mahalanobis(np.array([[25.0, 25.0]]))

    assert near_0[0] < 1.0
    assert near_1[0] < 1.0
    assert far_ood[0] > 10.0


def test_cosine_knn_distance():
    """Verify Cosine kNN correctly identifies far vectors."""
    X_train = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 1.0, 0.0],
    ])
    y_train = np.array([0, 0, 0, 1])

    detector = ApplicabilityDomainDetector(percentile=99.0, knn_k=2, space_name="test_3d")
    detector.fit(X_train, y_train, X_cal=X_train)

    in_domain = np.array([[1.0, 0.05, 0.0]])
    orthogonal_ood = np.array([[0.0, 0.0, 1.0]])

    knn_id = detector.compute_knn_cosine(in_domain)
    knn_ood = detector.compute_knn_cosine(orthogonal_ood)

    assert knn_id[0] < 0.05
    assert np.isclose(knn_ood[0], 1.0, atol=1e-2)


# =========================================================================
# 3. RULE-BASED BIOLOGICAL GUARDS
# =========================================================================

def test_chemical_modification_detection():
    """Verify chemical modification annotations trigger explicit abstention."""
    # 1. Sequence-encoded modifications
    assert check_chemical_modifications("Ac-AGLF-NH2") is not None
    assert check_chemical_modifications("cyclo(RGDfV)") is not None
    assert check_chemical_modifications("ACDEFGH[pSer]IKLMN") is not None
    assert check_chemical_modifications("D-Ala-D-Leu") is not None

    # 2. Metadata-encoded modifications
    meta_mod = {"modifications": "phospho_serine"}
    assert check_chemical_modifications("ACDEFGHIKLMN", metadata=meta_mod) is not None

    meta_cyclic = {"is_cyclic": True}
    assert check_chemical_modifications("ACDEFGHIKLMN", metadata=meta_cyclic) is not None

    # 3. Standard clean peptide
    assert check_chemical_modifications("ACDEFGHIKLMNPQRSTVWY") is None


def test_sequence_length_and_composition_guards():
    """Verify length, entropy, and single-residue composition guards."""
    # Under minimum length (< 5)
    assert evaluate_rule_guards("ACD") == "length_under_minimum_5"

    # Extreme length (> 1000)
    assert evaluate_rule_guards("A" * 1005) == "length_over_maximum_1000"

    # Extreme low complexity / homopolymer (Shannon entropy < 1.5 bits)
    entropy_a = compute_sequence_entropy("AAAAAAAAAAAAAAAAAAAA")
    assert entropy_a == 0.0
    guard_homo = evaluate_rule_guards("AAAAAAAAAAAAAAAAAAAA")
    assert "extreme_low_complexity" in guard_homo

    # Skewed single residue composition (> 50%)
    # e.g., 12 Glycines in a 20-mer = 60%
    skewed_seq = "GGGGGGGGGGGGACDEFHIK"
    guard_skew = evaluate_rule_guards(skewed_seq)
    assert "extreme_skewed_composition" in guard_skew


# =========================================================================
# 4. OUTPUT CONTRACT COMPLIANCE
# =========================================================================

def test_ood_output_contract_format():
    """Verify strict JSON output contract conforms to specification."""
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(40, 5))
    y_train = np.array([0] * 20 + [1] * 20)

    detector = ApplicabilityDomainDetector(percentile=95.0, knn_k=3, space_name="contract_test")
    detector.fit(X_train, y_train, X_cal=X_train)

    # 1. In-domain query (close to centroid of class 0)
    in_vec = X_train[0].copy()
    clean_seq = "ACDEFGHIKLMNPQRSTVWY"
    res_in = detector.inspect(clean_seq, in_vec)

    assert res_in["status"] == "in_domain"
    assert res_in["reason"] is None
    assert "mahalanobis" in res_in["details"]
    assert "threshold" in res_in["details"]
    assert res_in["details"]["guard"] is None

    # 2. Guard abstention
    res_guard = detector.inspect("cyclo(RGDfV)", in_vec)
    assert res_guard["status"] == "abstain"
    assert res_guard["reason"] == "out_of_applicability_domain"
    assert res_guard["details"]["guard"] is not None

    # 3. Far distance abstention
    far_vec = np.ones(5) * 50.0
    res_far = detector.inspect(clean_seq, far_vec)
    assert res_far["status"] == "abstain"
    assert res_far["reason"] == "out_of_applicability_domain"
    assert res_far["details"]["mahalanobis"] > res_far["details"]["threshold"]
    assert res_far["details"]["guard"] is None
