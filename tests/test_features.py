"""Unit and feature-parity test suite for versioned feature extractors."""

import numpy as np
import pytest

from src.features.aac import calculate_aac_vector
from src.features.acc import calculate_acc_features
from src.features.dipeptide import calculate_dpc_vector
from src.features.kmer import calculate_kmer_features
from src.features.pcp import calculate_pcp_vector
from src.features.plm import get_plm_feasibility_report
from src.physicochem import calculate_aac as legacy_calc_aac, calculate_pcp_descriptors as legacy_calc_pcp


GOLDEN_SEQUENCES = {
    "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "conotoxin": "CKGKGAKCSRLMYDCCTGSCRSGKC",
}


def test_aac_dimensions_and_sum():
    """Verify 20-D AAC vector structure and 100% sum."""
    for name, seq in GOLDEN_SEQUENCES.items():
        vec = calculate_aac_vector(seq)
        assert vec.shape == (20,), f"Expected 20-D for {name}"
        assert vec.dtype == np.float32
        assert np.sum(vec) == pytest.approx(100.0, rel=1e-2)


def test_dpc_dimensions_and_sum():
    """Verify 400-D Dipeptide Composition structure."""
    for name, seq in GOLDEN_SEQUENCES.items():
        vec = calculate_dpc_vector(seq)
        assert vec.shape == (400,), f"Expected 400-D for {name}"
        assert vec.dtype == np.float32
        assert np.sum(vec) == pytest.approx(100.0, rel=1e-2)


def test_kmer_features():
    """Verify k-mer feature extraction for k=1 and k=2."""
    seq = GOLDEN_SEQUENCES["melittin"]
    k1 = calculate_kmer_features(seq, k=1, as_frequency=True)
    assert k1.shape == (20,)
    assert np.sum(k1) == pytest.approx(1.0, rel=1e-3)

    k2 = calculate_kmer_features(seq, k=2, as_frequency=True)
    assert k2.shape == (400,)
    assert np.sum(k2) == pytest.approx(1.0, rel=1e-3)


def test_acc_dimensions_and_properties():
    """Verify Auto-Cross Covariance (ACC) dimensions with 5 Z-scales and lag 5."""
    for name, seq in GOLDEN_SEQUENCES.items():
        acc_full = calculate_acc_features(seq, max_lag=5, include_cross=True)
        # 5 auto * 5 lags + 20 cross * 5 lags = 25 + 100 = 125 dimensions
        assert acc_full.shape == (125,), f"Expected 125-D ACC for {name}"
        assert not np.isnan(acc_full).any()

        acc_auto = calculate_acc_features(seq, max_lag=5, include_cross=False)
        # 5 auto * 5 lags = 25 dimensions
        assert acc_auto.shape == (25,)


def test_pcp_dimensions():
    """Verify 30-D PCP descriptors."""
    for name, seq in GOLDEN_SEQUENCES.items():
        pcp = calculate_pcp_vector(seq)
        assert pcp.shape == (30,), f"Expected 30-D PCP for {name}"
        assert pcp.dtype == np.float32


def test_feature_parity_aac():
    """Verify exact numerical parity between legacy AAC and new versioned AAC."""
    for name, seq in GOLDEN_SEQUENCES.items():
        legacy_aac = legacy_calc_aac(seq)
        new_vec = calculate_aac_vector(seq)
        standard_aa = list("ACDEFGHIKLMNPQRSTVWY")
        for idx, aa in enumerate(standard_aa):
            assert legacy_aac[aa] == pytest.approx(float(new_vec[idx]), abs=1e-3)


def test_feature_parity_pcp():
    """Verify exact numerical parity between legacy PCP and new versioned PCP."""
    for name, seq in GOLDEN_SEQUENCES.items():
        legacy_pcp = legacy_calc_pcp(seq)
        new_vec = calculate_pcp_vector(seq)
        legacy_vals = list(legacy_pcp.values())
        for idx, val in enumerate(legacy_vals):
            assert val == pytest.approx(float(new_vec[idx]), abs=1e-3)


def test_plm_feasibility_report():
    """Verify PLM feasibility report provides required architectural comparisons."""
    rep = get_plm_feasibility_report()
    assert "esm2_t6_8M_UR50D" in rep["architecture_comparison"]
    assert "acc_pcp_classical_v2" in rep["architecture_comparison"]
    assert "decision_rationale" in rep
