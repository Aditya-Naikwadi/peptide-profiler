"""Unit and integration tests for ESM-2 ONNX representation module.

Tests:
1. Non-canonical residue mapping (U->C, O->K, B->N, Z->E, X->X) and OOD tracking.
2. Deterministic cache key generation.
3. Residue pooling logic (mean over residue tokens excluding BOS/EOS/PAD, mean+max).
4. FP32 vs dynamic INT8 ONNX parity (cosine similarity >= 0.99).
5. Offline batch inference and dimension consistency.
"""

import numpy as np
import pytest

from src.features.esm2_onnx import (
    DEFAULT_PRIMARY_MODEL_ID,
    ESM2ONNXEmbedder,
    compute_embedding_cache_key,
    sanitize_sequence,
    verify_fp32_int8_parity,
)


def test_sanitize_sequence_canonical():
    """Canonical sequences must remain unchanged and not flag OOD."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    clean_seq, meta = sanitize_sequence(seq)
    assert clean_seq == seq
    assert not meta["has_non_canonical"]
    assert not meta["ood_flag"]
    assert meta["unknown_residue_count"] == 0
    assert len(meta["non_canonical_counts"]) == 0


def test_sanitize_sequence_non_canonical_mapping():
    """Non-canonical amino acids (U, O, B, Z) must map properly and flag OOD."""
    # U -> C (Selenocysteine), O -> K (Pyrrolysine), B -> N (Asx), Z -> E (Glx)
    seq = "AUOCBZ"
    clean_seq, meta = sanitize_sequence(seq)
    assert clean_seq == "ACKCNE"
    assert meta["has_non_canonical"]
    assert meta["ood_flag"]
    assert meta["non_canonical_counts"]["U"] == 1
    assert meta["non_canonical_counts"]["O"] == 1
    assert meta["non_canonical_counts"]["B"] == 1
    assert meta["non_canonical_counts"]["Z"] == 1


def test_sanitize_sequence_unknown_residues():
    """Unknown residues must map to X and set OOD risk flag."""
    seq = "ACDXG"
    clean_seq, meta = sanitize_sequence(seq)
    assert clean_seq == "ACDXG"
    assert meta["ood_flag"]


def test_compute_cache_key_determinism():
    """Cache key must be deterministic and sensitive to model, revision, and pooling."""
    seq = "ACDEFGHIK"
    key1 = compute_embedding_cache_key(seq, pooling="mean", quantization="int8")
    key2 = compute_embedding_cache_key(seq, pooling="mean", quantization="int8")
    key_max = compute_embedding_cache_key(seq, pooling="mean_max", quantization="int8")
    key_fp32 = compute_embedding_cache_key(seq, pooling="mean", quantization="fp32")

    assert key1 == key2
    assert key1 != key_max
    assert key1 != key_fp32


def test_esm2_onnx_embedding_extraction():
    """Test offline ONNX INT8 embedding extraction with mean and mean_max pooling."""
    embedder = ESM2ONNXEmbedder(
        model_id=DEFAULT_PRIMARY_MODEL_ID,
        quantization="int8",
        onnx_dir="models/onnx",
    )

    test_seq = "ACDEFGHIKLMNPQRSTVWY"

    # Mean pooling (320-D)
    vec_mean, meta_mean = embedder.extract_embedding(test_seq, pooling="mean")
    assert vec_mean.shape == (320,)
    assert not np.isnan(vec_mean).any()
    assert not meta_mean["ood_flag"]

    # Mean+Max pooling (640-D)
    vec_meanmax, meta_mm = embedder.extract_embedding(test_seq, pooling="mean_max")
    assert vec_meanmax.shape == (640,)
    assert not np.isnan(vec_meanmax).any()

    # First 320 dims of mean_max must equal mean pooling
    np.testing.assert_allclose(vec_meanmax[:320], vec_mean, rtol=1e-5, atol=1e-5)


def test_esm2_fp32_int8_parity():
    """Parity check: FP32 vs INT8 mean cosine similarity must be >= 0.99."""
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "WVGSLKLRRCCHLSPRSKL",
        "KLRRCCHLSPRSKLTTWK",
        "MRLLALSGLLCMLLLCFCIF",
        "GHIKLMNPQRSTVWYACDEF",
    ]
    parity = verify_fp32_int8_parity(
        sequences=sequences,
        model_id=DEFAULT_PRIMARY_MODEL_ID,
        onnx_dir="models/onnx",
        tolerance_cosine=0.99,
    )

    assert parity["parity_passed"]
    assert parity["mean_cosine_similarity"] >= 0.99
    assert parity["min_cosine_similarity"] >= 0.985
