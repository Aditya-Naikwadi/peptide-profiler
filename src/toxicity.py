"""Toxicity prediction module for peptides using ToxinPred2."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Model paths to check in priority order
DEFAULT_MODEL_PATHS = [
    Path(__file__).resolve().parent.parent / "models" / "toxinpred2_rf.onnx",
    Path(sys.prefix) / "Lib" / "site-packages" / "toxinpred2" / "model" / "RF_model.onnx",
]

_CACHED_SESSION = None


def get_onnx_session():
    """Lazily load and cache the ONNX inference session for ToxinPred2."""
    global _CACHED_SESSION
    if _CACHED_SESSION is not None:
        return _CACHED_SESSION

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime is not installed. ONNX direct inference unavailable.")
        return None

    model_path = None
    for p in DEFAULT_MODEL_PATHS:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        # Search anywhere under site-packages for RF_model.onnx
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
        found = list(site_packages.glob("**/RF_model.onnx"))
        if found:
            model_path = found[0]

    if model_path is None or not model_path.exists():
        logger.warning("ToxinPred2 ONNX model file not found.")
        return None

    # Silence verbose onnxruntime logging
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    _CACHED_SESSION = ort.InferenceSession(str(model_path), sess_options=opts)
    return _CACHED_SESSION


def calculate_aac_features(sequence: str) -> np.ndarray:
    """Calculate amino acid composition percentage vector for standard 20 amino acids."""
    length = len(sequence)
    if length == 0:
        return np.zeros((1, 20), dtype=np.float32)
    vector = [(sequence.count(aa) / length) * 100.0 for aa in AMINO_ACIDS]
    return np.array([vector], dtype=np.float32)


def predict_toxicity_onnx(sequence: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
    """Predict toxicity using the native ToxinPred2 Random Forest ONNX model."""
    sess = get_onnx_session()
    if sess is None:
        return None

    aac = calculate_aac_features(sequence)
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]

    # Run inference
    res = sess.run(outputs, {input_name: aac})
    # res[0]: predicted labels array, e.g. [1] or [0]
    # res[1]: list of class probability maps, e.g. [{0: 0.28, 1: 0.72}]
    prob_dict = res[1][0]
    toxin_prob = float(prob_dict.get(1, 0.0))

    score = round(toxin_prob, 3)
    is_toxic = bool(score >= threshold)

    return {
        "toxicity_score": score,
        "is_toxic": is_toxic,
        "method": "ToxinPred2 (ONNX RF Model)",
        "threshold": threshold,
    }


def predict_toxicity_heuristic_fallback(sequence: str, threshold: float = 0.6) -> Dict[str, Any]:
    """Heuristic fallback based on known toxin motifs, charge, and hydrophobic amphipathicity."""
    cleaned = sequence.upper()
    length = len(cleaned)

    # Known toxic peptide characteristics (melittin, mastoparan, conotoxins, venom peptides):
    # - High basicity / positive charge (Lys, Arg)
    # - High hydrophobic moment / tryptophan/leucine enrichment
    # - Cysteine richness (disulfide bridges common in peptide toxins)
    cys_count = cleaned.count("C")
    cys_ratio = cys_count / length
    pos_charge = (cleaned.count("K") + cleaned.count("R")) / length
    hydrophobes = sum(cleaned.count(aa) for aa in "ILVFW") / length

    # Weighted estimate
    score = (pos_charge * 0.4) + (hydrophobes * 0.3) + min(1.0, cys_ratio * 3.0) * 0.3
    score = round(float(np.clip(score, 0.02, 0.98)), 3)

    return {
        "toxicity_score": score,
        "is_toxic": bool(score >= threshold),
        "method": "Physicochemical Heuristic Fallback",
        "threshold": threshold,
    }


def run_toxicity(sequence: str, threshold: float = 0.6, force_fallback: bool = False) -> Dict[str, Any]:
    """Run peptide toxicity prediction.

    Interface adheres to project specification:
    Returns {'toxicity_score': float, 'is_toxic': bool}

    Args:
        sequence: Validated uppercase amino acid sequence string.
        threshold: Decision threshold (default 0.6 per ToxinPred2).
        force_fallback: If True, skips ONNX model and uses heuristic fallback.

    Returns:
        Dictionary with:
            - toxicity_score: Probability between 0.0 and 1.0
            - is_toxic: Boolean classification
            - method: The model used
            - threshold: The threshold used
    """
    cleaned = sequence.strip().upper()

    if not force_fallback:
        res = predict_toxicity_onnx(cleaned, threshold=threshold)
        if res is not None:
            return res

    return predict_toxicity_heuristic_fallback(cleaned, threshold=threshold)
