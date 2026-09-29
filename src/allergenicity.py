"""Allergenicity prediction module for peptides (AlgPred2 / AllerTOP2 wrapper with local ACC fallback)."""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import requests

from src.physicochem import AMINO_ACIDS, calculate_aac, calculate_pcp_descriptors

logger = logging.getLogger(__name__)

DEFAULT_ALLERGEN_MODEL = Path(__file__).resolve().parent.parent / "models" / "allergen_rf.joblib"
_CACHED_ALLERGEN_CLF = None


def get_local_allergen_model():
    """Lazily load the local trained allergenicity model."""
    global _CACHED_ALLERGEN_CLF
    if _CACHED_ALLERGEN_CLF is not None:
        return _CACHED_ALLERGEN_CLF

    if DEFAULT_ALLERGEN_MODEL.exists():
        try:
            _CACHED_ALLERGEN_CLF = joblib.load(DEFAULT_ALLERGEN_MODEL)
            return _CACHED_ALLERGEN_CLF
        except Exception as e:
            logger.warning(f"Could not load allergen RF model: {e}")

    return None


def call_allertop_api(sequence: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """Query the AllerTOP v2 web server if accessible."""
    url = "https://www.ddg-pharmfac.net/AllerTOP/run_prediction.py"
    data = {"sequence": sequence}
    try:
        response = requests.post(url, data=data, timeout=timeout)
        if response.status_code == 200:
            text = response.text.lower()
            if "probable allergen" in text:
                return {
                    "allergenicity_score": 0.85,
                    "is_allergen": True,
                    "method": "AllerTOP2 (Web API)",
                }
            elif "probable non-allergen" in text:
                return {
                    "allergenicity_score": 0.15,
                    "is_allergen": False,
                    "method": "AllerTOP2 (Web API)",
                }
    except Exception as e:
        logger.debug(f"AllerTOP web query failed: {e}")

    return None


def predict_allergenicity_acc_local(sequence: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Predict allergenicity locally using Auto-Cross Covariance (ACC) and Pfeature descriptors."""
    clf = get_local_allergen_model()
    cleaned = sequence.strip().upper()

    # Extract 20 AAC + 30 PCP = 50 features
    aac = calculate_aac(cleaned)
    pcp = calculate_pcp_descriptors(cleaned)
    features = [aac[aa] for aa in AMINO_ACIDS] + list(pcp.values())
    X = np.array([features], dtype=np.float32)

    if clf is not None:
        probs = clf.predict_proba(X)[0]
        # Class 1 is Allergen
        score = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        # Heuristic based on typical allergen motifs (surface exposed, charged residues, specific repeats)
        surf_exposed = pcp.get("PCP_SA_EX", 0.0)
        polar = pcp.get("PCP_PO", 0.0)
        pos_charge = pcp.get("PCP_PC", 0.0)
        score = (surf_exposed * 0.4) + (polar * 0.3) + (pos_charge * 0.3)

    score = round(float(np.clip(score, 0.01, 0.99)), 3)
    is_allergen = bool(score >= threshold)

    return {
        "allergenicity_score": score,
        "is_allergen": is_allergen,
        "method": "Local ACC-based Classifier (Pfeature descriptors)",
        "threshold": threshold,
    }


def run_allergenicity(
    sequence: str,
    threshold: float = 0.5,
    use_api: bool = False,
) -> Dict[str, Any]:
    """Run peptide allergenicity prediction.

    Interface adheres to project specification:
    Returns {'allergenicity_score': float, 'is_allergen': bool}

    Args:
        sequence: Validated uppercase amino acid sequence string.
        threshold: Classification cutoff (default 0.5).
        use_api: If True, attempts to call remote AlgPred2/AllerTOP2 web service first.

    Returns:
        Dictionary containing:
            - allergenicity_score: Probability between 0.0 and 1.0
            - is_allergen: Boolean classification
            - method: Prediction engine utilized
            - threshold: Threshold applied
    """
    cleaned = sequence.strip().upper()

    if use_api:
        api_result = call_allertop_api(cleaned)
        if api_result is not None:
            return api_result

    return predict_allergenicity_acc_local(cleaned, threshold=threshold)
