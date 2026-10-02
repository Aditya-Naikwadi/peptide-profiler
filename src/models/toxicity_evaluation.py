"""Toxicity Modeling & Evaluation Module (WP6.2).

Compares:
1. Upstream ONNX baseline (ToxinPred2 RF model).
2. In-house trained model across 50-D AAC+PCP and 125-D ACC features.
3. Group-held-out (StratifiedGroupKFold on clusters) and Temporal splits.

Positive class = Toxic peptide/protein.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.acc import calculate_acc_features
from src.features.pcp import calculate_pcp_vector
from src.features.aac import calculate_aac_vector
from src.toxicity import run_toxicity

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def extract_toxicity_features_50d(sequence: str) -> np.ndarray:
    """Extract 20 AAC + 30 PCP = 50-D features."""
    clean = sequence.strip().upper()
    aac = calculate_aac_vector(clean)
    pcp = calculate_pcp_vector(clean)
    return np.hstack([aac, pcp])


def build_inhouse_toxicity_model(random_state: int = RANDOM_SEED) -> Any:
    """Build tuned in-house Random Forest model for toxicity prediction."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def predict_onnx_toxicity_batch(sequences: List[str]) -> np.ndarray:
    """Run batch inference using upstream ToxinPred2 ONNX model."""
    probs = []
    for seq in sequences:
        res = run_toxicity(seq)
        probs.append(res["toxicity_score"])
    return np.array(probs, dtype=np.float64)
