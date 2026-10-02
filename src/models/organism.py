"""Organism-specific antigenicity modeling with strict abstention guardrails.

Supports:
- Bacterial ('bacteria', 'bacterial')
- Viral ('virus', 'viral')
- Tumor ('tumor', 'cancer', 'human_tumor')

Abstains cleanly for unsupported organisms or unknown taxons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_ORGANISMS_MAP = {
    "bacteria": "bacteria",
    "bacterial": "bacteria",
    "virus": "virus",
    "viral": "virus",
    "tumor": "tumor",
    "cancer": "tumor",
    "human_tumor": "tumor",
}


class UnsupportedOrganismError(ValueError):
    """Raised when an organism is not supported by the antigenicity pipeline."""
    pass


def normalize_organism_type(organism_type: Optional[str]) -> str:
    """Normalize and validate organism type against supported classes.

    Raises:
        UnsupportedOrganismError: If organism is unsupported or None.
    """
    if not organism_type or not isinstance(organism_type, str):
        raise UnsupportedOrganismError("Organism type is required and cannot be empty.")

    cleaned = organism_type.strip().lower()
    if cleaned in SUPPORTED_ORGANISMS_MAP:
        return SUPPORTED_ORGANISMS_MAP[cleaned]

    # Check for substring matches (e.g. 'bacterial pathogen', 'influenza virus')
    if "bacter" in cleaned:
        return "bacteria"
    elif "vir" in cleaned:
        return "virus"
    elif "tumor" in cleaned or "cancer" in cleaned:
        return "tumor"

    raise UnsupportedOrganismError(
        f"Unsupported organism '{organism_type}'. Supported classes: 'bacteria', 'virus', 'tumor'."
    )


class OrganismSpecificAntigenicityManager:
    """Manages organism-specific antigenicity models with explicit abstention."""

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {
            "bacteria": 0.50,
            "virus": 0.40,
            "tumor": 0.50,
        }

    def register_model(self, organism: str, model: Any, threshold: Optional[float] = None) -> None:
        """Register a trained model for an organism."""
        canon = normalize_organism_type(organism)
        self.models[canon] = model
        if threshold is not None:
            self.thresholds[canon] = threshold

    def predict_sequence(
        self,
        sequence: str,
        organism_type: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict antigenicity with abstention guard for unsupported organisms.

        Returns:
            Dictionary with score, classification, and abstention status.
        """
        try:
            canon = normalize_organism_type(organism_type)
        except UnsupportedOrganismError as err:
            return {
                "antigenicity_score": None,
                "is_antigen": None,
                "status": "abstain",
                "reason": "unsupported_organism",
                "message": str(err),
                "organism_type": organism_type,
            }

        if canon not in self.models:
            return {
                "antigenicity_score": None,
                "is_antigen": None,
                "status": "abstain",
                "reason": "model_not_trained",
                "message": f"No model registered for organism class '{canon}'.",
                "organism_type": canon,
            }

        model = self.models[canon]
        cutoff = threshold if threshold is not None else self.thresholds.get(canon, 0.50)

        # Extract ACC features
        from src.models.vaxijen_replication import extract_vaxijen_acc_matrix
        X = extract_vaxijen_acc_matrix([sequence])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            score = float(probs[0, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs[0])
        else:
            score = float(model.predict(X)[0])

        score = round(float(np.clip(score, 0.001, 0.999)), 4)
        is_ant = bool(score >= cutoff)

        return {
            "antigenicity_score": score,
            "is_antigen": is_ant,
            "status": "success",
            "reason": None,
            "organism_type": canon,
            "threshold": cutoff,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize organism models and thresholds."""
        data = {
            "models": self.models,
            "thresholds": self.thresholds,
            "supported": list(SUPPORTED_ORGANISMS_MAP.keys()),
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "OrganismSpecificAntigenicityManager":
        """Load serialized organism models."""
        data = joblib.load(filepath)
        mgr = cls()
        mgr.models = data.get("models", {})
        mgr.thresholds = data.get("thresholds", {})
        return mgr
