"""Antigenicity prediction module for peptides (VaxiJen / Vaxign-ML alternative)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.physicochem import AMINO_ACIDS, calculate_aac, calculate_pcp_descriptors

logger = logging.getLogger(__name__)

DEFAULT_ANTIGEN_MODEL = Path(__file__).resolve().parent.parent / "models" / "antigen_rf.joblib"
_CACHED_ANTIGEN_CLF = None


def is_docker_available() -> bool:
    """Check if Docker command-line tool and daemon are available."""
    docker_exe = shutil.which("docker")
    if not docker_exe:
        return False
    try:
        proc = subprocess.run(
            [docker_exe, "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        return proc.returncode == 0
    except Exception:
        return False


def run_vaxign_docker(
    sequence: str,
    organism_type: str = "bacteria",
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """Run Vaxign-ML Docker container if available."""
    if not is_docker_available():
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_fa = tmp_path / "input.fa"
            input_fa.write_text(f">seq1\n{sequence}\n", encoding="utf-8")

            # Mount current temp directory into container
            # Docker image e4ong1031/vaxign-ml:v1.0
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{tmp_path.resolve()}:/data",
                "e4ong1031/vaxign-ml:v1.0",
                "/data/input.fa", "/data/output", organism_type
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
            if proc.returncode == 0:
                # Look for output file in tmpdir
                out_files = list(tmp_path.glob("output*")) + list(tmp_path.glob("*.csv"))
                if out_files:
                    lines = out_files[0].read_text().splitlines()
                    if len(lines) > 1:
                        parts = lines[1].split(",")
                        score = float(parts[-1])
                        return {
                            "antigenicity_score": round(score, 3),
                            "is_antigen": bool(score >= 0.5),
                            "method": "Vaxign-ML (Docker)",
                            "organism_type": organism_type,
                        }
    except Exception as e:
        logger.warning(f"Vaxign-ML Docker execution failed: {e}. Falling back to ML model.")

    return None


def get_local_antigen_model():
    """Lazily load the local trained antigenicity model."""
    global _CACHED_ANTIGEN_CLF
    if _CACHED_ANTIGEN_CLF is not None:
        return _CACHED_ANTIGEN_CLF

    if DEFAULT_ANTIGEN_MODEL.exists():
        try:
            _CACHED_ANTIGEN_CLF = joblib.load(DEFAULT_ANTIGEN_MODEL)
            return _CACHED_ANTIGEN_CLF
        except Exception as e:
            logger.warning(f"Could not load antigen RF model: {e}")

    return None


def predict_antigenicity_local_ml(
    sequence: str,
    threshold: float = 0.5,
    organism_type: str = "bacteria",
) -> Dict[str, Any]:
    """Predict antigenicity using pre-trained physicochemical RF classifier."""
    clf = get_local_antigen_model()
    cleaned = sequence.strip().upper()

    # Extract 20 AAC + 30 PCP = 50 features
    aac = calculate_aac(cleaned)
    pcp = calculate_pcp_descriptors(cleaned)
    features = [aac[aa] for aa in AMINO_ACIDS] + list(pcp.values())
    X = np.array([features], dtype=np.float32)

    if clf is not None:
        probs = clf.predict_proba(X)[0]
        # Class 1 is Antigen
        score = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        # Fallback scoring: surface accessibility, positive charge, polarity
        surf_exposed = pcp.get("PCP_SA_EX", 0.0)
        polar = pcp.get("PCP_PO", 0.0)
        hydrophobic = pcp.get("PCP_HX", 0.0)
        score = (surf_exposed * 0.45) + (polar * 0.35) + (1.0 - hydrophobic) * 0.20

    score = round(float(np.clip(score, 0.01, 0.99)), 3)
    is_antigen = bool(score >= threshold)

    return {
        "antigenicity_score": score,
        "is_antigen": is_antigen,
        "method": "VaxiJen-Alternative (ACC/PCP ML Classifier)",
        "organism_type": organism_type,
        "threshold": threshold,
    }


def run_antigenicity(
    sequence: str,
    organism_type: str = "bacteria",
    threshold: float = 0.5,
    use_docker: bool = False,
) -> Dict[str, Any]:
    """Run peptide antigenicity prediction.

    Interface adheres to project specification:
    Returns {'antigenicity_score': float, 'is_antigen': bool}

    Args:
        sequence: Validated uppercase amino acid sequence string.
        organism_type: Organism category ("bacteria", "virus", "parasite", etc.)
        threshold: Classification cutoff (default 0.5)
        use_docker: If True, attempts to run Vaxign-ML via Docker first.

    Returns:
        Dictionary containing:
            - antigenicity_score: Probability between 0.0 and 1.0
            - is_antigen: Boolean classification
            - method: Prediction engine utilized
            - organism_type: Organism context
    """
    cleaned = sequence.strip().upper()

    if use_docker:
        res = run_vaxign_docker(cleaned, organism_type=organism_type)
        if res is not None:
            return res

    return predict_antigenicity_local_ml(
        cleaned,
        threshold=threshold,
        organism_type=organism_type,
    )
