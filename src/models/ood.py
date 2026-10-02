"""Applicability Domain and Out-of-Distribution (OOD) Guardrail Engine.

Implements:
1. Multi-space distance metrics:
   - Mahalanobis distance with class-conditional centroids and tied Ledoit-Wolf shrinkage covariance.
   - Cosine k-Nearest-Neighbor (kNN) distance to training set manifold.
2. Group-held-out percentile thresholding (e.g., 99th percentile) on inner calibration folds (zero leakage).
3. Rule-based biological guards:
   - Chemical modifications (D-amino acids, cyclization, PTMs, terminal caps) triggering immediate abstention.
   - Biophysical length boundaries (e.g. < 5 aa or > 1000 aa).
   - Non-canonical amino acid presence/fraction.
   - Extreme low sequence complexity (Shannon entropy < 1.5 bits) and skewed single-residue composition (> 50%).
4. Strict output contract conforming to:
   {"status": "abstain", "reason": "out_of_applicability_domain",
    "details": {"mahalanobis": 41.2, "threshold": 33.5, "guard": null}}
"""

from __future__ import annotations

import collections
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

CHEMICAL_MODIFICATION_PATTERNS = [
    r"\bD-",
    r"\bd-",
    r"\[D-[A-Z]\]",
    r"\(D-[A-Z]\)",
    r"\bac-",
    r"-nh2\b",
    r"\bamidation\b",
    r"\bcyclic\b",
    r"\bcyclo\b",
    r"\bphospho\b",
    r"\bmethyl\b",
    r"\bptm\b",
    r"\bhydroxylation\b",
    r"\bacetyl\b",
    r"\bsulfation\b",
    r"\bcitrulline\b",
    r"[\[\(\{][^\]\)\}]+[\]\)\}]",  # Bracketed/parenthesized modification annotations
]


def compute_sequence_entropy(sequence: str) -> float:
    """Compute Shannon entropy (in bits) of amino acid distribution."""
    if not sequence:
        return 0.0
    counts = collections.Counter(sequence.upper())
    n = len(sequence)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


def compute_max_residue_fraction(sequence: str) -> float:
    """Compute maximum frequency of any single residue in the sequence."""
    if not sequence:
        return 0.0
    counts = collections.Counter(sequence.upper())
    return max(counts.values()) / len(sequence)


def check_chemical_modifications(sequence: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Detect chemical modifications (D-amino acids, cyclization, PTMs).

    Sequence-only ML models assume linear L-amino acid polypeptides.
    Any chemical modification annotations require explicit abstention.
    """
    if metadata:
        # Check explicit metadata fields
        mod_keys = ["modifications", "ptm", "is_cyclic", "d_amino_acids", "n_term_mod", "c_term_mod"]
        for k in mod_keys:
            val = metadata.get(k)
            if val is not None and val is not False and val != "" and val != []:
                return f"annotated_modification_{k}"

        desc = str(metadata.get("description", ""))
        for pat in CHEMICAL_MODIFICATION_PATTERNS:
            if re.search(pat, desc, flags=re.IGNORECASE):
                return "annotated_chemical_modification_in_metadata"

    # Check raw sequence string for non-alphabetic modification tags
    for pat in CHEMICAL_MODIFICATION_PATTERNS:
        if re.search(pat, sequence, flags=re.IGNORECASE):
            return "annotated_chemical_modification_in_sequence"

    return None


def evaluate_rule_guards(
    sequence: str,
    metadata: Optional[Dict[str, Any]] = None,
    min_length: int = 5,
    max_length: int = 1000,
    max_non_canonical_fraction: float = 0.0,
    min_entropy: float = 1.5,
    max_single_aa_frac: float = 0.50,
) -> Optional[str]:
    """Execute rule-based biological guard checks on candidate sequence."""
    clean_seq = sequence.strip()

    # 1. Chemical modification check
    chem_mod = check_chemical_modifications(clean_seq, metadata)
    if chem_mod:
        return chem_mod

    # 2. Length boundaries
    seq_len = len(clean_seq)
    if seq_len < min_length:
        return f"length_under_minimum_{min_length}"
    if seq_len > max_length:
        return f"length_over_maximum_{max_length}"

    # 3. Non-canonical residue check
    upper_seq = clean_seq.upper()
    non_canonical_count = sum(1 for aa in upper_seq if aa not in STANDARD_AMINO_ACIDS)
    non_canonical_frac = non_canonical_count / max(1, seq_len)
    if non_canonical_frac > max_non_canonical_fraction:
        return "non_canonical_residue_detected"

    # 4. Low-complexity and skewed composition
    entropy = compute_sequence_entropy(upper_seq)
    if entropy < min_entropy:
        return f"extreme_low_complexity_entropy_{entropy:.2f}_bits"

    max_aa_frac = compute_max_residue_fraction(upper_seq)
    if max_aa_frac > max_single_aa_frac:
        return f"extreme_skewed_composition_max_aa_{max_aa_frac:.2f}"

    return None


class ApplicabilityDomainDetector:
    """Applicability domain detector using Mahalanobis distance & Cosine kNN."""

    def __init__(
        self,
        percentile: float = 99.0,
        knn_k: int = 5,
        space_name: str = "embedding_space",
    ):
        """Initialize detector.

        Args:
            percentile: In-distribution calibration percentile for distance cutoff (e.g. 99.0).
            knn_k: Number of nearest neighbors for cosine kNN distance.
            space_name: Human-readable space identifier ('embedding_space' or 'handcrafted_space').
        """
        self.percentile = float(percentile)
        self.knn_k = int(knn_k)
        self.space_name = space_name

        self.centroids: Dict[int, np.ndarray] = {}
        self.tied_precision: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.mahalanobis_threshold: Optional[float] = None
        self.knn_threshold: Optional[float] = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
    ) -> "ApplicabilityDomainDetector":
        """Fit class-conditional centroids, tied Ledoit-Wolf precision, and kNN graph.

        Args:
            X_train: Training feature matrix, shape (N, D).
            y_train: Binary class labels, shape (N,).
            X_cal: Group-disjoint calibration feature matrix for threshold selection.
                   If None, X_train is used with a warning.
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=int)
        N, D = X_tr.shape

        # 1. Compute class-conditional centroids
        for c in (0, 1):
            mask = y_tr == c
            if not np.any(mask):
                raise ValueError(f"Training set has no samples for class {c}")
            self.centroids[c] = np.mean(X_tr[mask], axis=0)

        # 2. Tied centering: subtract each class's centroid from its samples
        X_centered = np.empty_like(X_tr)
        for c in (0, 1):
            mask = y_tr == c
            X_centered[mask] = X_tr[mask] - self.centroids[c]

        # 3. Fit tied shrinkage covariance with Ledoit-Wolf
        lw = LedoitWolf(assume_centered=True)
        lw.fit(X_centered)
        self.tied_precision = lw.precision_  # Shape (D, D)

        # 4. Fit Cosine Nearest Neighbors
        # Normalize vectors for cosine metric
        norms = np.linalg.norm(X_tr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_tr_norm = X_tr / norms

        k_val = min(self.knn_k, N)
        self.nn_model = NearestNeighbors(n_neighbors=k_val, metric="cosine")
        self.nn_model.fit(X_tr_norm)

        # Mark fitted before computing calibration distances
        self.is_fitted = True

        # 5. Fit Distance Thresholds on Calibration Set
        X_eval = np.asarray(X_cal, dtype=np.float64) if X_cal is not None else X_tr
        cal_mahal = self.compute_mahalanobis(X_eval)
        cal_knn = self.compute_knn_cosine(X_eval)

        self.mahalanobis_threshold = float(np.percentile(cal_mahal, self.percentile))
        self.knn_threshold = float(np.percentile(cal_knn, self.percentile))

        logger.info(
            f"Fitted {self.space_name} ApplicabilityDomainDetector: "
            f"Mahalanobis {self.percentile}th threshold = {self.mahalanobis_threshold:.2f}, "
            f"Cosine kNN threshold = {self.knn_threshold:.4f}"
        )
        return self

    def compute_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance to either class centroid."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing Mahalanobis distance.")
        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        dists = []
        for c in (0, 1):
            diff = X_mat - self.centroids[c]  # Shape (M, D)
            # Mahalanobis dist squared = sum_j (diff * (diff @ precision))
            d2 = np.sum((diff @ self.tied_precision) * diff, axis=1)
            d2 = np.maximum(d2, 0.0)
            dists.append(np.sqrt(d2))

        # Minimum distance to any class centroid
        min_dist = np.minimum(dists[0], dists[1])
        return min_dist

    def compute_knn_cosine(self, X: np.ndarray) -> np.ndarray:
        """Compute mean cosine distance to k-nearest training neighbors."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before computing kNN cosine distance.")
        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        norms = np.linalg.norm(X_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_mat / norms

        distances, _ = self.nn_model.kneighbors(X_norm)
        mean_knn_dist = np.mean(distances, axis=1)
        return mean_knn_dist

    def inspect(
        self,
        sequence: str,
        feature_vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inspect query candidate against applicability domain and rule guards.

        Returns strict output contract:
            {"status": "abstain" | "in_domain",
             "reason": "out_of_applicability_domain" | null,
             "details": {"mahalanobis": float | null, "threshold": float, "guard": str | null}}
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before inspection.")

        # 1. First priority: Rule-based guards
        guard_reason = evaluate_rule_guards(sequence, metadata)
        if guard_reason is not None:
            return {
                "status": "abstain",
                "reason": "out_of_applicability_domain",
                "details": {
                    "mahalanobis": None,
                    "threshold": round(float(self.mahalanobis_threshold), 1),
                    "guard": guard_reason,
                    "knn_cosine": None,
                    "knn_threshold": round(float(self.knn_threshold), 4),
                },
            }

        # 2. Second priority: Mahalanobis distance check
        vec = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        mahal_dist = float(self.compute_mahalanobis(vec)[0])
        knn_dist = float(self.compute_knn_cosine(vec)[0])

        is_ood = (mahal_dist > self.mahalanobis_threshold) or (knn_dist > self.knn_threshold)

        if is_ood:
            return {
                "status": "abstain",
                "reason": "out_of_applicability_domain",
                "details": {
                    "mahalanobis": round(mahal_dist, 1),
                    "threshold": round(float(self.mahalanobis_threshold), 1),
                    "guard": None,
                    "knn_cosine": round(knn_dist, 4),
                    "knn_threshold": round(float(self.knn_threshold), 4),
                },
            }

        return {
            "status": "in_domain",
            "reason": None,
            "details": {
                "mahalanobis": round(mahal_dist, 1),
                "threshold": round(float(self.mahalanobis_threshold), 1),
                "guard": None,
                "knn_cosine": round(knn_dist, 4),
                "knn_threshold": round(float(self.knn_threshold), 4),
            },
        }
