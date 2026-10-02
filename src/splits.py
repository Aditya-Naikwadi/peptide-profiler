"""Nested Group-Held-Out Splits Infrastructure.

Implements Task 3 of Shared Infrastructure:
1. Outer StratifiedGroupKFold (5 folds) on homology clusters for unbiased test evaluation.
2. Inner group-disjoint splits (Inner Train vs. Inner Calibration/Conformal) for zero-leakage
   hyperparameter tuning, probability calibration, and conformal prediction.
3. Cryptographic provenance: Persists all fold assignments with SHA-256 integrity hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


class NestedGroupSplitter:
    """Generates and verifies nested group-held-out splits on homology clusters."""

    def __init__(
        self,
        n_outer_splits: int = 5,
        inner_calibration_ratio: float = 0.25,
        random_state: int = RANDOM_SEED,
    ):
        self.n_outer_splits = n_outer_splits
        self.inner_calibration_ratio = inner_calibration_ratio
        self.random_state = random_state

    def generate_nested_splits(
        self,
        records: List[Dict[str, Any]],
        cluster_assignments: Dict[str, str],
        stratify_key: str = "is_antigen",
    ) -> Dict[str, Any]:
        """Generate nested outer and inner group-held-out splits.

        Args:
            records: List of sequence records with 'id' and labels.
            cluster_assignments: Mapping from sequence id to cluster_id.
            stratify_key: Primary endpoint key used to balance fold class proportions.

        Returns:
            Dictionary containing nested fold assignments and hashes.
        """
        seq_ids = [r["id"] for r in records]
        clusters = [cluster_assignments[sid] for sid in seq_ids]
        y_stratify = np.array([r.get(stratify_key, 0) for r in records], dtype=int)
        X_dummy = np.zeros((len(records), 1))

        # Outer Group-Held-Out Split
        outer_sgkf = StratifiedGroupKFold(
            n_splits=self.n_outer_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        folds_data = []

        for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_sgkf.split(X_dummy, y_stratify, groups=clusters)):
            outer_test_ids = [seq_ids[i] for i in test_idx]
            outer_test_clusters = sorted(list({clusters[i] for i in test_idx}))

            outer_train_ids = [seq_ids[i] for i in train_idx]
            outer_train_clusters = sorted(list({clusters[i] for i in train_idx}))

            # Strict verification of zero cluster overlap in outer fold
            assert set(outer_test_clusters).isdisjoint(set(outer_train_clusters)), (
                f"Outer fold {outer_fold_idx} has cluster leakage!"
            )

            # Inner Group-Disjoint Split on outer_train_clusters
            # Assign outer training clusters into Inner-Train vs. Inner-Calibration
            rng = np.random.RandomState(self.random_state + outer_fold_idx)
            shuffled_train_clusters = list(outer_train_clusters)
            rng.shuffle(shuffled_train_clusters)

            n_cal_clusters = max(1, int(len(shuffled_train_clusters) * self.inner_calibration_ratio))
            inner_cal_clusters = set(shuffled_train_clusters[:n_cal_clusters])
            inner_train_clusters = set(shuffled_train_clusters[n_cal_clusters:])

            inner_train_ids = [sid for sid in outer_train_ids if cluster_assignments[sid] in inner_train_clusters]
            inner_cal_ids = [sid for sid in outer_train_ids if cluster_assignments[sid] in inner_cal_clusters]

            # Strict inner disjointness verification
            assert inner_cal_clusters.isdisjoint(inner_train_clusters), "Inner train and cal clusters overlap!"
            assert inner_cal_clusters.isdisjoint(set(outer_test_clusters)), "Inner cal and outer test clusters overlap!"
            assert inner_train_clusters.isdisjoint(set(outer_test_clusters)), "Inner train and outer test clusters overlap!"

            folds_data.append({
                "outer_fold_id": outer_fold_idx,
                "outer_test": {
                    "sequence_count": len(outer_test_ids),
                    "cluster_count": len(outer_test_clusters),
                    "sequence_ids": outer_test_ids,
                    "cluster_ids": outer_test_clusters,
                },
                "outer_train": {
                    "sequence_count": len(outer_train_ids),
                    "cluster_count": len(outer_train_clusters),
                    "sequence_ids": outer_train_ids,
                    "cluster_ids": outer_train_clusters,
                },
                "inner_disjoint_partitions": {
                    "inner_train": {
                        "sequence_count": len(inner_train_ids),
                        "cluster_count": len(inner_train_clusters),
                        "sequence_ids": inner_train_ids,
                        "cluster_ids": sorted(list(inner_train_clusters)),
                    },
                    "inner_calibration": {
                        "sequence_count": len(inner_cal_ids),
                        "cluster_count": len(inner_cal_clusters),
                        "sequence_ids": inner_cal_ids,
                        "cluster_ids": sorted(list(inner_cal_clusters)),
                    },
                },
            })

        # Calculate structure checksum
        splits_json = json.dumps(folds_data, sort_keys=True)
        splits_sha256 = hashlib.sha256(splits_json.encode("utf-8")).hexdigest()

        return {
            "metadata": {
                "n_outer_splits": self.n_outer_splits,
                "inner_calibration_ratio": self.inner_calibration_ratio,
                "random_state": self.random_state,
                "stratify_key": stratify_key,
                "total_sequences": len(records),
                "total_clusters": len(set(clusters)),
                "splits_sha256": splits_sha256,
            },
            "folds": folds_data,
        }
