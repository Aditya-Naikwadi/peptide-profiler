"""Local replication of VaxiJen v3.0 Architecture.

Reference: Dimitrov et al., Vaccines 2020, 8:709.
Architecture components:
1. Feature space: 125-D Auto Cross-Covariance (ACC) based on 5 Z-scales with lag L=5.
2. Three-model ensemble:
   - XGBoost classifier
   - Random Subspace Method 1-Nearest Neighbor (RSM-1NN)
   - Random Forest with feature selection (SelectFromModel)
3. Combiner: Majority vote (binary) and soft average (probabilities).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.acc import calculate_acc_features

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


class RSM1NNClassifier(BaseEstimator, ClassifierMixin):
    """Random Subspace Method with 1-Nearest Neighbor base classifiers.

    Draws multiple random subsets of feature indices, trains a 1-NN on each subspace,
    and aggregates predictions by voting or probability averaging.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        subspace_fraction: float = 0.5,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.subspace_fraction = subspace_fraction
        self.random_state = random_state
        self.estimators_: List[KNeighborsClassifier] = []
        self.subspaces_: List[np.ndarray] = []
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RSM1NNClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        subspace_dim = max(1, int(n_features * self.subspace_fraction))

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.subspaces_ = []

        for _ in range(self.n_estimators):
            subspace_idx = rng.choice(n_features, size=subspace_dim, replace=False)
            knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn.fit(X[:, subspace_idx], y)
            self.estimators_.append(knn)
            self.subspaces_.append(subspace_idx)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        all_probs = []
        for knn, subspace_idx in zip(self.estimators_, self.subspaces_):
            probs = knn.predict_proba(X[:, subspace_idx])
            if probs.shape[1] == 1:
                # Only one class present in subset
                c = knn.classes_[0]
                full_prob = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
                full_prob[:, c] = 1.0
                all_probs.append(full_prob)
            else:
                all_probs.append(probs)

        avg_prob = np.mean(all_probs, axis=0)
        return avg_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def build_rsm_1nn_pipeline(
    n_estimators: int = 25,
    subspace_fraction: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build StandardScaler + RSM-1NN pipeline for zero leakage."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rsm_1nn", RSM1NNClassifier(
            n_estimators=n_estimators,
            subspace_fraction=subspace_fraction,
            random_state=random_state,
        )),
    ])


def build_rf_feature_selection_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build Random Forest with SelectFromModel feature selection inside training fold."""
    base_selector = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    feature_selector = SelectFromModel(
        estimator=base_selector,
        threshold="median",
        prefit=False,
    )
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([
        ("select", feature_selector),
        ("rf", final_rf),
    ])


class VaxiJenReplicationEnsemble(BaseEstimator, ClassifierMixin):
    """Local replication of VaxiJen v3.0 three-model ensemble.

    Models:
    1. XGBoost
    2. RSM-1NN (with StandardScaler)
    3. Random Forest with feature selection (SelectFromModel)
    """

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        voting: str = "majority",  # "majority" or "soft"
    ) -> None:
        self.random_state = random_state
        self.voting = voting
        self.xgb_ = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rsm_1nn_ = build_rsm_1nn_pipeline(random_state=self.random_state)
        self.rf_fs_ = build_rf_feature_selection_pipeline(random_state=self.random_state)
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VaxiJenReplicationEnsemble":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)

        # Fit all three models
        self.xgb_.fit(X, y)
        self.rsm_1nn_.fit(X, y)
        self.rf_fs_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        p_xgb = self.xgb_.predict_proba(X)
        p_rsm = self.rsm_1nn_.predict_proba(X)
        p_rf = self.rf_fs_.predict_proba(X)

        # Soft average
        p_avg = (p_xgb + p_rsm + p_rf) / 3.0
        return p_avg

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)

        # Majority vote of binary predictions
        pred_xgb = self.xgb_.predict(X)
        pred_rsm = self.rsm_1nn_.predict(X)
        pred_rf = self.rf_fs_.predict(X)

        votes = pred_xgb + pred_rsm + pred_rf
        return (votes >= 2).astype(int)

    def get_component_predictions(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Return probabilities and predictions for each individual component."""
        X = np.asarray(X, dtype=np.float32)
        p_xgb = self.xgb_.predict_proba(X)[:, 1]
        p_rsm = self.rsm_1nn_.predict_proba(X)[:, 1]
        p_rf = self.rf_fs_.predict_proba(X)[:, 1]
        p_ens = self.predict_proba(X)[:, 1]
        v_ens = self.predict(X)

        return {
            "xgboost": {"prob": p_xgb, "pred": (p_xgb >= 0.5).astype(int)},
            "rsm_1nn": {"prob": p_rsm, "pred": (p_rsm >= 0.5).astype(int)},
            "rf_selected": {"prob": p_rf, "pred": (p_rf >= 0.5).astype(int)},
            "ensemble": {"prob": p_ens, "pred": v_ens},
        }


def extract_vaxijen_acc_matrix(sequences: List[str], lag: int = 5) -> np.ndarray:
    """Extract 125-D ACC feature matrix for sequences."""
    n = len(sequences)
    X = np.zeros((n, 125), dtype=np.float32)
    for i, seq in enumerate(sequences):
        X[i] = calculate_acc_features(seq, max_lag=lag)
    return X
