"""Baseline classification models for peptide properties.

All models requiring feature scaling (LogisticRegression, kNN) encapsulate
a StandardScaler inside a scikit-learn Pipeline to guarantee zero leakage across folds.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 42


def get_logistic_regression(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Create a regularized L2 Logistic Regression with StandardScaler pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=RANDOM_SEED,
        )),
    ])


def get_random_forest(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 5,
) -> RandomForestClassifier:
    """Create a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def get_xgboost(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Create an XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def get_knn(n_neighbors: int = 5, metric: str = "cosine") -> Pipeline:
    """Create a k-Nearest Neighbors classifier with StandardScaler pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=-1,
        )),
    ])


def create_baseline_models() -> Dict[str, Any]:
    """Return dictionary of all standard baseline models."""
    return {
        "LogisticRegression": get_logistic_regression(),
        "RandomForest": get_random_forest(),
        "XGBoost": get_xgboost(),
        "kNN": get_knn(),
    }


def get_baseline_model(name: str, **kwargs: Any) -> Any:
    """Retrieve an instantiated baseline model by name."""
    name_lower = name.lower()
    if name_lower in ("lr", "logistic", "logisticregression"):
        return get_logistic_regression(**kwargs)
    elif name_lower in ("rf", "randomforest", "random_forest"):
        return get_random_forest(**kwargs)
    elif name_lower in ("xgb", "xgboost"):
        return get_xgboost(**kwargs)
    elif name_lower in ("knn", "k_nearest_neighbors"):
        return get_knn(**kwargs)
    else:
        raise ValueError(f"Unknown baseline model: {name}. Supported: 'LogisticRegression', 'RandomForest', 'XGBoost', 'kNN'.")
