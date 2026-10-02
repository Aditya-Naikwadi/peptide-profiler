"""Tests for antigenicity models, VaxiJen replication ensemble, and organism routing."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models import (
    OrganismSpecificAntigenicityManager,
    RSM1NNClassifier,
    UnsupportedOrganismError,
    VaxiJenReplicationEnsemble,
    build_rf_feature_selection_pipeline,
    build_rsm_1nn_pipeline,
    create_baseline_models,
    extract_vaxijen_acc_matrix,
    get_baseline_model,
    normalize_organism_type,
    run_nested_grouped_cv,
)


@pytest.fixture
def synthetic_acc_data():
    """Generate synthetic 125-D ACC dataset with cluster groupings."""
    rng = np.random.RandomState(42)
    n_samples = 60
    n_features = 125
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Informative signal on first 5 features
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0.0).astype(int)
    # 10 clusters of size 6
    clusters = [f"cluster_{i // 6}" for i in range(n_samples)]
    return X, y, clusters


def test_baseline_instantiation():
    """Verify all standard baselines instantiate correctly."""
    models = create_baseline_models()
    assert "LogisticRegression" in models
    assert "RandomForest" in models
    assert "XGBoost" in models
    assert "kNN" in models

    # Check scaling pipelines
    assert isinstance(models["LogisticRegression"], Pipeline)
    assert isinstance(models["kNN"], Pipeline)
    assert "scaler" in models["LogisticRegression"].named_steps
    assert "scaler" in models["kNN"].named_steps


def test_baseline_fit_and_predict(synthetic_acc_data):
    """Verify baseline models can fit and predict on synthetic ACC data."""
    X, y, clusters = synthetic_acc_data
    for name in ["lr", "rf", "xgb", "knn"]:
        model = get_baseline_model(name)
        model.fit(X[:40], y[:40])
        preds = model.predict(X[40:])
        probs = model.predict_proba(X[40:])
        assert len(preds) == 20
        assert probs.shape == (20, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_rsm_1nn_classifier(synthetic_acc_data):
    """Verify Random Subspace Method 1-NN ensemble."""
    X, y, _ = synthetic_acc_data
    rsm = RSM1NNClassifier(n_estimators=10, subspace_fraction=0.3, random_state=42)
    rsm.fit(X[:40], y[:40])
    probs = rsm.predict_proba(X[40:])
    preds = rsm.predict(X[40:])
    assert probs.shape == (20, 2)
    assert len(preds) == 20
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_rf_feature_selection_pipeline(synthetic_acc_data):
    """Verify Random Forest with feature selection inside training fold."""
    X, y, _ = synthetic_acc_data
    pipe = build_rf_feature_selection_pipeline(n_estimators=20, max_depth=4, random_state=42)
    pipe.fit(X[:40], y[:40])
    probs = pipe.predict_proba(X[40:])
    assert probs.shape == (20, 2)
    # Ensure feature selection was applied
    selected_features = pipe.named_steps["select"].get_support()
    assert selected_features.sum() < X.shape[1]


def test_vaxijen_replication_ensemble(synthetic_acc_data):
    """Verify VaxiJen v3.0 replication ensemble with majority voting."""
    X, y, _ = synthetic_acc_data
    ens = VaxiJenReplicationEnsemble(random_state=42, voting="majority")
    ens.fit(X[:40], y[:40])

    preds = ens.predict(X[40:])
    probs = ens.predict_proba(X[40:])
    assert len(preds) == 20
    assert probs.shape == (20, 2)

    components = ens.get_component_predictions(X[40:])
    assert "xgboost" in components
    assert "rsm_1nn" in components
    assert "rf_selected" in components
    assert "ensemble" in components
    assert len(components["xgboost"]["prob"]) == 20


def test_organism_normalization():
    """Verify organism type normalization and error handling."""
    assert normalize_organism_type("bacteria") == "bacteria"
    assert normalize_organism_type("Bacterial") == "bacteria"
    assert normalize_organism_type("virus") == "virus"
    assert normalize_organism_type("Viral") == "virus"
    assert normalize_organism_type("tumor") == "tumor"
    assert normalize_organism_type("human_tumor") == "tumor"
    assert normalize_organism_type("cancer") == "tumor"

    with pytest.raises(UnsupportedOrganismError):
        normalize_organism_type("fungus")

    with pytest.raises(UnsupportedOrganismError):
        normalize_organism_type("plant")

    with pytest.raises(UnsupportedOrganismError):
        normalize_organism_type(None)


def test_organism_manager_abstention(tmp_path):
    """Verify organism manager abstains cleanly on unsupported or unregistered taxa."""
    mgr = OrganismSpecificAntigenicityManager()
    dummy_model = get_baseline_model("rf")
    # Fit dummy model on small data
    dummy_model.fit(np.zeros((10, 125)), np.array([0, 1] * 5))
    mgr.register_model("bacteria", dummy_model, threshold=0.5)

    # Valid supported organism
    res_bact = mgr.predict_sequence("ACDEFGHIKLMNPQRSTVWY", organism_type="bacteria")
    assert res_bact["status"] == "success"
    assert res_bact["antigenicity_score"] is not None

    # Unsupported organism (e.g. protozoa/fungal)
    res_unsupp = mgr.predict_sequence("ACDEFGHIKLMNPQRSTVWY", organism_type="fungal")
    assert res_unsupp["status"] == "abstain"
    assert res_unsupp["reason"] == "unsupported_organism"
    assert res_unsupp["antigenicity_score"] is None
    assert res_unsupp["is_antigen"] is None

    # Supported but unregistered model (e.g. viral)
    res_unreg = mgr.predict_sequence("ACDEFGHIKLMNPQRSTVWY", organism_type="viral")
    assert res_unreg["status"] == "abstain"
    assert res_unreg["reason"] == "model_not_trained"

    # Test serialization and loading
    save_path = tmp_path / "organism_models.joblib"
    mgr.save(save_path)
    loaded_mgr = OrganismSpecificAntigenicityManager.load(save_path)
    assert "bacteria" in loaded_mgr.models
    assert loaded_mgr.thresholds["bacteria"] == 0.5


def test_nested_grouped_cv(synthetic_acc_data):
    """Verify nested grouped CV executes without data leakage."""
    X, y, clusters = synthetic_acc_data
    rf = get_baseline_model("rf", n_estimators=10, max_depth=3)
    results = run_nested_grouped_cv(rf, X, y, clusters, n_outer_splits=3)

    assert "auroc" in results
    assert "auprc" in results
    assert "auroc_ci" in results
    assert len(results["fold_metrics"]) == 3
    assert len(results["oof_probs"]) == len(y)
