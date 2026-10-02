"""Machine learning models and ensembling for Peptide Profiler."""

from src.models.baselines import (
    create_baseline_models,
    get_baseline_model,
    get_knn,
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
)
from src.models.nested_cv import (
    cluster_bootstrap_ci,
    compute_comprehensive_metrics,
    grouped_permutation_importance,
    run_nested_grouped_cv,
)
from src.models.organism import (
    OrganismSpecificAntigenicityManager,
    UnsupportedOrganismError,
    normalize_organism_type,
)
from src.models.vaxijen_replication import (
    RSM1NNClassifier,
    VaxiJenReplicationEnsemble,
    build_rf_feature_selection_pipeline,
    build_rsm_1nn_pipeline,
    extract_vaxijen_acc_matrix,
)

__all__ = [
    "create_baseline_models",
    "get_baseline_model",
    "get_logistic_regression",
    "get_random_forest",
    "get_xgboost",
    "get_knn",
    "RSM1NNClassifier",
    "VaxiJenReplicationEnsemble",
    "build_rsm_1nn_pipeline",
    "build_rf_feature_selection_pipeline",
    "extract_vaxijen_acc_matrix",
    "OrganismSpecificAntigenicityManager",
    "UnsupportedOrganismError",
    "normalize_organism_type",
    "compute_comprehensive_metrics",
    "cluster_bootstrap_ci",
    "run_nested_grouped_cv",
    "grouped_permutation_importance",
]
