"""Master Script: Build and Freeze Shared Infrastructure.

Executes all 6 tasks required by the Shared Infrastructure specification:
1. Inventory and deduplicate all datasets (antigenicity, toxicity, allergenicity).
2. Cluster sequences with configurable homology (40% identity, 80% coverage, short-peptide k-mer fallback).
3. Produce nested group-held-out splits (outer test, inner train, inner calibration).
4. Run strict leakage audit (fails if any cluster or identity >= 40% leaks across partitions).
5. Freeze baseline metrics for ACC (125-D) and RF/GB models with cluster-bootstrap CIs and length stratification.
6. Provide reproducible command-line execution and versioned artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.audit import LeakageAuditError, LeakageAuditor
from src.clustering import SequenceClusterer
from src.data.inventory import inventory_and_deduplicate
from src.evaluation import evaluate_screening_classifier
from src.features.acc import calculate_acc_features
from src.features.aac import calculate_aac_vector
from src.features.pcp import calculate_pcp_vector
from src.models.metrics import LENGTH_BINS
from src.splits import NestedGroupSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

RANDOM_SEED = 42


def run_pipeline():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # TASK 1: DATASET INVENTORY & DEDUPLICATION
    # ---------------------------------------------------------
    logger.info("=== STEP 1: Inventorying and Deduplicating Datasets ===")
    raw_dataset_path = DATA_DIR / "evaluation_dataset.json"
    sources_config_path = CONFIG_DIR / "sources.yaml"

    inventory, clean_records = inventory_and_deduplicate(
        raw_dataset_path=raw_dataset_path,
        sources_config_path=sources_config_path,
        near_dup_threshold=0.98,
    )

    clean_dataset_path = DATA_DIR / "curated_deduplicated_dataset.json"
    with open(clean_dataset_path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2)

    # Also save clean FASTA
    clean_fasta_path = DATA_DIR / "curated_deduplicated_dataset.fa"
    with open(clean_fasta_path, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(f">{r['id']} {r.get('description', '')}\n{r['sequence']}\n")

    inventory_path = DATA_DIR / "dataset_inventory.json"
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    logger.info(
        "Inventory complete: %d raw -> %d clean sequences (removed %d duplicates). SHA-256: %s",
        inventory["sizes"]["raw_sequences_count"],
        inventory["sizes"]["clean_sequences_count"],
        inventory["sizes"]["removed_duplicates_count"],
        inventory["metadata"]["clean_dataset_sha256"][:12],
    )

    # ---------------------------------------------------------
    # TASK 2: HOMOLOGY-AWARE SEQUENCE CLUSTERING (40% identity)
    # ---------------------------------------------------------
    logger.info("=== STEP 2: Homology Clustering (40% Identity, Short-Peptide Fallback) ===")
    clusterer = SequenceClusterer(
        identity_threshold=0.40,
        coverage_threshold=0.80,
        short_peptide_len=20,
        kmer_k=3,
        kmer_jaccard_threshold=0.25,
    )
    clustering_result = clusterer.cluster_records(clean_records)

    cluster_artifact_path = DATA_DIR / "cluster_assignments_40pct.json"
    with open(cluster_artifact_path, "w", encoding="utf-8") as f:
        json.dump(clustering_result, f, indent=2)

    cluster_map = clustering_result["assignments"]
    logger.info(
        "Clustering complete: %d sequences partitioned into %d homology clusters (%d singletons).",
        clustering_result["metadata"]["total_sequences"],
        clustering_result["metadata"]["total_clusters"],
        clustering_result["metadata"]["singletons"],
    )

    # ---------------------------------------------------------
    # TASK 3: NESTED GROUP-HELD-OUT SPLITS
    # ---------------------------------------------------------
    logger.info("=== STEP 3: Generating Nested Group-Held-Out Splits ===")
    splitter = NestedGroupSplitter(
        n_outer_splits=5,
        inner_calibration_ratio=0.25,
        random_state=RANDOM_SEED,
    )
    splits_data = splitter.generate_nested_splits(
        records=clean_records,
        cluster_assignments=cluster_map,
        stratify_key="is_antigen",
    )

    nested_splits_path = DATA_DIR / "nested_group_splits.json"
    with open(nested_splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    logger.info(
        "Nested splits generated: 5 outer folds with disjoint inner calibration sets. Checksum: %s",
        splits_data["metadata"]["splits_sha256"][:12],
    )

    # ---------------------------------------------------------
    # TASK 4: STRICT LEAKAGE AUDIT
    # ---------------------------------------------------------
    logger.info("=== STEP 4: Running Strict Leakage Audit ===")
    auditor = LeakageAuditor(identity_threshold=0.40)
    try:
        audit_certificate = auditor.audit_splits(
            splits_data=splits_data,
            records=clean_records,
            cluster_assignments=cluster_map,
        )
        logger.info(
            "LEAKAGE AUDIT PASSED! %d cross-comparisons verified. Max cross-partition identity: %.2f%% (threshold %.2f%%).",
            audit_certificate["pairwise_cross_comparisons_performed"],
            audit_certificate["max_cross_partition_identity_observed"] * 100,
            audit_certificate["identity_threshold_enforced"] * 100,
        )
    except LeakageAuditError as e:
        logger.error("FATAL: Leakage audit failed: %s", e)
        raise

    audit_path = RESULTS_DIR / "leakage_audit_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_certificate, f, indent=2)

    # ---------------------------------------------------------
    # TASK 5 & 6: FREEZE BASELINE METRICS UNDER NESTED SPLITS
    # ---------------------------------------------------------
    logger.info("=== STEP 5: Freezing Baseline Metrics (ACC & RF/GB) ===")
    seq_ids = [r["id"] for r in clean_records]
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}
    sequences = [r["sequence"] for r in clean_records]
    lengths = [len(s) for s in sequences]
    clusters = [cluster_map[sid] for sid in seq_ids]

    y_antigen = np.array([r.get("is_antigen", 0) for r in clean_records], dtype=int)
    y_toxic = np.array([r.get("is_toxic", 0) for r in clean_records], dtype=int)
    y_allergen = np.array([r.get("is_allergen", 0) for r in clean_records], dtype=int)

    # Extract standard feature sets
    logger.info("Extracting features: 125-D ACC and 50-D AAC+PCP...")
    feats_125d = np.array([calculate_acc_features(s, max_lag=5) for s in sequences], dtype=np.float32)
    feats_50d = np.array(
        [np.hstack([calculate_aac_vector(s), calculate_pcp_vector(s)]) for s in sequences],
        dtype=np.float32,
    )

    # Model evaluation containers (out-of-fold predictions)
    oof_antigen_acc_rf = np.zeros(len(clean_records), dtype=np.float64)
    oof_antigen_acc_gb = np.zeros(len(clean_records), dtype=np.float64)
    oof_allergen_rf = np.zeros(len(clean_records), dtype=np.float64)
    oof_toxic_rf = np.zeros(len(clean_records), dtype=np.float64)

    for fold in splits_data["folds"]:
        fold_id = fold["outer_fold_id"]
        train_indices = [id_to_idx[sid] for sid in fold["outer_train"]["sequence_ids"]]
        test_indices = [id_to_idx[sid] for sid in fold["outer_test"]["sequence_ids"]]

        # 1. Antigenicity: 125-D ACC + Random Forest
        rf_antigen = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_antigen.fit(feats_125d[train_indices], y_antigen[train_indices])
        oof_antigen_acc_rf[test_indices] = rf_antigen.predict_proba(feats_125d[test_indices])[:, 1]

        # 2. Antigenicity: 125-D ACC + Gradient Boosting
        gb_antigen = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
        gb_antigen.fit(feats_125d[train_indices], y_antigen[train_indices])
        oof_antigen_acc_gb[test_indices] = gb_antigen.predict_proba(feats_125d[test_indices])[:, 1]

        # 3. Allergenicity: 50-D AAC+PCP + Random Forest
        rf_allergen = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_allergen.fit(feats_50d[train_indices], y_allergen[train_indices])
        oof_allergen_rf[test_indices] = rf_allergen.predict_proba(feats_50d[test_indices])[:, 1]

        # 4. Toxicity: 50-D AAC+PCP + Random Forest
        rf_toxic = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_toxic.fit(feats_50d[train_indices], y_toxic[train_indices])
        oof_toxic_rf[test_indices] = rf_toxic.predict_proba(feats_50d[test_indices])[:, 1]

    # Evaluate models using Shared Evaluation Module
    frozen_baselines = {
        "metadata": {
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clean_dataset_sha256": inventory["metadata"]["clean_dataset_sha256"],
            "cluster_assignments_sha256": clustering_result["metadata"]["cluster_assignments_sha256"],
            "splits_sha256": splits_data["metadata"]["splits_sha256"],
            "total_clean_samples": len(clean_records),
            "total_clusters": clustering_result["metadata"]["total_clusters"],
            "identity_threshold": 0.40,
            "random_seed": RANDOM_SEED,
            "evaluation_protocol": "EVALUATION_PROTOCOL.md §1-§3",
        },
        "models": {
            "antigenicity_acc_rf": evaluate_screening_classifier(
                y_true=y_antigen,
                y_prob=oof_antigen_acc_rf,
                cluster_ids=clusters,
                lengths=lengths,
                threshold=0.5,
                positive_class_label="antigen",
            ),
            "antigenicity_acc_gb": evaluate_screening_classifier(
                y_true=y_antigen,
                y_prob=oof_antigen_acc_gb,
                cluster_ids=clusters,
                lengths=lengths,
                threshold=0.5,
                positive_class_label="antigen",
            ),
            "allergenicity_pcp_rf": evaluate_screening_classifier(
                y_true=y_allergen,
                y_prob=oof_allergen_rf,
                cluster_ids=clusters,
                lengths=lengths,
                threshold=0.5,
                positive_class_label="allergen",
            ),
            "toxicity_pcp_rf": evaluate_screening_classifier(
                y_true=y_toxic,
                y_prob=oof_toxic_rf,
                cluster_ids=clusters,
                lengths=lengths,
                threshold=0.5,
                positive_class_label="toxic",
            ),
        },
    }

    frozen_baseline_path = RESULTS_DIR / "frozen_baseline_metrics.json"
    with open(frozen_baseline_path, "w", encoding="utf-8") as f:
        json.dump(frozen_baselines, f, indent=2)

    logger.info("Frozen baseline metrics saved to %s", frozen_baseline_path)

    # ---------------------------------------------------------
    # GENERATE MARKDOWN DOCUMENTATION REPORT
    # ---------------------------------------------------------
    markdown_report = generate_markdown_summary(inventory, clustering_result, audit_certificate, frozen_baselines)
    doc_path = DOCS_DIR / "SHARED_INFRASTRUCTURE_REPORT.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    logger.info("Comprehensive report written to %s", doc_path)
    return frozen_baselines


def generate_markdown_summary(inventory, clustering, audit, baselines) -> str:
    m_ant_rf = baselines["models"]["antigenicity_acc_rf"]["metrics"]
    ci_ant_rf = baselines["models"]["antigenicity_acc_rf"]["confidence_intervals_95"]

    m_ant_gb = baselines["models"]["antigenicity_acc_gb"]["metrics"]
    ci_ant_gb = baselines["models"]["antigenicity_acc_gb"]["confidence_intervals_95"]

    m_alg_rf = baselines["models"]["allergenicity_pcp_rf"]["metrics"]
    ci_alg_rf = baselines["models"]["allergenicity_pcp_rf"]["confidence_intervals_95"]

    m_tox_rf = baselines["models"]["toxicity_pcp_rf"]["metrics"]
    ci_tox_rf = baselines["models"]["toxicity_pcp_rf"]["confidence_intervals_95"]

    lines = [
        "# Shared Infrastructure, Leakage Audit & Frozen Baselines Report",
        "",
        "## 1. Dataset Inventory & Deduplication (Task 1)",
        f"- **Raw Sequences:** {inventory['sizes']['raw_sequences_count']}",
        f"- **Clean Deduplicated Sequences:** {inventory['sizes']['clean_sequences_count']}",
        f"- **Removed Duplicates/Near-Duplicates (>=98% ID):** {inventory['sizes']['removed_duplicates_count']}",
        f"- **Clean Dataset SHA-256:** `{inventory['metadata']['clean_dataset_sha256']}`",
        "",
        "### Class Balance Summary",
        "| Endpoint | Clean Positives | Clean Negatives | Prevalence |",
        "|---|---|---|---|",
        f"| Antigenicity | {inventory['class_balance']['antigenicity']['clean']['positive_count']} | {inventory['class_balance']['antigenicity']['clean']['negative_count']} | {inventory['class_balance']['antigenicity']['clean']['positive_prevalence']:.1%} |",
        f"| Toxicity | {inventory['class_balance']['toxicity']['clean']['positive_count']} | {inventory['class_balance']['toxicity']['clean']['negative_count']} | {inventory['class_balance']['toxicity']['clean']['positive_prevalence']:.1%} |",
        f"| Allergenicity | {inventory['class_balance']['allergenicity']['clean']['positive_count']} | {inventory['class_balance']['allergenicity']['clean']['negative_count']} | {inventory['class_balance']['allergenicity']['clean']['positive_prevalence']:.1%} |",
        "",
        "### Length Distribution Breakdown",
        f"- Length span: {inventory['length_distributions']['clean']['min']} - {inventory['length_distributions']['clean']['max']} aa (median: {inventory['length_distributions']['clean']['median']:.1f} aa, mean: {inventory['length_distributions']['clean']['mean']:.1f} aa)",
        "",
        "| Length Bin | Sample Count | Fraction of Dataset |",
        "|---|---|---|",
    ]
    for b_name, b_data in inventory["length_distributions"]["clean"]["bins"].items():
        lines.append(f"| [{b_name}] | {b_data['count']} | {b_data['fraction']:.1%} |")

    lines.extend([
        "",
        "## 2. Homology Clustering (Task 2)",
        f"- **Identity Threshold:** {clustering['metadata']['identity_threshold']:.1%}",
        f"- **Coverage Threshold:** {clustering['metadata']['coverage_threshold']:.1%}",
        "- **Short Peptide (<20 aa) Fallback:** k-mer Jaccard (k=3, threshold=0.25) & normalized edit distance.",
        f"- **Total Homology Clusters:** {clustering['metadata']['total_clusters']}",
        f"- **Singletons:** {clustering['metadata']['singletons']}",
        f"- **Cluster Assignments SHA-256:** `{clustering['metadata']['cluster_assignments_sha256']}`",
        "",
        "## 3. Nested Group-Held-Out Splits (Task 3)",
        "- **Outer Evaluation:** 5-fold `StratifiedGroupKFold` on cluster IDs.",
        "- **Inner Partitions:** Outer training clusters are split into disjoint Inner-Train and Inner-Calibration partitions (25% calibration clusters).",
        f"- **Splits SHA-256:** `{baselines['metadata']['splits_sha256']}`",
        "",
        "## 4. Strict Leakage Audit (Task 4)",
        f"- **Audit Status:** **`{audit['status']}`**",
        f"- **Pairwise Cross-Partition Identity Comparisons:** {audit['pairwise_cross_comparisons_performed']:,}",
        f"- **Maximum Cross-Partition Identity Observed:** {audit['max_cross_partition_identity_observed']:.2%}",
        f"- **Identity Threshold Enforced:** {audit['identity_threshold_enforced']:.2%}",
        f"- **Verdict:** {audit['verdict']}",
        "",
        "## 5. Frozen Reference Baseline Metrics (Task 5 & 6)",
        "",
        "| Model | Feature Set | ROC-AUC (95% CI) | PR-AUC (95% CI) | Balanced Acc | MCC | Brier Score | Adaptive ECE (15 bins) |",
        "|---|---|---|---|---|---|---|---|",
        f"| Antigenicity (RF) | 125-D ACC | {m_ant_rf['auroc']:.4f} [{ci_ant_rf['auroc']['ci_lower_95']:.4f} - {ci_ant_rf['auroc']['ci_upper_95']:.4f}] | {m_ant_rf['auprc']:.4f} [{ci_ant_rf['auprc']['ci_lower_95']:.4f} - {ci_ant_rf['auprc']['ci_upper_95']:.4f}] | {m_ant_rf['balanced_accuracy']:.4f} | {m_ant_rf['mcc']:.4f} | {m_ant_rf['brier_score']:.4f} | {m_ant_rf['adaptive_ece_15bins']:.4f} |",
        f"| Antigenicity (GB) | 125-D ACC | {m_ant_gb['auroc']:.4f} [{ci_ant_gb['auroc']['ci_lower_95']:.4f} - {ci_ant_gb['auroc']['ci_upper_95']:.4f}] | {m_ant_gb['auprc']:.4f} [{ci_ant_gb['auprc']['ci_lower_95']:.4f} - {ci_ant_gb['auprc']['ci_upper_95']:.4f}] | {m_ant_gb['balanced_accuracy']:.4f} | {m_ant_gb['mcc']:.4f} | {m_ant_gb['brier_score']:.4f} | {m_ant_gb['adaptive_ece_15bins']:.4f} |",
        f"| Allergenicity (RF) | 50-D AAC+PCP | {m_alg_rf['auroc']:.4f} [{ci_alg_rf['auroc']['ci_lower_95']:.4f} - {ci_alg_rf['auroc']['ci_upper_95']:.4f}] | {m_alg_rf['auprc']:.4f} [{ci_alg_rf['auprc']['ci_lower_95']:.4f} - {ci_alg_rf['auprc']['ci_upper_95']:.4f}] | {m_alg_rf['balanced_accuracy']:.4f} | {m_alg_rf['mcc']:.4f} | {m_alg_rf['brier_score']:.4f} | {m_alg_rf['adaptive_ece_15bins']:.4f} |",
        f"| Toxicity (RF) | 50-D AAC+PCP | {m_tox_rf['auroc']:.4f} [{ci_tox_rf['auroc']['ci_lower_95']:.4f} - {ci_tox_rf['auroc']['ci_upper_95']:.4f}] | {m_tox_rf['auprc']:.4f} [{ci_tox_rf['auprc']['ci_lower_95']:.4f} - {ci_tox_rf['auprc']['ci_upper_95']:.4f}] | {m_tox_rf['balanced_accuracy']:.4f} | {m_tox_rf['mcc']:.4f} | {m_tox_rf['brier_score']:.4f} | {m_tox_rf['adaptive_ece_15bins']:.4f} |",
        "",
        "### Length-Bin Breakdown (Antigenicity RF Baseline)",
        "| Length Bin | Sample Count | Positives | AUROC | Balanced Acc | Sensitivity | Specificity | PPV | NPV |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    ant_strat = baselines["models"]["antigenicity_acc_rf"]["length_stratified"]
    for b_name, _, _ in LENGTH_BINS:
        sd = ant_strat[b_name]
        auc_str = f"{sd['auroc']:.4f}" if sd["auroc"] is not None else "N/A"
        bal_str = f"{sd['balanced_accuracy']:.4f}" if sd["balanced_accuracy"] is not None else "N/A"
        sens_str = f"{sd['sensitivity']:.4f}" if sd["sensitivity"] is not None else "N/A"
        spec_str = f"{sd['specificity']:.4f}" if sd["specificity"] is not None else "N/A"
        ppv_str = f"{sd['ppv']:.4f}" if sd["ppv"] is not None else "N/A"
        npv_str = f"{sd['npv']:.4f}" if sd["npv"] is not None else "N/A"
        lines.append(f"| [{b_name}] | {sd['sample_count']} | {sd['positive_count']} | {auc_str} | {bal_str} | {sens_str} | {spec_str} | {ppv_str} | {npv_str} |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline()
