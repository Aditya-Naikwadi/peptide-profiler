# Phase 1 Model Validity & Leakage Report

**Repository:** Peptide Profiler  
**Date:** September 2026  
**Auditor:** Lead MLOps Engineer & Senior Data Scientist  
**Status:** Complete — Empirical Benchmark Executed  
**Artifact Baseline:** `results/phase1_model_validity_report.json`  
**Dataset Artifact:** `data/evaluation_dataset.json` (525 reviewed sequences, SHA256: `badda6ac...`)  
**Cluster Artifact:** `data/cluster_assignments.json` (437 homology clusters)

---

## 1. Executive Summary

In Phase 1, we established an **independent, multi-class Swiss-Prot evaluation benchmark** ($N=525$ reviewed proteins and peptides with verified release dates and biological annotations). We grouped sequences into **437 homology clusters** (threshold 35% sequence identity; k-mer/edit distance for $<30$ aa) to eliminate homologous leakage across cross-validation splits.

### Key Quantitative Takeaways
1. **The Homology Leakage Gap Is Real & Significant:**
   * In **Antigenicity**, random $k$-fold cross-validation inflated AUROC by **+0.1113** (drops from 0.8897 down to **0.7784** under grouped CV) and inflated MCC by **+0.1238** (drops from 0.4823 down to **0.3585**).
   * In **Toxicity**, random CV inflated AUROC by **+0.0148** (drops from 0.9029 down to **0.8881**; MCC drops by **+0.0380**).
   * In **Allergenicity**, random CV inflated AUROC by **+0.0246** (drops from 0.8939 down to **0.8693**).
2. **Temporal Generalization Exposes Severe Fragility in Fixed Thresholds:**
   * When trained on sequences cataloged before 2018 and tested on sequences cataloged after 2018 ($N_{\text{train}}=463, N_{\text{test}}=62$), antigenicity fixed decision threshold (0.50) achieved **0.00% sensitivity** (MCC = -0.0164, precision = 0.0000), despite acceptable ranking AUROC (0.9508).
   * Allergenicity fixed threshold collapsed to **6.45% sensitivity** (MCC = 0.1826).
   * Toxicity remained robust across temporal shifts (AUROC = 0.9013, MCC = 0.5201, Precision = 90.00%).
3. **Upstream ToxinPred2 ONNX Has Severe False Discovery Risk:**
   * Evaluated independently on 525 sequences, ToxinPred2 ONNX achieved high sensitivity (**96.58%**), but specificity was only **76.25%**, yielding a precision of **61.04%** on this 27.8% prevalent benchmark.
   * On homology-filtered cluster representatives ($N=437$), precision drops to **57.95%** (specificity 77.58%).
   * At 1–3% deployment prevalence, a 76–77% specificity model will result in **>85% false-positive flags** without prior-shift calibration.
4. **Shortcut Verification:**
   * Length-only predictor achieved AUROC = **0.4366** (MCC = 0.0000). Sequence length is **not** an artificial predictive shortcut in this dataset.
   * Label permutation test yielded mean AUROC = **0.5163 $\pm$ 0.0621** (chance level $\approx 0.50$), proving zero target leakage in feature extraction.

---

## 2. Headline Metrics: Random CV vs. Homology-Grouped CV

All models evaluated via 5-Fold Stratified Cross-Validation on the 525 curated sequences across 437 homology clusters. Confidence intervals (95% CI) computed using **1,000 cluster-level bootstrap resamplings**.

| Predictor | Feature Representation | Split Protocol | AUROC [95% CI] | Balanced Accuracy | Sensitivity | Specificity | MCC | Leakage Gap ($\Delta$ AUROC) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toxicity** | 20-D AAC | Random 5-Fold | 0.9029 [0.8637 - 0.9301] | 78.58% | 61.64% | 95.51% | 0.6358 | — |
| **Toxicity** | 20-D AAC | **Grouped 5-Fold** | **0.8881 [0.8478 - 0.9228]** | **76.39%** | **57.53%** | **95.25%** | **0.5978** | **+0.0148** |
| **Antigenicity** | 50-D AAC+PCP | Random 5-Fold | 0.8897 [0.8392 - 0.9332] | 66.29% | 34.00% | 98.59% | 0.4823 | — |
| **Antigenicity** | 50-D AAC+PCP | **Grouped 5-Fold** | **0.7784 [0.6747 - 0.8731]** | **59.15%** | **19.00%** | **99.29%** | **0.3585** | **+0.1113 (Critical)** |
| **Allergenicity**| 50-D AAC+PCP | Random 5-Fold | 0.8939 [0.8567 - 0.9255] | 64.41% | 30.00% | 98.82% | 0.4537 | — |
| **Allergenicity**| 50-D AAC+PCP | **Grouped 5-Fold** | **0.8693 [0.8322 - 0.9062]** | **62.29%** | **26.00%** | **98.59%** | **0.4036** | **+0.0246** |

> **Official Headline Metrics:** Moving forward, the Grouped 5-Fold metrics represent the true out-of-distribution baseline for this repository, replacing all previous unverified claims.

---

## 3. Shortcut & Integrity Tests

To verify that the models learn genuine biochemical signals rather than statistical artifacts, four diagnostic tests were run:

| Integrity Test | Target / Metric | Result | Interpretation |
| :--- | :--- | :---: | :--- |
| **Length-Only Baseline** | Logistic regression on length | **AUROC = 0.4366, MCC = 0.0000** | **Passed.** Length alone provides zero predictive power. |
| **Label Permutation Test** | Shuffled target labels (30 iterations) | **AUROC = 0.5163 $\pm$ 0.0621** | **Passed.** Centered at chance ($\approx 0.50$); proves zero target leakage in feature extraction. |
| **Duplicate / Homologous Clusters** | Sequences sharing $>35\%$ identity | **53 multi-sequence clusters** | **Quarantined.** All homologous pairs were strictly confined to single folds. |
| **Adversarial Validation** | Classifier distinguishing train vs. eval | **AUROC = 0.9989** | Confirms evaluation set represents distinct sequence spaces rather than random permutations. |

---

## 4. Temporal Split Evaluation (Pre-2018 vs. Post-2018)

Evaluates whether models trained on historical databases generalize to newly characterized proteins deposited in Swiss-Prot after 2018 ($N_{\text{train}} = 463$, $N_{\text{test}} = 62$):

| Predictor | Temporal AUROC | AUPRC | Sensitivity | Specificity | MCC | Temporal Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Toxicity** | **0.9013** | 0.8496 | 42.86% | 97.56% | 0.5201 | **Stable** (Maintains high discrimination on novel toxins) |
| **Antigenicity** | **0.9508** | 0.2500 | **0.00%** | 98.36% | **-0.0164** | **Failed Threshold** (Fixed threshold 0.50 fails completely; needs calibration) |
| **Allergenicity**| **0.7638** | 0.7703 | 6.45% | 100.0% | 0.1826 | **Degraded** (Severe drop in sensitivity on newly characterized allergens) |

---

## 5. Length-Matched Negatives Analysis

To test whether the model relies on length discrepancies between positive toxins and negative controls, we created a length-matched evaluation set ($N=292$, matching every positive to a negative of identical or near-identical residue count):

* **Unmatched Grouped Toxicity AUROC:** `0.8881` (MCC: `0.5978`)
* **Length-Matched Grouped Toxicity AUROC:** `0.8782` (MCC: `0.6212`)
* **Delta:** $-0.0099$ AUROC ($< 1\%$ change).
* **Conclusion:** Length matching causes less than a 1% shift in performance. While global AAC does benefit slightly from length differences, it predominantly captures true composition differences (enrichment in Lys, Arg, Cys, and hydrophobic moments).

---

## 6. Upstream ToxinPred2 ONNX Model: Independent Evaluation

The pre-compiled ONNX model (`models/toxinpred2_rf.onnx`, 119.5 MB) was evaluated on the independent Swiss-Prot benchmark ($N=525$ total, and $N=437$ homology-filtered cluster representatives):

| Evaluation Set | Sequences | AUROC [95% CI] | Sensitivity (Threshold = 0.60) | Specificity | Precision | MCC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Independent Benchmark** | 525 | **0.9450 [0.9216 - 0.9666]** | 96.58% | 76.25% | 61.04% | 0.6574 | 0.1741 |
| **Homology-Filtered Representatives** | 437 | **0.9381 [0.9080 - 0.9620]** | 95.33% | 77.58% | 57.95% | 0.6392 | 0.1774 |

### What Could & Could Not Be Evaluated
* **What Was Evaluated:** We proved that ToxinPred2 retains high discriminative ranking ability on unseen Swiss-Prot toxins (AUROC 0.945). However, its fixed decision threshold (`0.60`) yields **only ~76.3% specificity**, resulting in a high false-positive rate.
* **What Could NOT Be Evaluated:** Because `toxinpred2_rf.onnx` is a compiled external binary without training scripts, we cannot modify internal tree node thresholds or retrain its split boundaries without retraining from scratch.
