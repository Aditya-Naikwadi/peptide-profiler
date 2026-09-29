# Phase 2 Calibration & Prevalence Report

**Repository:** Peptide Profiler  
**Date:** September 2026  
**Auditor:** Lead MLOps Engineer & Senior Data Scientist  
**Status:** Complete — Empirical Benchmark Executed  
**Artifact Baseline:** `results/phase2_calibration_and_prevalence_report.json`  
**Dataset Artifact:** `data/evaluation_dataset.json` (525 reviewed sequences across all length bins)

---

## 1. Executive Summary & Calibration Impact

Uncalibrated biological classifiers output heuristic pseudo-probabilities that do not reflect empirical likelihoods of toxicity, antigenicity, or allergenicity. Furthermore, evaluating balanced models (trained at 50% prevalence) without Bayesian prior correction creates severe false-positive discovery rates in production screening libraries where true toxic/allergenic sequences represent only **1% to 3%** of candidates.

### Key Highlights
1. **Dramatic Calibration Gain:**
   * Fitting **Platt Scaling** on group-held-out folds reduced Expected Calibration Error (ECE) for **Toxicity** from **27.39% down to 3.56% (an 87.0% error reduction)**, and cut the Brier score by more than half (0.1741 $\rightarrow$ 0.0818).
   * For **Allergenicity**, ECE dropped from **5.19% down to 1.62% (a 68.8% reduction)**.
2. **The Non-ML Stakeholder Production Reality:**
   * At an uncalibrated default cutoff of 0.60, **ToxinPred2 ONNX** yields **76.25% specificity**. 
   * When deployed in a screening library at **1% real-world prevalence**, **96 out of every 100 positive alarms raised by the model will be false positives** (PPV = 3.95%).
   * At **2% prevalence**, **92 out of 100 alarms are false positives** (PPV = 7.66%).
   * At **3% prevalence**, **89 out of 100 alarms are false positives** (PPV = 11.17%).
3. **Length Disparities Uncovered:**
   * **Short Peptides (< 15 aa):** Specificity collapses to **44.00%** in toxicity. Very short peptides trigger substantial false-positive alarms due to sparse amino acid composition vectors.
   * **Medium Peptides (15–50 aa):** Peak performance zone for ToxinPred2 (AUROC = **0.9830**, PR-AUC = **0.9821**, MCC = **0.7421**, Precision@10 = 100%).
   * **Antigenicity Deficit:** Antigenicity models fail entirely on sequences $< 50$ aa (Sensitivity = 0.0%), functioning only on medium-to-large proteins (>50–200 aa).

---

## 2. Model Calibration Benchmarks (Group-Held-Out Folds)

Evaluated across out-of-fold predictions using 5-Fold `StratifiedGroupKFold` on the 525-sequence independent benchmark.

| Predictor | Baseline Brier Score | Baseline ECE (Error %) | Platt Calibrated Brier | Platt Calibrated ECE | ECE Error Reduction | Isotonic Calibrated ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toxicity (ONNX RF)** | 0.1741 | 27.39% | **0.0818** | **3.56%** | **-87.00%** | 3.69% |
| **Antigenicity (Grouped RF)** | 0.1201 | 6.88% | **0.1191** | **5.70%** | **-17.15%** | 2.98% |
| **Allergenicity (Grouped RF)** | 0.1102 | 5.19% | **0.1057** | **1.62%** | **-68.79%** | 3.51% |

---

## 3. Stakeholder Production Reality Table: True vs. False Alarms in Deployment

This table directly translates statistical metrics into operational drug-discovery impact for non-ML decision-makers. It answers:
> *"If our screening pipeline flags 100 peptide candidates as Toxic, Allergenic, or Antigenic, how many of those 100 flags are true biological positives vs. wasted laboratory assays?"*

| Target Predictor | Deployment Prevalence ($\pi$) | Sensitivity (Recall) | Specificity | True Positive Rate / Precision (PPV) | **True Positives per 100 Flags** | **False Alarms per 100 Flags (Wasted Assays)** | Business Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Toxicity** (Cutoff 0.60) | **1%** | 96.58% | 76.25% | **3.95%** | **4** | **96** | **96% of toxic alarms are false positives** |
| **Toxicity** (Cutoff 0.60) | **2%** | 96.58% | 76.25% | **7.66%** | **8** | **92** | 92% of toxic alarms are false positives |
| **Toxicity** (Cutoff 0.60) | **3%** | 96.58% | 76.25% | **11.17%** | **11** | **89** | 89% of toxic alarms are false positives |
| **Allergenicity** (Cutoff 0.50) | **1%** | 26.00% | 98.59% | **15.68%** | **16** | **84** | 84% of allergen flags are false alarms |
| **Allergenicity** (Cutoff 0.50) | **2%** | 26.00% | 98.59% | **27.32%** | **27** | **73** | 73% of allergen flags are false alarms |
| **Allergenicity** (Cutoff 0.50) | **3%** | 26.00% | 98.59% | **36.29%** | **36** | **64** | 64% of allergen flags are false alarms |
| **Antigenicity** (Cutoff 0.50) | **1%** | 19.00% | 99.29% | **21.38%** | **21** | **79** | 79% of antigen flags are false alarms |
| **Antigenicity** (Cutoff 0.50) | **2%** | 19.00% | 99.29% | **35.46%** | **35** | **65** | 65% of antigen flags are false alarms |
| **Antigenicity** (Cutoff 0.50) | **3%** | 19.00% | 99.29% | **45.43%** | **45** | **55** | 55% of antigen flags are false alarms |

---

## 4. Length-Stratified Performance Breakdown

Evaluated across all 525 independent Swiss-Prot entries partitioned into five biophysical length bins:

### 4.1 Toxicity Model (ToxinPred2 ONNX)
| Length Bin | Sample Count ($N$) | Positives ($N$) | Local Prevalence | AUROC | PR-AUC | Sensitivity | Specificity | MCC | Precision@10 | Enrichment Factor ($EF_{10}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 15 aa** | 50 | 25 | 50.0% | 0.8304 | 0.8528 | **100.0%** | **44.00%** | 0.5311 | 100.0% | 2.00x |
| **15–50 aa** | 72 | 35 | 48.6% | **0.9830** | **0.9821** | 97.14% | 75.68% | **0.7421** | 100.0% | 2.06x |
| **50–200 aa** | 233 | 50 | 21.5% | **0.9606** | 0.9226 | 98.00% | 71.04% | 0.5713 | 100.0% | **4.66x** |
| **200–500 aa** | 125 | 18 | 14.4% | 0.9255 | 0.8884 | 88.89% | 92.52% | 0.7257 | 100.0% | **6.94x** |
| **> 500 aa** | 45 | 18 | 40.0% | 0.9691 | 0.9574 | 94.44% | 77.78% | 0.7078 | 100.0% | 2.50x |

### 4.2 Antigenicity Model (50-D AAC+PCP Grouped RF)
| Length Bin | Sample Count ($N$) | Positives ($N$) | Local Prevalence | AUROC | PR-AUC | Sensitivity | Specificity | MCC | Precision@10 | Enrichment Factor ($EF_{10}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 15 aa** | 50 | 0 | 0.0% | 0.5000 | 0.0000 | 0.00% | 100.0% | 0.0000 | 0.0% | 1.00x |
| **15–50 aa** | 72 | 1 | 1.4% | 0.1972 | 0.0172 | **0.00%** | 98.59% | -0.0141 | 0.0% | 0.00x |
| **50–200 aa** | 233 | 75 | 32.2% | **0.7803** | **0.7234** | 18.67% | 99.37% | 0.3433 | 90.0% | **2.80x** |
| **200–500 aa** | 125 | 20 | 16.0% | **0.8181** | 0.6540 | 25.00% | 100.0% | 0.4677 | 70.0% | **4.38x** |
| **> 500 aa** | 45 | 4 | 8.9% | 0.9390 | 0.5111 | 0.00% | 97.56% | -0.0471 | 40.0% | 4.50x |

### 4.3 Allergenicity Model (50-D AAC+PCP Grouped RF)
| Length Bin | Sample Count ($N$) | Positives ($N$) | Local Prevalence | AUROC | PR-AUC | Sensitivity | Specificity | MCC | Precision@10 | Enrichment Factor ($EF_{10}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 15 aa** | 50 | 0 | 0.0% | 0.5000 | 0.0000 | 0.00% | 100.0% | 0.0000 | 0.0% | 1.00x |
| **15–50 aa** | 72 | 1 | 1.4% | 0.0845 | 0.0152 | 0.00% | 100.0% | 0.0000 | 0.0% | 0.00x |
| **50–200 aa** | 233 | 35 | 15.0% | **0.8691** | 0.5205 | 11.43% | 99.49% | 0.2693 | 60.0% | **3.99x** |
| **200–500 aa** | 125 | 43 | 34.4% | **0.8335** | **0.7184** | 34.88% | 95.12% | 0.3970 | 80.0% | **2.33x** |
| **> 500 aa** | 45 | 21 | 46.7% | 0.6806 | 0.7421 | 33.33% | 95.83% | 0.3806 | 90.0% | 1.93x |

---

## 5. Key Engineering Insights

1. **Short Peptides Require Specificity Guardrails:**
   * ToxinPred2 ONNX maintains 100% recall on short peptides $<15$ aa, but drops to **44.0% specificity**. Because short peptides produce sparse zero-heavy composition vectors, non-toxic short peptides frequently trigger false alarms.
   * **Recommendation for Phase 3/4:** Implement an explicit low-confidence abstention flag for sequences $<15$ aa unless supported by exact motif matching.
2. **Prior-Shift Recalibration Is Essential for Deployment:**
   * Uncalibrated probabilities from models trained on balanced datasets overestimate the probability of toxicity by an order of magnitude.
   * Applying the Bayesian prior shift function ([`bayesian_prior_shift`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/calibration.py)) maps raw scores to true deployment probabilities, ensuring decision thresholds are aligned with screening realities.

---

## 6. Acceptance Criteria Checklist

- [x] **Group-Held-Out Calibration:** Platt scaling and Isotonic regression fitted strictly on out-of-fold predictions.
- [x] **Brier Score & ECE Quantified:** ECE reduced by 87.0% for Toxicity (27.39% $\rightarrow$ 3.56%) and 68.8% for Allergenicity.
- [x] **Configurable Bayesian Prior Shift:** Implemented in `src/calibration.py` with validation unit tests passing.
- [x] **Stakeholder Production Table:** Clear table showing true positives vs. false alarms per 100 flags at 1%, 2%, and 3% prevalence.
- [x] **Length-Stratified Metrics:** Complete evaluation across $<15$, $15-50$, $50-200$, $200-500$, and $>500$ aa bins with PR-AUC, MCC, Precision@10, and Enrichment Factors reported.
- [x] **Regression Suite Maintained:** 23/23 unit and integration tests passing in `tests/test_pipeline.py`.
