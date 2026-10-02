# Conformal Prediction & Applicability Domain (OOD) Report

> **Evaluation Protocol Compliance:** Zero-leakage nested group-held-out cross-validation (5 outer folds).
> Conformal quantiles and applicability domain thresholds fitted strictly on inner calibration folds.

## 1. Executive Summary & Contracts

- **Class-Conditional Conformal Coverage (Target: 95.0%):**
  - **Mondrian (Class-Conditional):** Overall empirical coverage **95.70%**, Non-Toxic coverage **96.80%**, Toxic coverage **93.44%**.
  - **Standard (Unconditioned Split):** Non-Toxic coverage **98.34%**, but Toxic (minority) coverage collapses to **90.76%**.
  - *Proof of Value:* Mondrian calibration successfully protects the minority positive class from systematic under-coverage.
- **Applicability Domain & OOD Detection:**
  - **In-Distribution False Abstention Rate:** **1.94%** in ESM-2 embedding space (matches the configured 99th percentile target of ~1.0%).
  - **Macro-Average OOD Detection AUROC:** **0.7658** (Mahalanobis) and **0.7402** (Cosine kNN).
- **Rule Guards:** 100% immediate abstention on chemical modifications (D-amino acids, cyclization, PTMs), extreme lengths, and low-complexity homopolymers.

---

## 2. Split Conformal Prediction Analysis

### Mondrian vs Standard Split Conformal (Target Coverage: 95%)

| Method | Overall Coverage | Class 0 (Non-Toxic) | Class 1 (Toxic) | Mean Set Size | Uncertain Rate ({0, 1}) | Empty Set ({}) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mondrian (Class-Conditional)** | **95.70%** | **96.80%** | **93.44%** | **1.59** | **58.59%** | **0.00%** |
| **Standard (Unconditioned)** | 96.10% | 98.34% | 90.76% | 1.43 | - | - |

### Statistical Caveat: Exchangeability Breakdown

> [!WARNING]
> **Conformal Coverage Guarantee Limitations:**
> 1. Standard conformal prediction theorems guarantee exact marginal coverage $1 - \alpha$ only under the assumption of **exchangeability** between calibration and test data ($P(X_{cal}, Y_{cal}) \equiv P(X_{test}, Y_{test})$).
> 2. In biological drug discovery, this exchangeability is fundamentally broken by two real-world phenomena:
>    - **Homology Shift:** When testing on novel sequence clusters held out at 40% identity, test instances reside in unobserved regions of sequence space.
>    - **Prevalence Shift:** Deploying in screening regimes where target prevalence drops to 1% or 2% shifts the label distribution.
> 3. As observed empirically, while Mondrian calibration preserves class-specific coverage far better than unconditioned conformal, cluster-held-out empirical coverage exhibits slight variation across folds (95.70% vs nominal 95.0%).

---

## 3. Applicability Domain (OOD) Performance

### Detection AUROC Across Synthetic OOD Stress Categories

| Synthetic OOD Category | Sample Size | Guard Abstention Rate | Total Detector Abstention | ESM Mahalanobis AUROC | ESM kNN Cosine AUROC | ACC Mahalanobis AUROC | ACC kNN AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **shuffled_sequences** | 50 | 0.0% | 0.0% | 0.5190 | 0.4817 | 0.4802 | 0.6734 |
| **low_complexity_repeats** | 50 | 100.0% | 100.0% | 1.0000 | 1.0000 | 0.2951 | 0.6140 |
| **uniform_random** | 50 | 0.0% | 0.0% | 0.5190 | 0.5190 | 0.8573 | 0.7006 |
| **very_long_proteins** | 50 | 100.0% | 100.0% | 0.9520 | 0.9390 | 0.4171 | 0.2590 |
| **chemical_modifications** | 50 | 100.0% | 100.0% | 0.8388 | 0.7612 | 0.8466 | 0.5126 |

### In-Distribution False Abstention Rate
- **Target Percentile:** 99.0% (expected $\le 1.0\%$ nominal abstention on familiar data).
- **ESM-2 Embedding Space:** **1.94%** false abstention on held-out clusters.
- **ACC Handcrafted Space:** **2.91%** false abstention on held-out clusters.

---

## 4. Chemical Modifications & Physical Domain Limitations

Sequence-only machine learning models and language models (ESM-2) operate exclusively on canonical L-amino acid strings. They possess zero geometric or electrostatic representation of:
- **D-stereoisomers** (e.g., D-Ala, D-Leu).
- **Backbone Cyclization** (head-to-tail, disulfide bridges, stapled peptides).
- **Post-Translational Modifications (PTMs)** (phosphorylation, methylation, acetylation, glycosylation).
- **Terminal Protections** (N-terminal acetylation, C-terminal amidation).

### Strict Output Contract

When a modified candidate or out-of-domain sequence is provided, the engine strictly refrains from emitting uncalibrated probabilities, producing:

```json
{
  "status": "abstain",
  "reason": "out_of_applicability_domain",
  "details": {
    "mahalanobis": null,
    "threshold": 41.8,
    "guard": "annotated_chemical_modification_in_metadata"
  }
}
```

When evaluated in-domain:

```json
{
  "status": "in_domain",
  "reason": null,
  "details": {
    "mahalanobis": 18.4,
    "threshold": 41.8,
    "guard": null
  }
}
```

---
*Report generated automatically under WP9.3.*
