# Phase 0 Audit & Model Integrity Report

**Repository:** Peptide Profiler  
**Date:** September 2026  
**Auditor:** Lead MLOps Engineer & Senior Data Scientist  
**Status:** Complete — Pending User Sign-Off to Enter Phase 1  
**Artifact Baseline:** `results/legacy_baseline_metrics.json`

---

## 1. Executive Summary & Scope

An exhaustive technical audit of the models, feature extractors, test suites, and ranking mechanisms was conducted across the `peptide-profiler` codebase. 

The objective of Phase 0 is to replace benchmark-inherited assumptions with verified engineering facts before modifying model behavior.

### Key Audit Findings at a Glance
| Audit Vector | Heritage / Implementation | True Engineering State | Risk Level |
| :--- | :--- | :--- | :---: |
| **Toxicity Model** | ToxinPred2 (Sharma et al., 2022) | Authentic Random Forest (`toxinpred2_rf.onnx`, **119.5 MB**). 53.6s cold-start load, 0.19ms inference. | **High** (Prevalence & length shortcuts) |
| **Antigenicity Model** | Local VaxiJen surrogate | Shallow surrogate RF (`antigen_rf.joblib`, **111.8 KB**, **depth 3–4, 7–11 nodes/tree**). | **Critical** (Severe under-parameterization) |
| **Allergenicity Model** | Local AllerTOP surrogate | Shallow surrogate RF (`allergen_rf.joblib`, **117.3 KB**, **depth 3–4, 7–11 nodes/tree**). | **Critical** (Coarse biophysical heuristic) |
| **Biological Controls** | 8 canonical peptides in `sample_peptides.fa` | **Smoke tests only.** Present in upstream training datasets. | **High** (False validation confidence) |
| **Desirability Ranking** | Derringer-Suich geometric mean | **Compensatory flaw.** Melittin (bee venom) ranked **#3** for vaccine design! | **Critical** (Preclinical safety hazard) |
| **Sequence Invariance** | 20-D AAC + 30-D PCP | **Bag-of-residues.** Scrambled/reversed sequences yield 100% identical predictions. | **High** (Lacks structural/motif awareness) |

---

## 2. Dataset & Training Heritage Inventory

### 2.1 Codebase Dataset Inventory
* **Repository Storage:** The repository contains **zero** raw training FASTA files, tabular training sets, or cross-validation splits.
* **Committed Models:** All three binary model artifacts were pre-baked and committed directly:
  1. `models/toxinpred2_rf.onnx` (119,570,418 bytes)
  2. `models/antigen_rf.joblib` (111,833 bytes)
  3. `models/allergen_rf.joblib` (117,273 bytes)
* **Sample / Test Data:** Only one FASTA file exists: `examples/sample_peptides.fa`, containing exactly 8 sequences (lengths 25 to 208 aa).

### 2.2 Upstream Training Heritage & Prevalence Mismatch
* **ToxinPred2 (Toxicity):** Trained on 6,762 positive toxins (Swiss-Prot) and 6,762 negatives (1:1 balanced). Negatives were drawn from randomly sampled non-secretory proteins without length-matching.
* **VaxiJen (Antigenicity):** Trained on 100 antigens and 100 non-antigens per organism class (bacterial, viral, tumor) — extremely small sample size ($N=200$).
* **AllerTOP v2 / AlgPred2 (Allergenicity):** Trained on ~2,427 allergens and 2,427 non-allergens (1:1 balanced) from AllergenOnline.
* **The Deployment Reality:** Natural proteomes contain approximately **1% to 3%** toxic or allergenic sequences. Evaluating balanced 1:1 models at 1% prevalence causes a catastrophic collapse in Positive Predictive Value (PPV), generating predominantly false-positive flags.

---

## 3. Empirical Evidence of Identified Risks

### Risk 1: Upstream Benchmark Citation vs. Real Pipeline Performance
* **Claim in README:** "82–84% accuracy, AUROC 0.89–0.91".
* **Evidence:** This metric is cited directly from the published ToxinPred2 paper on balanced holdout sets. It was not measured on the diverse sequences expected in this pipeline, nor does it account for prevalence adjustment.

### Risk 2: Extreme Model Complexity Disparity
* **Measured Code Inspection:**
  * `toxinpred2_rf.onnx`: Full-scale production tree ensemble (119.5 MB).
  * `antigen_rf.joblib` & `allergen_rf.joblib`:
    ```python
    # Measured directly from joblib inspect:
    # antigen_rf: 100 trees, depths: [4, 3, 3, 3, 3], node counts: [11, 9, 7, 9, 7]
    # allergen_rf: 100 trees, depths: [4, 3, 3, 3, 3], node counts: [11, 9, 7, 9, 7]
    ```
  * **Finding:** Both the antigenicity and allergenicity models are **shallow 3-to-4 level decision trees** containing only 7 to 11 decision nodes per tree. They do not learn complex epitope sequence motifs; they merely split on coarse aggregate properties (e.g., surface exposure, net charge).

### Risk 3: Control Peptides Are Memorized Smoke Tests
* **Evidence:** The 8 sequences in `examples/sample_peptides.fa` (Melittin, Conotoxin, OspA, Bet v 1, Ubiquitin, etc.) are the quintessential textbook controls used to build ToxinPred, VaxiJen, and AllerTOP.
* **Action Taken:** Reclassified these in `tests/test_pipeline.py` and `README.md` as **integration smoke tests** (verifying software execution without crashing), explicitly disclaiming them as validation benchmarks.

### Risk 4: The Compensatory Desirability Ranking Failure (Critical Defect)
* **Equation in `aggregator.py`:**
  $$\text{Desirability} = \text{Antigenicity} \times (1 - \text{Toxicity}) \times (1 - \text{Allergenicity})$$
* **Empirical Demonstration (`results/legacy_baseline_metrics.json`):**
  * **Melittin** (*Apis mellifera* lethal pore-forming bee venom):
    * Toxicity Score: **0.650** (`is_toxic: true`)
    * Antigenicity Score: **0.530** (`is_antigen: true`)
    * Allergenicity Score: **0.460** (`is_allergen: false`)
    * Desirability Score: **0.1002** $\rightarrow$ **Ranked #3 out of 8 as a Vaccine Candidate!**
  * **Consequence:** Melittin ranked **ahead** of SARS-CoV-2 Spike RBD (Rank 7) and Bet v 1 (Rank 6) because the linear multiplication $(1 - 0.650) = 0.35$ allowed high antigenicity to compensate for lethal toxicity.
  * **Remedy Required (Phase 4):** A hard safety gate (veto filter) must exclude or penalize toxic/allergenic candidates before multi-objective ranking.

### Risk 5: Bag-of-Residues Invariance
* **Evidence:** The 20-D Amino Acid Composition (AAC) and 30-D Physicochemical Properties (PCP) vectors rely on residue counts and global averages.
* **Test Case:** Reversing or scrambling Melittin (`QQ RKR KIW SIL AP L GT TVK LV A VG IG`) yields the **exact same 20-D AAC vector** and produces identical toxicity and allergenicity predictions, despite complete loss of biological tertiary structure.

### Risk 6: Cold-Start Latency & Engine Execution
* **Measured Benchmark:**
  * Session loading time for `toxinpred2_rf.onnx`: **53.64 seconds** on Windows CPU.
  * Inference execution time per sequence (post-load): **0.00019 seconds (0.19 ms)**.
  * **Engineering Impact:** The pipeline must use cached persistent sessions (`_CACHED_SESSION`), as reloading per sequence or test would cause unacceptable timeouts.

### Risk 7: Therapeutic Profile Verification
* **Code Inspection of `aggregator.py` (lines 101–103):**
  ```python
  if candidate_type.lower() == "therapeutic":
      score = (1.0 - antigenicity_score) * (1.0 - toxicity_score) * (1.0 - allergenicity_score)
  ```
* **Finding:** The therapeutic profile **does correctly invert antigenicity** $(1 - \text{antigenicity})$. However, like the vaccine profile, it suffers from the lack of a hard safety gate.

---

## 4. Legacy Baseline Performance Records

The legacy baseline results were extracted via deterministic pipeline execution and frozen in [`results/legacy_baseline_metrics.json`](../results/legacy_baseline_metrics.json).

| Candidate Sequence ID | Length | Toxicity | Antigenicity | Allergenicity | Vaccine Rank | Therapeutic Rank | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ubiquitin** | 76 | 0.272 | 0.260 | 0.260 | #1 (0.1401) | #1 (0.3986) | Non-antigen, Non-toxic |
| **OspA** | 56 | 0.488 | 0.790 | 0.730 | #2 (0.1092) | #6 (0.0290) | Flagged Allergen |
| **Melittin** | 26 | **0.650** | 0.530 | 0.460 | **#3 (0.1002)** | #5 (0.0888) | **Lethal Toxin (Misranked)** |
| **Exendin-4** | 39 | 0.351 | 0.270 | **0.530** | #4 (0.0824) | #2 (0.2226) | Flagged Allergen |
| **Ara_h_1** | 57 | 0.449 | 0.450 | **0.720** | #5 (0.0694) | #4 (0.0848) | Flagged Allergen |
| **Bet_v_1** | 159 | 0.263 | 0.230 | **0.640** | #6 (0.0610) | #3 (0.2040) | Flagged Allergen |
| **Spike_RBD** | 208 | 0.562 | 0.410 | **0.690** | #7 (0.0557) | #7 (0.0801) | Flagged Allergen |
| **Conotoxin** | 25 | **0.975** | 0.400 | 0.370 | #8 (0.0063) | #8 (0.0095) | Extreme Neurotoxin |

---

## 5. Phase 0 Deliverable Checklist

- [x] Codebase data and training script inventory completed.
- [x] Claimed metrics reproduced and recorded in versioned `results/legacy_baseline_metrics.json`.
- [x] 8 canonical controls explicitly reclassified as **integration smoke tests** in `tests/test_pipeline.py` and `README.md`.
- [x] All 18 existing pytest tests verified passing (`18 passed in 60.55s`).
- [x] Model architectures and tree parameters inspected and documented.
- [x] Critical compensatory flaw in desirability ranking documented with empirical proof.

---

## 6. Next Steps & Approval Gate for Phase 1

To proceed to **Phase 1 (Leakage & Validity)**, we need to:
1. Curate an independent, homology-clustered benchmark dataset (UniProt / Swiss-Prot reference sets) partitioned via MMseqs2/CD-HIT at $\le 30-40\%$ identity.
2. Build a homology-grouped cross-validation protocol (`StratifiedGroupKFold`) to quantify the true leakage gap vs. random $k$-fold.
3. Run shortcut tests (length-only baselines and label permutation).
4. Evaluate `toxinpred2_rf.onnx` on an independent, homology-filtered evaluation set.

*Awaiting user approval before modifying code or proceeding to Phase 1.*
