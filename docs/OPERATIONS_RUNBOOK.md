# PRODUCTION OPERATIONS RUNBOOK: PEPTIDE PROFILER

**System:** Local Offline Peptide Characterization Pipeline  
**Version:** 2.1.0  
**Target Audience:** MLOps Engineers, Computational Biologists, Bioinformaticians  
**Artifact File:** `docs/OPERATIONS_RUNBOOK.md`  
**Configuration File:** [`config/monitoring_config.json`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/config/monitoring_config.json)

---

## 1. ARCHITECTURE & LOCAL AUDIT LOGGING

The Peptide Profiler runs completely offline without external network dependencies. Every inference request is recorded to a local append-only audit log:
- **Audit Log Path:** `logs/inference_audit.jsonl`
- **Logged Attributes:**
  - `timestamp_utc`: ISO 8601 UTC timestamp.
  - `id`, `length`: Sequence identifier and amino acid length.
  - `sanitizer_edit_rate`: Fraction of characters sanitized (e.g., lowercase conversion, whitespace stripping).
  - `candidate_status`: `APPROVED`, `FLAGGED`, or `EXCLUDED`.
  - `reason_codes`: e.g., `SAFETY_VIOLATION_TOXICITY`, `SAFETY_VIOLATION_ALLERGENICITY`, `IMMUNOGENICITY_RISK`.
  - `abstentions`: e.g., `LOW_CONFIDENCE_SHORT_PEPTIDE`.
  - `scores`: Raw and Platt-calibrated probabilities for toxicity, allergenicity, and antigenicity.
  - `desirability` & `conservative_rank_score`: Ranking scores with uncertainty penalties.
  - `metadata`: Model SHA-256 hashes, feature extractor version (`pfeature_pcp_aac_v1.0`), and calibration set ID.

---

## 2. DRIFT MONITORING METRICS & TRIPWIRES

Drift diagnostics are executed via `src/monitoring.py` comparing incoming screening batches against the baseline Swiss-Prot calibration set (`data/evaluation_dataset.json`, 525 sequences across 437 homology clusters).

### 2.1 Monitored Metrics
1. **Population Stability Index (PSI):**
   - Evaluated on sequence length, toxicity score, and GRAVY index.
   - Formula: $\text{PSI} = \sum_{i=1}^k (A_i - E_i) \cdot \ln(A_i / E_i)$
2. **Kolmogorov-Smirnov (KS) Test:**
   - Two-sample non-parametric test comparing continuous distributions of predicted probabilities.
3. **Rolling Adversarial Validation (AUROC):**
   - A logistic classifier is trained with 3-fold cross-validation to distinguish incoming production sequences from baseline calibration sequences.
   - $\text{AUROC} \approx 0.50$: Sequences share identical feature distributions (Nominal).
   - $\text{AUROC} \ge 0.70$: Substantial covariate shift detected.
   - $\text{AUROC} \ge 0.80$: Severe distribution drift (Critical).
4. **Length-Bin Mix Shift:**
   - Tracks proportion of sequences in $<15$, $15-50$, $50-200$, $200-500$, and $>500$ aa bins.
5. **Abstention Rate:**
   - Fraction of queries triggering `LOW_CONFIDENCE_SHORT_PEPTIDE` ($<15$ aa).

### 2.2 Alert Levels & Action Matrix

| Alert Level | Trigger Conditions | Operational Action |
| :---: | :--- | :--- |
| **GREEN** *(Nominal)* | PSI $< 0.10$<br>KS $p \ge 0.05$<br>Adversarial AUROC $< 0.70$<br>Abstention Rate $< 8\%$ | Automated pipeline execution continues normally. No manual intervention required. |
| **YELLOW** *(Investigate)* | PSI $\ge 0.10$<br>KS $p < 0.05$<br>Adversarial AUROC $\ge 0.70$<br>Abstention Rate $\ge 10\%$ | Alert logged to `logs/monitoring_alerts.log`. Weekly review by bioinformatician. Inspect incoming sequence taxonomy and length distribution. |
| **RED** *(Critical)* | PSI $\ge 0.25$<br>KS $p < 0.01$<br>Adversarial AUROC $\ge 0.80$<br>Abstention Rate $\ge 15\%$ | **Screening pipeline pauses automatic exports.** Requires sign-off from Lead Data Scientist. Evaluate need for model recalibration or domain retraining. |

---

## 3. SHADOW MODE & CANARY AUTO-ROLLBACK

When upgrading models (e.g. replacing surrogate random forests with deep learning or retrained models), candidate models MUST undergo shadow mode execution via `src/shadow.py`.

### 3.1 Shadow Mode Execution Protocol
1. Production and Candidate models run concurrently on all incoming sequences.
2. Production model provides the customer-facing output and ranking.
3. The Candidate model's outputs are recorded to `logs/shadow_evaluations.jsonl`.
4. The `ShadowComparisonEngine` computes:
   - **Score Delta:** $|s_{\text{prod}} - s_{\text{cand}}|$ (Mean and Max).
   - **Status Disagreement Rate:** Fraction of sequences where candidate status diverges (e.g., `APPROVED` vs `EXCLUDED`).
   - **Top-$k$ Jaccard Overlap:** Overlap of the top-$k$ ranked survivors.
   - **Spearman Rank Correlation:** Rank stability between production and candidate.

### 3.2 Automatic Rollback Tripwire
- If the **Status Disagreement Rate** exceeds **10.0%** (`max_disagreement_tripwire`), or if unhandled exceptions occur:
  - Canary traffic is instantly reset to **0.0%**.
  - System status transitions to `ROLLED_BACK`.
  - An emergency alert is logged to `logs/shadow_evaluations.jsonl`.
  - The production model maintains 100% operational authority.

---

## 4. WET-LAB FEEDBACK & RETRAINING PROTOCOL

### 4.1 Assay Feedback Recording
Experimental outcomes from wet-lab assays (cytotoxicity MTT, hemolytic assays, ELISA binding, IgE tests) must be logged using the schema in [`data/wet_lab_feedback_schema.json`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/data/wet_lab_feedback_schema.json):
```bash
python scripts/log_wet_lab_feedback.py
```

### 4.2 Mitigating Selection Bias
When choosing candidates for wet-lab synthesis, **never test only top-ranked leads**. That creates an irrecoverable survivorship bias where false negatives are never observed.
- Use `sample_candidates_with_bias_mitigation()`:
  - Select Top $k$ Leads (80% of budget).
  - Select Stratified Lower-Ranked / Excluded Controls (20% of budget).

### 4.3 Retraining Triggers
A formal model retraining cycle is triggered if ANY of the following occur:
1. **Sustained Drift:** 3 consecutive screening batches trigger RED alert on PSI ($> 0.25$) or Adversarial AUROC ($> 0.80$).
2. **Back-Test Degradation:** Periodic back-testing on newly published Swiss-Prot entries shows an AUROC or Precision@$k$ drop exceeding **10.0%**.
3. **Wet-Lab Assay Accumulation:** At least **50 wet-lab assay outcomes** have been recorded with ground-truth experimental labels.

### 4.4 Retraining Procedure
1. Cluster all newly accumulated sequences with existing training corpora at $\le 35\%$ homology identity using `src/homology.py`.
2. Execute 5-fold Stratified Group K-Fold cross-validation.
3. Compute honest AUROC, PR-AUC, and Matthews Correlation Coefficient.
4. Fit Platt scalers on group-held-out validation folds.
5. Re-optimize cost thresholds per length bin using `scripts/optimize_thresholds.py`.
6. Deploy into Shadow Mode for at least 500 sequences before canary release.
