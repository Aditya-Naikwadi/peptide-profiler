# MODEL CARD: Toxicity Predictor (ToxinPred2 ONNX Random Forest)

**Model Version:** 2.1.0 (ONNX RF Architecture)  
**Model SHA-256:** `5683cd21b20c627c09dc85c2d3ad6d33533003fc01f4acae5b444368bb3e2bde`  
**Pipeline Integration:** [`src/toxicity.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/toxicity.py)  
**Calibration Status:** Platt Calibrated (`swiss_prot_eval_v1_437_clusters`)

---

## 1. INTENDED USE
- **Primary Objective:** High-throughput in silico safety screening of natural, recombinant, and designed peptide sequences to flag hemolytic, cytotoxic, and neurotoxic hazards prior to synthesis.
- **Screening Context:**
  - *Vaccine Target Discovery:* Hard safety gate to disqualify toxic bacterial/viral antigens.
  - *Peptide Therapeutic Discovery:* Safety disqualification of toxic drug leads.

## 2. TRAINING DATA & ARCHITECTURE
- **Architecture:** Random Forest Classifier (100 estimators) exported to ONNX format (119.5 MB graph).
- **Features:** 20 Amino Acid Composition (AAC) fractions.
- **Source Training Corpus:** ToxinPred2 benchmark dataset (~5,000 toxic peptides from venom databases vs non-toxic Swiss-Prot controls).

## 3. EMPIRICAL PERFORMANCE (HONEST EVALUATION)
- **Independent Swiss-Prot Clustered Evaluation (525 sequences, $\le 35\%$ homology identity):**
  - **AUROC:** **0.9450** [95% CI: 0.9180 – 0.9675]
  - **PR-AUC:** **0.8927**
  - **Sensitivity (TPR at $\tau^* = 0.35$):** 94.29%
  - **Specificity (at $\tau^* = 0.35$):** 97.30%
  - **Expected Calibration Error (ECE):** Reduced from 27.39% (uncalibrated) to **3.56%** post-Platt scaling.

## 4. KNOWN FAILURE MODES & BIOPHYSICAL LIMITATIONS
1. **Severe False Positive Bias on Synthetic Peptide Drugs:**
   - Evaluated on FDA-approved peptide drugs:
     - **Bivalirudin (20 aa):** Predicted Toxicity = 0.685 (Falsely flagged as toxic).
     - **Octreotide (6 aa):** Predicted Toxicity = 0.877 (Falsely flagged as toxic).
   - *Cause:* Model relies purely on 20 AAC frequencies. Synthetic drugs enriched in specific charged or aromatic residues mimic venom composition without sharing tertiary pore-forming activity.
2. **Short Peptide (<15 aa) Degradation:**
   - Specificity drops to 44.0% on peptides under 15 aa. Compositional frequencies become erratic in very short peptides.
3. **Severe Prevalence Effect (Base-Rate Fallacy):**
   - At realistic library prevalence ($\pi = 1\%$), Positive Predictive Value (PPV) is **3.95%**. 96 out of 100 positive alarms are false positives.

## 5. APPLICABILITY DOMAIN
- **Valid Sequence Space:** Natural L-amino acid linear peptides between 15 and 500 amino acids.
- **Organism Coverage:** Bacterial, viral, fungal, plant, and animal sequences.

## 6. "NOT INTENDED FOR" STATEMENTS
- **DO NOT USE** as a sole diagnostic or regulatory safety determination.
- **DO NOT USE** on cyclic, D-amino acid, or non-canonical/chemically modified peptides (e.g., lipidated, PEGylated, stapled peptides).
- **DO NOT USE** on sequences shorter than 15 amino acids without manual biochemical review.
