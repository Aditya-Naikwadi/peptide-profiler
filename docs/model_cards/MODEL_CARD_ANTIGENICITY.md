# MODEL CARD: Antigenicity Predictor (VaxiJen-Alternative ML Classifier)

**Model Version:** 2.1.0 (Random Forest ACC/PCP)  
**Model SHA-256:** `c1ee5bf6aac96b3236d2e91493de12a7dfcdd834985e9ff18cdc4ff780890d60`  
**Pipeline Integration:** [`src/antigenicity.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/antigenicity.py)  
**Calibration Status:** Platt Calibrated (`swiss_prot_eval_v1_437_clusters`)

---

## 1. INTENDED USE
- **Primary Objective:** In silico ranking of immunogenic potential.
- **Screening Context:**
  - *Vaccine Target Discovery:* Prioritizes protective immunogens capable of eliciting antibody responses.
  - *Peptide Therapeutic Discovery:* **Inverted objective** ($1 - P_{\text{cal}}(\text{antigen})$) to minimize anti-drug antibodies (ADAs).

## 2. TRAINING DATA & ARCHITECTURE
- **Architecture:** Shallow Random Forest (8 estimators, max depth 4) stored in joblib format (111 KB).
- **Features:** 50-D Descriptor Vector:
  - 20 Amino Acid Composition (AAC) fractions.
  - 30 Physicochemical Properties (PCP) derived via Pfeature (charge, polarity, surface accessibility, aromaticity, hydrophobic moments).
- **Training Corpus:** Benchmark bacterial antigen dataset (~1,000 bacterial proteins from legacy VaxiJen benchmark).

## 3. EMPIRICAL PERFORMANCE (HONEST EVALUATION)
- **Leakage Gap Quantified:**
  - Random Cross-Validation AUROC: 0.8683 (Inflated benchmark number)
  - **Homology-Grouped CV AUROC ($\le 35\%$ identity):** **0.7861 [95% CI: 0.6961 – 0.8500]** (Leakage gap: **-0.0822**)
  - Matthews Correlation Coefficient (MCC): Dropped by **-0.1238** under grouped validation.
- **Independent Swiss-Prot Evaluation (General Sequences):**
  - **AUROC on general Swiss-Prot:** **0.5132** (Near random guess across diverse taxonomic clades).
  - Model has moderate discrimination on bacterial outer membrane proteins, but collapses when evaluated on viral, eukaryotic, or host proteins.

## 4. KNOWN FAILURE MODES & BIOPHYSICAL LIMITATIONS
1. **Short Peptide Collapse (< 50 aa):**
   - The model has zero discriminative power on peptides shorter than 50 amino acids. Linear epitopes require secondary and tertiary structural context that 50-D AAC+PCP cannot capture.
2. **Taxonomic Overfitting:**
   - Overfitted to bacterial envelope lipidation and transmembrane beta-barrel signatures. Scores eukaryotic housekeeping proteins (e.g. Ubiquitin) unpredictably.
3. **Platt Compression:**
   - Calibrated probabilities compress into the 0.18 – 0.22 range due to high baseline uncertainty.

## 5. APPLICABILITY DOMAIN
- **Valid Domain:** Full-length bacterial outer membrane, secretory, or envelope proteins $> 50$ amino acids.

## 6. "NOT INTENDED FOR" STATEMENTS
- **DO NOT USE** for B-cell or T-cell epitope mapping without sliding-window Kolaskar-Tongaonkar analysis.
- **DO NOT USE** on peptide fragments shorter than 30 amino acids.
- **DO NOT USE** to guarantee in vivo protective efficacy or MHC class I/II binding affinity.
