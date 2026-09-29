# MODEL CARD: Allergenicity Predictor (AllerTOP/AlgPred Alternative)

**Model Version:** 2.1.0 (Random Forest ACC/PCP)  
**Model SHA-256:** `b1cbddfe919b01483cc37483c115243efc3ddefe84a6f9e565eab86f14d498fc`  
**Pipeline Integration:** [`src/allergenicity.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/allergenicity.py)  
**Calibration Status:** Platt Calibrated (`swiss_prot_eval_v1_437_clusters`)

---

## 1. INTENDED USE
- **Primary Objective:** In silico safety screening to identify potential IgE-mediated hypersensitivity and allergic risk in engineered peptides and therapeutic candidates.
- **Screening Context:**
  - *Vaccine Target Discovery:* Soft safety gate to identify and flag hypersensitivity risks in immunogens.
  - *Peptide Therapeutic Discovery:* Safety gate to avoid systemic anaphylaxis and allergic reactions.

## 2. TRAINING DATA & ARCHITECTURE
- **Architecture:** Shallow Random Forest (8 estimators, max depth 3) stored in joblib format (117 KB).
- **Features:** 50-D Descriptor Vector:
  - 20 Amino Acid Composition (AAC) fractions.
  - 30 Physicochemical Properties (PCP) derived via Pfeature (surface accessibility, polarity, charge).
- **Training Corpus:** Allergen benchmark dataset (~800 known environmental, food, and insect allergens vs non-allergenic controls).

## 3. EMPIRICAL PERFORMANCE (HONEST EVALUATION)
- **Homology-Grouped Validation ($\le 35\%$ identity):**
  - **AUROC:** **0.8687** [95% CI: 0.8332 – 0.9035] (on matched allergen benchmark).
- **Independent Swiss-Prot Clustered Evaluation (General Sequences):**
  - **AUROC on Swiss-Prot:** **0.5285**
  - **PR-AUC on Swiss-Prot:** **0.2620**
  - *Finding:* On diverse protein databases, the shallow surrogate tree produces noisy predictions with scores clustered between 0.30 and 0.70 across both benign proteins and allergens.

## 4. KNOWN FAILURE MODES & BIOPHYSICAL LIMITATIONS
1. **The "Screening Bottleneck" False Exclusion Hazard:**
   - Because the model predicts scores of 0.25 – 0.40 on benign proteins, applying uncalibrated thresholding with high false-negative penalties ($C_{FN} = \$50\text{k}$) results in a 100% false rejection rate across all candidates.
2. **False Positives on Surface-Exposed Helices:**
   - Richness in lysine, arginine, and polar surface residues (common in all globular proteins and approved drugs) triggers false allergenicity alarms.
3. **Inability to Discriminate Non-Allergenic Antigens:**
   - Both Lyme disease antigen OspA (0.730) and SARS-CoV-2 Spike RBD (0.690) receive high allergenicity scores, requiring calibration and human review.

## 5. APPLICABILITY DOMAIN
- **Valid Domain:** Plant pollen proteins, food storage proteins, and insect venom proteins between 50 and 500 amino acids.

## 6. "NOT INTENDED FOR" STATEMENTS
- **DO NOT USE** as clinical evidence for lack of allergenicity or FDA/EMA safety clearance.
- **DO NOT USE** for diagnosing IgE antibody cross-reactivity without experimental basophil activation or ELISA testing.
- **DO NOT USE** on short synthetic peptide fragments ($< 20$ aa).
