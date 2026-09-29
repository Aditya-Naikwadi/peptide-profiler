# Data Card: Swiss-Prot Multi-Task Evaluation Dataset v1.0

**Curated File:** `data/evaluation_dataset.json`  
**Curated FASTA:** `data/evaluation_dataset.fa`  
**File SHA-256:** `badda6ac98a85e3b20b7a51533cbd2e53df2bb2c6fe129c510dcf4d3c4a9d057`  
**Homology Cluster Artifact:** `data/cluster_assignments.json` (`cluster_assignments_v1_35pct`, **437 clusters**)  
**License:** **Creative Commons Attribution (CC BY 4.0)**  
**Curation Date:** 2026-09-29  

---

## 1. Overview & Dataset Identity
The Swiss-Prot Multi-Task Evaluation Dataset is an independent, multi-class protein and peptide benchmark curated directly from reviewed Swiss-Prot records. It provides simultaneous ground truth annotations for **toxicity**, **antigenicity**, and **allergenicity**, alongside temporal metadata (`first_public_date`) to support homology-aware, prevalence-aware, and temporal evaluation.

## 2. Sources & Governance
* **Primary Source:** UniProtKB / Swiss-Prot (Reviewed) via UniProt REST API.
* **Source License:** Creative Commons Attribution (CC BY 4.0). Redistribution permitted with attribution.
* **Label Evidence Type:** Curated literature annotations and experimentally supported Swiss-Prot keywords.
* **Governance Status:** **VERIFIED** (complies with `DATA_GOVERNANCE.md` Section 1).

## 3. Dataset Size & Class Balance
* **Total Sequences:** **525**
* **Toxicity:**
  * Positive Toxins: **146 (27.8%)** (Keywords: `KW-0800` Toxin/Venom, Pore-forming)
  * Non-Toxins: **379 (72.2%)**
* **Allergenicity:**
  * Positive Allergens: **100 (19.0%)** (Keywords: `KW-0020` Allergen, clinical IgE binding)
  * Non-Allergens: **425 (81.0%)**
* **Antigenicity:**
  * Positive Antigens: **100 (19.0%)** (Keywords: `KW-0044` Surface antigen, protective antigens)
  * Non-Antigens: **425 (81.0%)**
* **Pure Triple Negatives:** **179 (34.1%)** (Explicit negative controls: ribosomal and metabolic proteins devoid of toxic, allergenic, or surface annotations).

## 4. Length Distribution & Histogram
* **Range:** Min **7 aa**, Median **137 aa**, Mean **209.5 aa**, Max **3321 aa**.
* **Stratified Length Bins:**
  * `< 15 aa`: **50 sequences (9.5%)** (24 toxins, 25 negatives)
  * `15–50 aa`: **72 sequences (13.7%)** (22 toxins, 34 negatives)
  * `50–200 aa`: **233 sequences (44.4%)**
  * `200–500 aa`: **125 sequences (23.8%)**
  * `> 500 aa`: **45 sequences (8.6%)**

## 5. Temporal Distribution
* **Coverage:** 100% of sequences (525/525) contain verified Swiss-Prot deposit dates (`first_public_date`).
* **Date Range:** `1986-07-21` to `2026-06-10` (40-year span).
* **Temporal Cutoff for Temporal Split:** Cutoff date `2020-01-01` separates historical training from modern prospective holdouts.

## 6. Organism & Taxonomic Mix
* **Total Taxa Represented:** **193 unique species** across bacteria, viruses, fungi, plants, arthropods, reptiles, and mammals.
* **Organism Classes:**
  * Eukaryota: 341 sequences (human, snake venom, bee venom, birch pollen, peanut, etc.)
  * Bacteria: 122 sequences (Borrelia, Mycobacterium, Staphylococcus, etc.)
  * Viruses: 62 sequences (Coronaviridae, Influenza, HIV, Flavivirus, etc.)

## 7. Known Biases & Label Noise
1. **Length Matching:** Includes explicit size-matched negative cohorts (<15 aa and 15–50 aa) to mitigate length shortcuts.
2. **Residual Label Noise:** Estimated at `< 1%`. All records derive from Swiss-Prot reviewed curator inspection.
3. **Database Representation:** Natural toxins in Swiss-Prot are over-represented by venomous reptiles and arthropods; natural allergens are dominated by pollen and food allergens.

## 8. Intended Use & Limitations
* **Intended Use:** Homology-grouped benchmarking, temporal generalization testing, length-matched negative evaluations, and prior-shift calibration of peptide/protein classifiers.
* **Not Intended For:** Direct clinical diagnosis, regulatory bioequivalence claims, or training without homology partitioning.
