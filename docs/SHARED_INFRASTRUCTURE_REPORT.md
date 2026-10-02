# Shared Infrastructure, Leakage Audit & Frozen Baselines Report

## 1. Dataset Inventory & Deduplication (Task 1)
- **Raw Sequences:** 525
- **Clean Deduplicated Sequences:** 512
- **Removed Duplicates/Near-Duplicates (>=98% ID):** 13
- **Clean Dataset SHA-256:** `165c3f26882b8bf1fd6cce6bcb09dd401b5606ca53bd970014ccee17ffc5c8bb`

### Class Balance Summary
| Endpoint | Clean Positives | Clean Negatives | Prevalence |
|---|---|---|---|
| Antigenicity | 93 | 419 | 18.2% |
| Toxicity | 142 | 370 | 27.7% |
| Allergenicity | 100 | 412 | 19.5% |

### Length Distribution Breakdown
- Length span: 7 - 3321 aa (median: 138.0 aa, mean: 212.3 aa)

| Length Bin | Sample Count | Fraction of Dataset |
|---|---|---|
| [5-9] | 10 | 1.9% |
| [10-14] | 38 | 7.4% |
| [15-24] | 16 | 3.1% |
| [25-49] | 50 | 9.8% |
| [50+] | 398 | 77.7% |

## 2. Homology Clustering (Task 2)
- **Identity Threshold:** 40.0%
- **Coverage Threshold:** 80.0%
- **Short Peptide (<20 aa) Fallback:** k-mer Jaccard (k=3, threshold=0.25) & normalized edit distance.
- **Total Homology Clusters:** 385
- **Singletons:** 323
- **Cluster Assignments SHA-256:** `6f8f98cb03ba3d12d1ee7968c177ed164330b5772dbbb5be7e5719e260651cb9`

## 3. Nested Group-Held-Out Splits (Task 3)
- **Outer Evaluation:** 5-fold `StratifiedGroupKFold` on cluster IDs.
- **Inner Partitions:** Outer training clusters are split into disjoint Inner-Train and Inner-Calibration partitions (25% calibration clusters).
- **Splits SHA-256:** `966773ac26e842b787ca89ae59b62e30a754d0b4451245ccfb0769dc4bdc299a`

## 4. Strict Leakage Audit (Task 4)
- **Audit Status:** **`PASS`**
- **Pairwise Cross-Partition Identity Comparisons:** 209,714
- **Maximum Cross-Partition Identity Observed:** 40.00%
- **Identity Threshold Enforced:** 40.00%
- **Verdict:** ZERO_LEAKAGE_VERIFIED: All outer test, inner calibration, and inner training sets are strictly disjoint.

## 5. Frozen Reference Baseline Metrics (Task 5 & 6)

| Model | Feature Set | ROC-AUC (95% CI) | PR-AUC (95% CI) | Balanced Acc | MCC | Brier Score | Adaptive ECE (15 bins) |
|---|---|---|---|---|---|---|---|
| Antigenicity (RF) | 125-D ACC | 0.7297 [0.6343 - 0.8140] | 0.4181 [0.2394 - 0.6090] | 0.5914 | 0.2658 | 0.1481 | 0.1292 |
| Antigenicity (GB) | 125-D ACC | 0.6348 [0.5266 - 0.7316] | 0.3151 [0.1726 - 0.5344] | 0.5472 | 0.1835 | 0.1516 | 0.1011 |
| Allergenicity (RF) | 50-D AAC+PCP | 0.8588 [0.8152 - 0.8979] | 0.5555 [0.4232 - 0.6968] | 0.7488 | 0.4446 | 0.1362 | 0.1370 |
| Toxicity (RF) | 50-D AAC+PCP | 0.8864 [0.8337 - 0.9238] | 0.7965 [0.6597 - 0.8840] | 0.7701 | 0.5693 | 0.1201 | 0.0826 |

### Length-Bin Breakdown (Antigenicity RF Baseline)
| Length Bin | Sample Count | Positives | AUROC | Balanced Acc | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|---|---|---|
| [5-9] | 10 | 0 | N/A | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| [10-14] | 38 | 0 | N/A | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| [15-24] | 16 | 0 | N/A | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| [25-49] | 50 | 1 | 0.0612 | 0.4694 | 0.0000 | 0.9388 | 0.0000 | 0.9787 |
| [50+] | 398 | 92 | 0.7095 | 0.5896 | 0.2283 | 0.9510 | 0.5833 | 0.8039 |
