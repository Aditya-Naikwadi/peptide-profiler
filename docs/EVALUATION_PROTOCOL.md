# EVALUATION_PROTOCOL: How We Measure

A result is only valid if it follows this protocol. Numbers from any other method are labeled "exploratory" and never used for decisions.

## 1. Splits
- Cluster all sequences by homology: ~30-40% identity for proteins (MMseqs2 or CD-HIT). For peptides under ~30 aa, group by shared k-mers or edit distance. Store cluster IDs as a versioned artifact.
- Primary CV: stratified group K-fold on cluster IDs.
- Additional tests: leave-one-family-out, source-held-out, temporal (train before cutoff date, test after).
- **Final test set** is created once, frozen, and used only in WP8.
- Random K-fold is run only to quantify the leakage gap.

## 2. Leakage and shortcut tests (must pass before any model is promoted)
| Test | Pass condition |
|---|---|
| Label permutation | AUROC ≈ 0.5 |
| Length-only baseline | Reported; if close to the full model, length is a shortcut → rebuild negatives |
| Cross-split duplicates/near-duplicates | None |
| Source-held-out | Reported; large drop means source bias |
| Adversarial validation (train vs real incoming sequences) | Reported; AUROC well above 0.5 means distribution shift |
| Fold hygiene | Scaling, feature selection and calibration fit inside training folds only |

## 3. Metrics
- Discrimination: AUROC, PR-AUC, MCC, recall at fixed precision.
- Screening utility: precision@k, enrichment factor (top 1%, 5%, 10%).
- Calibration: Brier score, ECE, reliability curves (on group-held-out data).
- **Prevalence-aware:** PPV at 1%, 2%, 3% deployment prevalence (and the owner's `OWNER-INPUT` value). Formula: PPV = sens·prev / (sens·prev + (1−spec)(1−prev)).
- Stratify by length bin: <15, 15-50, 50-200, 200-500, >500 aa.
- Every metric reported with **cluster-bootstrap confidence intervals** (resample clusters, not sequences).

## 4. Model selection
Nested grouped CV. Prefer the simpler model when the gain is within fold-to-fold standard deviation. Feature pruning by grouped permutation importance. Thresholds and calibration chosen on calibration data only.

## 5. Threshold and decision rules
Cost-optimal thresholds per organism and profile using the config cost matrix (`config/costs.yaml`, values `OWNER-INPUT`) at deployment prevalence. Frozen in versioned config with calibration-set ID.

## 6. Benchmark against published standards
Facts to respect (cite as published, do not restate as our results):
- VaxiJen v3.0 (bacterial): 317 immunogens + 317 non-immunogens (250/250 train, 67/67 test); ACC features; XGBoost, RSM-1NN and RF with feature selection combined by majority vote; evaluated on the proteomes of 11 bacterial species with known protective antigens; fold-enrichment ~1.2 (v2.0) vs ~4.5 (v3.0). Source: Dimitrov et al., Vaccines 2020, 8:709.
- Original VaxiJen v2.0 organism-specific thresholds (0.5 bacterial, 0.4 viral, 0.5 tumor) are reference only.

Protocol:
1. Reproduce the bacterial benchmark if the data is obtainable; metrics: observed protective antigens, sensitivity, fold-enrichment, fraction of potential vaccine candidates. If not obtainable, list the exact request for the owner.
2. Build a comparison FASTA (a few hundred held-out sequences not in training). The owner submits it manually to VaxiJen and returns a CSV.
3. On the returned CSV compute: AUROC on shared labels, Cohen's kappa between predictors, disagreement analysis (length, organism, homology-to-training).
4. VaxiJen is a comparator, not ground truth. Goal: equal or beat it on the same held-out sequences.

## 7. Robustness suite
Point mutations (1-10%), truncations, N/C-terminal tags, injected ambiguous residues (X/B/Z), formatting noise. Report score stability. Sliding-window validation: embed known toxic segments in benign scaffolds of varied length and check recall.

## 8. Promotion gate (new model replaces current only if ALL hold)
1. Beats current model on grouped and temporal test sets, with CIs that do not overlap the current model's point estimate, or the owner explicitly accepts a tie for simplicity/scalability reasons.
2. Passes leakage, permutation, ONNX parity and robustness tests.
3. Calibration within tolerance (`OWNER-INPUT`).
4. Threshold and calibration artifacts frozen and versioned.
5. Model card and data card updated.
Otherwise keep the legacy model and document why.

## 9. Smoke tests (not validation)
OspA, melittin, conotoxin, Bet v 1 and ubiquitin are expected to be in the training data of upstream tools. They confirm the pipeline runs and are never cited as evidence of generalization.

## 10. Report template
Model · data version · split · metric (value, 95% CI) · length-bin breakdown · PPV at prevalence · calibration · known failures · decision.
