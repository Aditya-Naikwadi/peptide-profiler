# ANTIGENICITY_MODEL_REPORT_PHASE_5: Model Selection & VaxiJen v3.0 Replication

## 1. Executive Summary
- **Dataset Evaluated:** Curated evaluation set of 525 sequences (100 antigens, 425 non-antigens) across 437 homology clusters.
- **Evaluation Methodology:** Homology-aware 5-fold `StratifiedGroupKFold` cross-validation with cluster-level bootstrap (1,000 resamples) 95% confidence intervals.
- **Zero Leakage:** All feature scalers, selectors, and estimators fitted strictly inside CV training folds.

## 2. Model Comparison: 50-D AAC+PCP vs. 125-D ACC (VaxiJen v3.0)

| Architecture | Feature Space | AUROC (95% CI) | AUPRC (95% CI) | Balanced Acc | MCC | Brier Score |
|---|---|---|---|---|---|---|
| LogisticRegression | 50-D AAC+PCP | 0.6938 [0.6088 - 0.7669] | 0.3044 [0.2182 - 0.4013] | 0.5391 | 0.0995 | 0.1643 |
| RandomForest | 50-D AAC+PCP | 0.7849 [0.6796 - 0.8735] | 0.6106 [0.4980 - 0.7131] | 0.5915 | 0.3585 | 0.1200 |
| XGBoost | 50-D AAC+PCP | 0.7602 [0.6628 - 0.8489] | 0.6001 [0.4822 - 0.7085] | 0.6859 | 0.4887 | 0.1141 |
| kNN | 50-D AAC+PCP | 0.7624 [0.6791 - 0.8330] | 0.4393 [0.3394 - 0.5375] | 0.6594 | 0.3827 | 0.1325 |
|---|---|---|---|---|---|---|
| LogisticRegression | 125-D ACC | 0.7581 [0.6963 - 0.8160] | 0.3688 [0.2789 - 0.4741] | 0.6541 | 0.3082 | 0.1727 |
| RandomForest | 125-D ACC | 0.7785 [0.7223 - 0.8300] | 0.4687 [0.3453 - 0.5947] | 0.5200 | 0.1806 | 0.1339 |
| XGBoost | 125-D ACC | 0.8221 [0.7681 - 0.8721] | 0.6051 [0.4920 - 0.7063] | 0.6103 | 0.3922 | 0.1166 |
| kNN | 125-D ACC | 0.7682 [0.7093 - 0.8250] | 0.4802 [0.3710 - 0.5906] | 0.6285 | 0.3681 | 0.1248 |
| **VaxiJen v3.0 Ensemble** | **125-D ACC** | **0.8586 [0.8042 - 0.9033]** | **0.6552 [0.5346 - 0.7622]** | **0.5776** | **0.3351** | **0.1126** |
| **Pruned Ensemble (Top 50)** | **50-D ACC** | **0.8775 [0.8243 - 0.9201]** | **0.6823 [0.5688 - 0.7746]** | **0.5926** | **0.3713** | **0.1079** |

## 3. VaxiJen v3.0 Replication Architecture & Implementation Differences

### Published VaxiJen v3.0 Architecture (Dimitrov et al., Vaccines 2020, 8:709):
1. **Feature Space:** 125-D Auto Cross-Covariance (ACC) descriptors computed from 5 Z-scales (Sandberg et al., 1998) at lag $L=5$.
2. **Ensemble Base Classifiers:**
   - Gradient Boosted Trees (XGBoost)
   - Random Subspace Method with 1-Nearest Neighbor (RSM-1NN)
   - Random Forest with feature selection (`SelectFromModel`)
3. **Decision Rule:** Binary majority voting across the 3 base estimators.

### Differences in Our Local Replication:
- **Zero Network Latency & True Offline Execution:** Runs purely locally with pinned seeds (`random_state=42`), eliminating VaxiJen web server outages and query limits.
- **Strict Homology Splitting:** Validated using `StratifiedGroupKFold` clustering (MMseqs2 30-40% identity equivalents), eliminating homologous sequence leakage between training and validation folds.
- **Subspace Standardization:** RSM-1NN base classifiers encapsulate `StandardScaler` inside training folds to prevent feature dominance across physicochemical scales.

## 4. Feature Pruning & Permutation Importance
- Analyzed grouped permutation importance across all 125 ACC descriptors over 5 CV folds.
- Top features are concentrated in auto-covariances of lipophilicity ($z_1$) and polarity ($z_3$) at short lags ($L=1, 2$).
- Pruned model using top 50 ACC descriptors achieved AUROC of 0.8775 [0.8243 - 0.9201] with MCC of 0.3713, demonstrating that 60% of descriptors can be eliminated while preserving discriminative performance within fold-to-fold error margin.

## 5. Length-Bin Breakdown (VaxiJen Replication)

| Length Bin | Sample Count | Antigens | AUROC | Balanced Acc | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| <15 aa | 50 | 0 | N/A | N/A | N/A | N/A |
| 15-50 aa | 72 | 1 | 0.1127 | 0.4930 | 0.0000 | 0.9859 |
| 50-200 aa | 233 | 75 | 0.8642 | 0.5702 | 0.1467 | 0.9937 |
| 200-500 aa | 125 | 20 | 0.8676 | 0.5750 | 0.1500 | 1.0000 |
| >500 aa | 45 | 4 | 0.9878 | 0.7500 | 0.5000 | 1.0000 |

## 6. Organism-Specific Routing & Guardrail Abstentions
- Implemented `OrganismSpecificAntigenicityManager` supporting `bacteria`, `virus`, and `tumor`.
- When queried with an unsupported organism (e.g. fungal, parasite, plant, or synthetic), the system explicitly returns `status='abstain', reason='unsupported_organism'` rather than outputting misleading extrapolation scores.
- Serialized artifacts stored at `models/antigen_vaxijen_acc_ensemble.joblib` and `models/antigen_organism_manager.joblib`.

## 7. Recommendation for Gate 5 Sign-Off
1. **Primary Production Model:** Adopt the **VaxiJen v3.0 Replication Ensemble on 125-D ACC features** as the core antigenicity engine.
2. **Safety Routing:** Route all queries through `OrganismSpecificAntigenicityManager` with explicit abstention guardrails for unsupported taxa.