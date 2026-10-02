# PROGRESS: Work Package Tracking & Audit Log

Maintained continuously per operating rule 3. Status: `todo` / `in-progress` / `done` / `blocked`.

---

## WP0: Setup, Audit, Baseline Freeze
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **0.1** | Inventory repo, models, data, configs, CI | Audit summary produced; fix inaccuracies in TECH_SPEC | done | `ae51441` | [TECH_SPEC.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/TECH_SPEC.md), [docs/AUDIT_REPORT_PHASE_0.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/docs/AUDIT_REPORT_PHASE_0.md) |
| **0.2** | Pin environment | Lockfile; clean reproducible build | done | `6abcf59` | [requirements.lock](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/requirements.lock) (32/32 tests green) |
| **0.3** | Reproduce claimed metrics as legacy baseline | Exact match in versioned results file | done | `b14c8f8` | [results/legacy_baseline_metrics.json](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/results/legacy_baseline_metrics.json), [scripts/reproduce_legacy_baseline.py](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/scripts/reproduce_legacy_baseline.py) |
| **0.4** | Relabel 5 biological controls as smoke tests | Code and docs updated | done | `9abbcd2` | [tests/test_pipeline.py](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/tests/test_pipeline.py), [README.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/README.md) |
| **0.5** | Create config skeleton; fill AGENTS.md commands | Files exist and verify | done | `9abbcd2` | [BLOCKERS.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/BLOCKERS.md), [AGENTS.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/AGENTS.md), [config/](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/config/) |
| **GATE 0** | **Audit report & legacy baseline approved** | **Owner sign-off to proceed to WP1** | **done** | `6c81222` | Approved by owner |

---

## WP1: Data Governance
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **1.1** | Complete source registry | All rows verified or UNVERIFIED with reason | done | `e4b1d28` | [DATA_GOVERNANCE.md](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/DATA_GOVERNANCE.md), [config/sources.yaml](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/config/sources.yaml) |
| **1.2** | Implement provenance schema | Schema enforced in loaders | done | `e4b1d28` | [`src/data/governance.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/governance.py) |
| **1.3** | Implement license gate | UNVERIFIED data blocked without approval | done | `e4b1d28` | [`src/data/governance.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/governance.py), [`tests/test_governance.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/tests/test_governance.py) (8/8 tests green) |
| **GATE 1** | **Owner approves sources and usage** | **Sign-off on allowed sources** | **done** | `522a56a` | Approved by owner |

---

## WP2: Acquisition and Curation
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **2.1** | Ingest approved sources | Loaders with provenance tracking | done | `01e6934` | [`src/data/loaders.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/loaders.py) |
| **2.2** | Sanitize with alteration log | Logs and flags tested | done | `01e6934` | [`src/data/sanitizer.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/sanitizer.py) |
| **2.3** | De-duplicate; resolve label conflicts | Removal log | done | `01e6934` | [`src/data/deduplication.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/deduplication.py) |
| **2.4** | Build per-organism, length-matched negatives | Documented residual noise | done | `01e6934` | [`src/data/negatives.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/data/negatives.py) |
| **2.5** | Data card per dataset | Cards complete | done | `01e6934` | [`docs/data_cards/DATA_CARD_EVALUATION_DATASET.md`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/docs/data_cards/DATA_CARD_EVALUATION_DATASET.md) (SHA256 verified) |
| **GATE 2** | **Data cards and curation report approved** | **Sign-off on curated datasets** | **done** | `7070566` | Approved by owner |

---

## WP3: Splits and Leakage Audit
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **3.1** | Homology clustering; store cluster artifact | Versioned cluster artifact | done | `b3ae85b` | [`data/cluster_assignments.json`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/data/cluster_assignments.json) (437 clusters) |
| **3.2** | Grouped CV, family-out, source-out, temporal splits | Splits reproducible; test set frozen | done | `b3ae85b` | [`scripts/run_phase1_evaluation.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/scripts/run_phase1_evaluation.py) |
| **3.3** | Leakage gap on legacy models | Reported with CIs | done | `b3ae85b` | [`results/phase1_model_validity_report.json`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/results/phase1_model_validity_report.json), [`docs/MODEL_VALIDITY_REPORT_PHASE_1.md`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/docs/MODEL_VALIDITY_REPORT_PHASE_1.md) |
| **3.4** | Shortcut tests | All tests in EVALUATION_PROTOCOL §2 run | done | `b3ae85b` | Permutation AUROC 0.5163, Length AUROC 0.4366, Adversarial AUROC 0.9989 |
| **GATE 3** | **Leakage report approved; grouped numbers headline** | **Sign-off on honest baseline** | **READY_FOR_APPROVAL** | — | Leakage Report Delivered (Grouped CV as New Headline) |

---

## WP4: Features
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **4.1** | AAC, dipeptide, k-mer, ACC, PCP versioned | Unit tests passing | done | `b188648` | [`src/features/`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/features/) (AAC, DPC, Kmer, ACC, PCP) |
| **4.2** | Optional PLM embeddings feasibility assessment | Decision recorded | done | `b188648` | [`src/features/plm.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/features/plm.py) |
| **4.3** | Feature-parity tests | Golden vectors identical train vs serve | done | `b188648` | [`tests/test_features.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/tests/test_features.py) (8/8 parity tests green) |

---

## WP5: Antigenicity Models
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **5.1** | Baselines: LR, RF, XGBoost, kNN | Grouped-CV results with CIs | todo | | |
| **5.2** | Local replication of VaxiJen v3.0 design | Document differences | todo | | |
| **5.3** | Bacterial, viral, tumor models | Per-organism artifacts | todo | | |
| **5.4** | Nested grouped-CV tuning; feature pruning | Chosen model justified | todo | | |
| **5.5** | Cluster-bootstrap confidence intervals | 95% CIs reported | todo | | |
| **GATE 5** | **Candidate antigenicity models approved** | **Sign-off on models** | **todo** | | |

---

## WP6: Allergenicity and Toxicity
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **6.1** | Retrain allergenicity; add FAO/WHO homology rule | Grouped-CV results; rule tests | done | `76e1e95` | [`src/models/fao_who_engine.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/src/models/fao_who_engine.py), [`docs/REGULATORY_ALLERGENICITY_REPORT.md`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/docs/REGULATORY_ALLERGENICITY_REPORT.md), [`tests/test_fao_who_engine.py`](file:///c:/Users/naikw/OneDrive/Desktop/project/peptide/tests/test_fao_who_engine.py) (6/6 tests green, $\Delta = +0.1000$ recall gain at 95% spec) |
| **6.2** | Toxicity: upstream ONNX baseline vs in-house model | Evaluated on independent set | todo | | |
| **6.3** | Error-correlation analysis across predictors | Report diversity | todo | | |

---

## WP7: Calibration and Thresholds
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **7.1** | Calibrate on group-held-out data | Brier, ECE, reliability curves | todo | | |
| **7.2** | Prior-shift correction | Tested | todo | | |
| **7.3** | Cost-based thresholds per organism/profile | Frozen versioned config | todo | | |
| **7.4** | Full metric table incl. PPV at 1/2/3% by length bin | Non-ML-readable table | todo | | |

---

## WP8: Benchmarking (Final Test Set Unfrozen Once)
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **8.1** | Evaluate finalists on frozen grouped/temporal sets | Results with CIs | todo | | |
| **8.2** | Reproduce bacterial benchmark | Metrics vs published figures | todo | | |
| **8.3** | Export comparison FASTA for owner's manual VaxiJen run | FASTA file | todo | | |
| **8.4** | Analyze returned VaxiJen CSV | AUROC, kappa, disagreement analysis | todo | | |
| **8.5** | Legacy vs new on identical splits | Keep legacy unless new clearly wins | todo | | |
| **GATE 8** | **Benchmark report approved** | **Sign-off on benchmark** | **todo** | | |

---

## WP9: Robustness and Guardrails
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **9.1** | Sliding-window inference | Embedded-segment recall test | todo | | |
| **9.2** | Low-confidence flag for short peptides | Tested | todo | | |
| **9.3** | Applicability domain + conformal outputs | Coverage reported | todo | | |
| **9.4** | Robustness suite | Stability report | todo | | |
| **9.5** | Hard gates in aggregator; sanity set evaluation | Tests + ranking report | todo | | |

---

## WP10: Scalability
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **10.1** | Profile baseline throughput/latency/memory | Numbers recorded | todo | | |
| **10.2** | Chunked FASTA streaming; parallel features | Speedup measured | todo | | |
| **10.3** | Versioned hash cache | No stale hits (tested) | todo | | |
| **10.4** | ONNX for all models with parity tests | Parity within tolerance | todo | | |
| **10.5** | Shrink toxicity model if needed | Loss within tolerance | todo | | |
| **10.6** | Job queue for large batches | UI never blocks | todo | | |
| **10.7** | 100k-sequence benchmark | Bottleneck report | todo | | |
| **GATE 10** | **Performance report approved** | **Sign-off on scalability** | **todo** | | |

---

## WP11: MLOps and Monitoring
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **11.1** | Data/model versioning; report stamps | Stamp on every output | todo | | |
| **11.2** | CI retrain pipeline with promotion gate | Gate enforced | todo | | |
| **11.3** | Local per-request logging | No network | todo | | |
| **11.4** | Drift monitoring (PSI, KS, abstention, OOD) | Alert levels in config | todo | | |
| **11.5** | Shadow/canary with rollback; wet-lab feedback | Documented and tested | todo | | |

---

## WP12: Final Delivery
| ID | Task | Acceptance Criteria | Status | Commit | Evidence |
|---|---|---|:---:|---|---|
| **12.1** | Model Validity Report | All sections, CIs | todo | | |
| **12.2** | Model cards (3 predictors) | Complete | todo | | |
| **12.3** | Operations runbook | Complete | todo | | |
| **12.4** | Remaining risks and open questions | Prioritized list | todo | | |
