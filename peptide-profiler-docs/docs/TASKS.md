# TASKS: Work Packages, Gates, Blockers, Open Decisions

The agent updates this file after every task. Status: `todo` / `in-progress` / `done` / `blocked`. Never mark `done` without evidence (commit hash and a test or measurement output).

Per-task protocol: restate goal → plan → small change → tests → run → commit → update this file. Stop at every GATE.

## Open Decisions (owner fills in)
| ID | Decision | Value | Needed by |
|---|---|---|---|
| D1 | Deployment prevalence: toxic / allergenic / antigenic | OWNER-INPUT | WP7 |
| D2 | Cost values: missed toxic/allergenic vs wrongly discarded candidate (per profile) | OWNER-INPUT | WP7 |
| D3 | Throughput target and hardware | OWNER-INPUT | WP10 |
| D4 | Approved data sources and license interpretation | OWNER-INPUT | Gate 1 |
| D5 | Compute budget (CPU/RAM/GPU) | OWNER-INPUT | WP0 |
| D6 | Max altered-residue fraction; minimum sequence length | OWNER-INPUT | WP2 |
| D7 | Calibration/ECE tolerance; ONNX parity tolerance; compression metric-loss tolerance | OWNER-INPUT | WP7, WP10 |

## Blockers (agent fills in)
| ID | Blocker | Impact | Needed from owner |
|---|---|---|---|

## WP0: Setup, audit, baseline freeze
| ID | Task | Acceptance | Status | Commit | Evidence |
|---|---|---|---|---|---|
| 0.1 | Inventory repo, models, data, configs, CI | Audit summary; fix inaccuracies in TECH_SPEC | todo | | |
| 0.2 | Pin environment | Lockfile; clean reproducible build | todo | | |
| 0.3 | Reproduce claimed metrics as legacy baseline | Versioned results file | todo | | |
| 0.4 | Relabel 5 biological controls as smoke tests | Code and docs updated | todo | | |
| 0.5 | Create config skeleton; fill AGENTS.md commands | Files exist | todo | | |
**GATE 0:** audit report and legacy baseline approved.

## WP1: Data governance
| 1.1 | Complete source registry (URL, version, license text, redistribution) | All rows verified or UNVERIFIED with reason | todo | | |
| 1.2 | Implement provenance schema | Schema enforced in loaders | todo | | |
| 1.3 | Implement license gate | UNVERIFIED data blocked without approval | todo | | |
**GATE 1:** owner approves sources and usage.

## WP2: Acquisition and curation
| 2.1 | Ingest approved sources; list requests for unavailable files | Loaders with provenance | todo | | |
| 2.2 | Sanitize with alteration log | Logs and flags tested | todo | | |
| 2.3 | De-duplicate; resolve label conflicts | Removal log | todo | | |
| 2.4 | Build per-organism, length-matched, homology-filtered negatives | Documented residual noise | todo | | |
| 2.5 | Data card per dataset | Cards complete | todo | | |
**GATE 2:** data cards and curation report approved.

## WP3: Splits and leakage audit
| 3.1 | Homology clustering; store cluster artifact | Versioned cluster file | todo | | |
| 3.2 | Grouped CV, family-out, source-held-out, temporal; freeze test set | Splits reproducible | todo | | |
| 3.3 | Leakage gap on legacy models | Reported with CIs | todo | | |
| 3.4 | Shortcut tests | All in EVALUATION_PROTOCOL §2 run | todo | | |
**GATE 3:** leakage report approved; grouped numbers become headline.

## WP4: Features
| 4.1 | AAC, dipeptide, k-mer, ACC, PCP with version strings | Unit tests | todo | | |
| 4.2 | Optional PLM embeddings feasibility (ONNX, size, latency, offline) | Decision recorded | todo | | |
| 4.3 | Feature-parity tests | Golden vectors identical train vs serve | todo | | |

## WP5: Antigenicity models
| 5.1 | Baselines: LR, RF, XGBoost, kNN | Grouped-CV results with CIs | todo | | |
| 5.2 | Local replication of VaxiJen v3.0 design (ACC, 3-model vote); document differences | Results and diff notes | todo | | |
| 5.3 | Bacterial, viral, tumor models; abstain for unsupported organisms | Per-organism artifacts | todo | | |
| 5.4 | Nested grouped-CV tuning; feature pruning | Chosen model justified | todo | | |
**GATE 5:** candidate antigenicity models approved.

## WP6: Allergenicity and toxicity
| 6.1 | Retrain allergenicity; add FAO/WHO-style homology rule | Grouped-CV results; rule tests | todo | | |
| 6.2 | Toxicity: upstream ONNX baseline vs in-house model on independent later-dated set | Clear statement of what could not be re-evaluated | todo | | |
| 6.3 | Error-correlation analysis across predictors | Report | todo | | |

## WP7: Calibration and thresholds
| 7.1 | Calibrate on group-held-out data | Brier, ECE, reliability curves | todo | | |
| 7.2 | Prior-shift correction (configurable) | Tested | todo | | |
| 7.3 | Cost-based thresholds per organism/profile (needs D1, D2) | Frozen versioned config | todo | | |
| 7.4 | Full metric table incl. PPV at 1/2/3% by length bin | Non-ML-readable table | todo | | |

## WP8: Benchmarking (final test set unfrozen once)
| 8.1 | Evaluate finalists once on frozen grouped and temporal sets | Results with CIs | todo | | |
| 8.2 | Reproduce bacterial benchmark (or list data request) | Metrics vs published figures | todo | | |
| 8.3 | Export comparison FASTA for owner's manual VaxiJen v3.0 run | FASTA file | todo | | |
| 8.4 | Analyze returned VaxiJen CSV | AUROC, kappa, disagreement analysis | todo | | |
| 8.5 | Legacy vs new on identical splits | Keep legacy unless new clearly wins | todo | | |
**GATE 8:** benchmark report approved.

## WP9: Robustness and guardrails
| 9.1 | Sliding-window inference | Embedded-segment recall test | todo | | |
| 9.2 | Low-confidence flag for short peptides | Tested | todo | | |
| 9.3 | Applicability domain + conformal outputs | Coverage reported | todo | | |
| 9.4 | Robustness suite | Stability report | todo | | |
| 9.5 | Hard gates in aggregator; therapeutic antigenicity inversion; ranking sanity set | Tests + ranking report | todo | | |

## WP10: Scalability
| 10.1 | Profile baseline throughput/latency/memory | Numbers recorded | todo | | |
| 10.2 | Chunked FASTA streaming; parallel feature extraction | Speedup measured | todo | | |
| 10.3 | Versioned hash cache | No stale hits (tested) | todo | | |
| 10.4 | ONNX for all models with parity tests | Parity within tolerance | todo | | |
| 10.5 | Shrink toxicity model if needed; re-validate | Loss within tolerance | todo | | |
| 10.6 | Job queue for large batches | UI never blocks | todo | | |
| 10.7 | 100k-sequence benchmark vs D3 | Bottleneck report | todo | | |
**GATE 10:** performance report approved.

## WP11: MLOps and monitoring
| 11.1 | Data/model versioning; report stamps | Stamp on every output | todo | | |
| 11.2 | CI retrain pipeline with promotion gate | Gate enforced (EVALUATION_PROTOCOL §8) | todo | | |
| 11.3 | Local per-request logging | No network | todo | | |
| 11.4 | Drift monitoring (PSI, KS, abstention, OOD, length mix) | Alert levels in config | todo | | |
| 11.5 | Shadow/canary with rollback; wet-lab feedback schema; back-testing | Documented and tested | todo | | |

## WP12: Final delivery
| 12.1 | Model Validity Report | All sections, CIs | todo | | |
| 12.2 | Model cards (3 predictors) | Complete | todo | | |
| 12.3 | Operations runbook | Complete | todo | | |
| 12.4 | Remaining risks and open questions | Prioritized list | todo | | |

## Decision log
| Date | Decision | By | Reason |
|---|---|---|---|
