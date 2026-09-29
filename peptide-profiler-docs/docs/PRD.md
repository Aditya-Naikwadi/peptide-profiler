# PRD: Peptide Profiler v2 (Model Upgrade and Validation)

Status: Draft for owner review · Items marked `OWNER-INPUT` must be filled by the owner. Agents must not invent them.

## 1. Problem
Legacy bioinformatics tools for antigenicity, toxicity and allergenicity (e.g. VaxiJen, ToxinPred, AllerTOP) are mostly academic web servers that are rate-limited or intermittently unavailable. Peptide Profiler unifies these predictions locally and offline, with multi-objective candidate ranking. Current reported accuracy is mostly inherited from benchmarks and may not hold on real inputs (random-split leakage, balanced benchmarks vs. rare real-world positives, length shortcuts).

## 2. Users and use cases
| User | Need |
|---|---|
| Vaccine researcher | Rank sequences with high antigenicity and low toxicity/allergenicity |
| Peptide-therapeutics screener | Find stable candidates with low immunogenicity, no toxicity, no allergenicity |
| Maintainer | Retrain, validate, monitor and roll back models safely |

## 3. Goals
- G1. Replace inflated or inherited metrics with honest, homology-aware, prevalence-aware metrics.
- G2. Retrain antigenicity, allergenicity (and evaluate toxicity) on curated, provenance-tracked, newer data.
- G3. Benchmark against published standards, including VaxiJen v3.0, without runtime dependence on external servers.
- G4. Add guardrails: calibration, abstention/out-of-domain detection, length-aware inference, hard safety gates.
- G5. Make the system scalable and reproducible: streaming, caching, ONNX, versioned data and models, CI promotion gate.
- G6. Add monitoring: drift, shadow mode, feedback loop.

## 4. Non-goals
- Conformational (3D) epitope prediction (linear B-cell epitopes only; state this in outputs).
- Replacing the upstream ToxinPred2 model's internals (it is a fixed baseline).
- Automated scraping of any third-party web server.
- Clinical or regulatory claims. Outputs are decision support for research screening.

## 5. Success metrics
All measured on homology-grouped and temporal held-out data, with cluster-bootstrap confidence intervals.

| Metric | Target |
|---|---|
| Leakage gap (random CV vs grouped CV AUROC) | Measured and reported for every model (no target; it is a finding) |
| Label-permutation AUROC | ≈ 0.5 (must pass) |
| PR-AUC, MCC, precision@k, enrichment | Report; targets `OWNER-INPUT` after baseline is known |
| PPV at 1%, 2%, 3% prevalence | Report for each model; minimum acceptable PPV `OWNER-INPUT` |
| Calibration (Brier, ECE) | Report; ECE tolerance `OWNER-INPUT` |
| Bacterial benchmark (Rappuoli-style, 11 proteomes) | Compare fold-enrichment and sensitivity to published VaxiJen v2.0 (~1.2) and v3.0 (~4.5) fold-enrichment; goal: equal or better |
| Throughput | `OWNER-INPUT` sequences/second on `OWNER-INPUT` hardware; 100k-sequence run required |
| Regression | 18 existing tests + new tests green |

## 6. Functional requirements
- FR-1 Ingest, sanitize and log residue alterations; flag heavily altered sequences.
- FR-2 Per-organism antigenicity models (bacterial, viral, tumor); unsupported organisms abstain.
- FR-3 Allergenicity ML score plus deterministic FAO/WHO-style homology rule.
- FR-4 Toxicity: upstream ToxinPred2 ONNX baseline plus an evaluated in-house model.
- FR-5 Calibrated probabilities, prior-shift correction, cost-optimal thresholds per organism and profile.
- FR-6 Outputs include: positive / negative / uncertain / out-of-domain, with reason codes.
- FR-7 Sliding-window inference for long proteins; low-confidence flag for short peptides.
- FR-8 Aggregator applies hard safety gates first, then ranks survivors by desirability; antigenicity direction inverted for the therapeutic profile.
- FR-9 Reports (CSV/JSON/HTML) stamped with model hash, feature version, thresholds, calibration set ID, data-cluster version.
- FR-10 Optional reference column for VaxiJen comparison (imported from owner-supplied CSV; never fetched at runtime).
- FR-11 Retraining pipeline with promotion gate; shadow mode; drift monitoring; wet-lab feedback schema.

## 7. Non-functional requirements
Offline at runtime; deterministic (fixed seeds); pinned dependencies; CLI and Streamlit interfaces preserved; cross-platform (Linux/macOS/Windows, Python 3.10+); CPU-only inference; memory budget `OWNER-INPUT`.

## 8. Release plan
| Release | Scope (see TASKS.md) |
|---|---|
| R1 Honest baseline | WP0-WP3: audit, governance, curated data, leakage report |
| R2 Better models | WP4-WP8: features, retrained models, calibration, benchmarks |
| R3 Safe and scalable | WP9-WP10: guardrails, performance |
| R4 Operable | WP11-WP12: MLOps, monitoring, reports, model cards |

## 9. Risks
| Risk | Mitigation |
|---|---|
| Dataset licenses unconfirmed | License gate; UNVERIFIED data needs owner approval |
| Small datasets (hundreds of proteins) | Grouped CV, CIs, simple models preferred, abstention |
| Poor negatives inflate scores | Length-matched, homology-filtered negatives; document residual noise |
| Honest metrics lower than claims | Report as-is; keep legacy model unless new one clearly wins |
| Compression of toxicity ONNX shifts scores | Re-validate; accept within stated tolerance only |

## 10. Open decisions (owner)
Deployment prevalence · cost values (missed toxic/allergenic vs discarded good candidate) · throughput target · approved data sources · compute budget · deadline. Track in `docs/TASKS.md` → Open Decisions.
