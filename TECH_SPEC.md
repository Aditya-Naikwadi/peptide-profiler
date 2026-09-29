# TECH_SPEC: Peptide Profiler

The "Current" section reflects the owner's description. The agent must verify it against the repo in WP0 and correct this file where it differs.

## 1. Current architecture
```
FASTA -> Sequence Sanitizer/Parser (20 standard AA; cleans X/B/Z)
   -> Physicochemical (Biopython ProtParam + Pfeature 30-D PCP)
   -> Toxicity     : ToxinPred2 RF (ONNX, ~119.5 MB, 20-D AAC), optional NCBI BLASTp hybrid, threshold 0.60
   -> Antigenicity : shallow 100-tree RF (antigen_rf.joblib, depths 2-5, median 3.0, nodes 5-15, median 9.0), 50-D AAC+PCP, threshold 0.50 (optional Vaxign-ML Docker)
   -> Allergenicity: shallow 100-tree RF (allergen_rf.joblib, depths 2-5, median 3.0, nodes 5-17, median 9.0), 50-D AAC+PCP, threshold 0.50 (web API / motif fallbacks)
   -> Aggregator: Kolaskar-Tongaonkar linear B-cell epitopes (window 7, threshold 1.0)
                  + legacy desirability product:
                    Vaccine: D = d_ant * (1 - d_tox) * (1 - d_alg)
                    Therapeutic: D = (1 - d_ant) * (1 - d_tox) * (1 - d_alg)
                    (Vulnerable to compensatory ranking: toxic candidates rank well if antigenic)
   -> Reports (CSV/JSON/HTML)
```
Runtime: Python 3.10-3.13 (active .venv: Python 3.13.5, scikit-learn 1.6.1, onnxruntime 1.20.1, biopython 1.85); Docker (python:3.11-slim, ncbi-blast+); docker-compose with `peptide-profiler` (Streamlit :8501) and `vaxign-ml`. Interfaces: CLI and Streamlit. CI: None initially configured.

## 2. Target architecture (changes)
| Area | Change |
|---|---|
| Data | Versioned, provenance-tracked datasets; homology cluster IDs as artifact |
| Features | Versioned AAC, dipeptide, k-mer, ACC, PCP; optional small protein language-model embeddings (only if they improve held-out-family results and export to ONNX) |
| Antigenicity | Per-organism models (bacterial, viral, tumor); local replication of the VaxiJen v3.0 design: ACC features, XGBoost + RSM-1NN + RF with feature selection, majority vote |
| Allergenicity | Retrained on curated data + deterministic FAO/WHO-style homology rule |
| Toxicity | Upstream ONNX kept as baseline; in-house model evaluated against it |
| Calibration | Per-model, on group-held-out data; prior-shift correction to deployment prevalence |
| Decisions | Cost-optimal thresholds per organism and profile from a config cost matrix |
| Guardrails | Applicability domain, class-conditional conformal prediction, abstention, short-peptide flag, sliding windows for long proteins |
| Aggregator | Hard safety gates before desirability ranking; uncertainty-aware ranking |
| Serving | All models to ONNX where feasible; chunked FASTA streaming; parallel feature extraction; hash-keyed cache; job queue for large batches |
| MLOps | Data/model versioning, CI retrain with promotion gate, local logging, drift monitoring, shadow/canary, feedback schema |

## 3. Proposed layout (adapt to the existing repo; do not reorganize without approval)
```
config/            thresholds.yaml, costs.yaml, prevalence.yaml, alerts.yaml, sources.yaml
data/raw|curated|clusters/    (DVC-tracked; UNVERIFIED-license data never committed)
src/data/          loaders, sanitizer, curation, negatives builder
src/features/      aac, dipeptide, kmer, acc, pcp (each with version string)
src/models/        train, calibrate, ensemble, onnx_export, registry
src/eval/          splits, leakage, shortcuts, metrics, benchmark, robustness
src/guards/        applicability_domain, conformal, windows
src/serve/         pipeline, cache, queue
src/monitor/       drift, shadow, feedback
reports/           validity report, model cards, benchmark outputs
tests/             existing 18 + leakage, permutation, parity, robustness
```

## 4. Key design decisions
- **Retrain, not fine-tune** (random forests and boosted trees don't fine-tune).
- **One config-driven threshold file per organism and profile**, frozen with calibration-set ID; never edited by hand after freeze.
- **Registry**: each model artifact stores hash, feature version, training-data version, cluster version, calibration set ID, metrics with CIs.
- **Report stamp** (every CSV/JSON/HTML): model hash, feature version, thresholds version, calibration set ID, data-cluster version, run date.
- **VaxiJen comparison** enters only as an owner-supplied CSV in `reports/benchmark/`; never as a runtime call.
- **Cache key** = sequence hash + model version + feature version + threshold version.
- **ONNX parity**: native vs ONNX probabilities must match within a stated tolerance (set in WP10).

## 5. Interfaces to preserve
CLI: `python src/cli.py -i input.fa -o report.csv --format all` · Streamlit: `streamlit run app.py` · Existing output columns remain; new columns are additive.

## 6. Verification hooks
Golden-sequence feature vectors; ONNX vs native parity; label-permutation test in CI; robustness suite (mutations, truncations, tags, ambiguous residues); throughput benchmark script for 100k sequences.
