# BLOCKERS: Open Blockers & Required Owner Inputs

This file tracks all active blockers, missing inputs, and decisions requiring owner input. Agents do not invent values for these.

## Active Blockers
*None currently blocking WP0.*

## Open Decisions (from PRD.md / TASKS.md)

| ID | Decision | Current Value / Placeholder | Needed By | Notes |
|---|---|---|---|---|
| **D1** | Deployment prevalence: toxic / allergenic / antigenic | `[OWNER-INPUT: default placeholder 0.01 (1%)]` | WP7 | Natural proteome prevalence for toxic/allergenic peptides is estimated ~1-3%. |
| **D2** | Cost values: missed toxic/allergenic vs wrongly discarded candidate | `[OWNER-INPUT: default placeholder 500.0 USD missed, 10.0 USD false alarm]` | WP7 | Needed to parameterize cost-optimal decision matrices. |
| **D3** | Throughput target and hardware | `[OWNER-INPUT: default placeholder 100 seq/s, 100k batch, CPU-only]` | WP10 | Needed for WP10 scalability benchmark targets. |
| **D4** | Approved data sources and license interpretation | `[OWNER-INPUT: pending Gate 1 approval]` | Gate 1 | Sources in `DATA_GOVERNANCE.md` (Zaharieva et al., VaxiGen, Doneva et al., etc.). |
| **D5** | Compute budget (CPU / RAM / GPU) | `[OWNER-INPUT: local machine, 16GB RAM, CPU-only]` | WP0 | Informs model selection and language model feasibility. |
| **D6** | Max altered-residue fraction; minimum sequence length | `[OWNER-INPUT: default placeholder max 5% altered, min len 15 aa]` | WP2 | Configures sequence sanitizer thresholds. |
| **D7** | Calibration/ECE tolerance; ONNX parity tolerance; compression loss tolerance | `[OWNER-INPUT: ECE < 0.10, ONNX parity < 1e-4, metric loss < 0.02]` | WP7, WP10 | Promotion gate quantitative thresholds. |
