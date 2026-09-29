# AGENTS.md: Peptide Profiler (read this first)

Works as `CLAUDE.md` too. Copy it to the repo root (and duplicate under the other name if your tool needs it).

## What this project is
A fully local, offline pipeline that predicts **toxicity**, **antigenicity** and **allergenicity** of peptides/proteins and ranks candidates for two profiles: **vaccine design** and **peptide-therapeutic screening**. We are upgrading the models on better data, benchmarking them honestly, and making the system scalable and reproducible.

## Read order (before any work)
1. `docs/PRD.md`: what and why, success criteria
2. `docs/TECH_SPEC.md`: current and target architecture
3. `docs/DATA_GOVERNANCE.md`: allowed sources, licenses, provenance, curation rules
4. `docs/EVALUATION_PROTOCOL.md`: how we measure; nothing is "done" without this
5. `docs/TASKS.md`: the work packages, status, gates, blockers, open decisions

## Non-negotiable rules
1. **Never fabricate** numbers, citations, licenses, file contents or dataset availability. Every reported number comes from code run in this session. If something is missing, log it in `docs/TASKS.md` (Blockers) and move on to an independent task.
2. **Homology-aware splits only.** Random K-fold is allowed solely to quantify leakage.
3. **No leakage:** fit preprocessing, feature selection, scaling and calibration inside training folds only. Calibration data is disjoint from train and test.
4. **The final test set is frozen** until WP8. Never tune on it.
5. **Offline at runtime.** No network calls in inference. Never scrape or auto-query the VaxiJen server; comparisons are run manually by the owner.
6. **Do not use data marked UNVERIFIED** in `docs/DATA_GOVERNANCE.md` without owner approval, and never commit or redistribute it.
7. **Preserve interfaces:** CLI (`python src/cli.py ...`) and Streamlit app. Keep the 18 existing tests green.
8. **Determinism:** fixed seeds, pinned dependencies.
9. **Never invent owner decisions** (costs, prevalence, throughput targets). Use the labeled `OWNER-INPUT` placeholders and flag them.
10. **Report negative results plainly.** If new models are not better on grouped/temporal data, keep the legacy model.

## Workflow for every task
1. Restate goal and acceptance criteria (from `docs/TASKS.md`).
2. Write a short plan.
3. Make one small, reviewable change.
4. Add or update tests.
5. Run tests and measurements.
6. Commit (one logical change per commit).
7. Update `docs/TASKS.md` (status, commit hash, evidence).
8. At a **GATE**, stop and report. Do not continue without owner approval.

## Report format (after each task/gate)
What changed · what was measured (numbers with confidence intervals) · what surprised you · what failed or was skipped and why · what you need from the owner.

## Commands
Fill these in during WP0 after reading the repo. Do not guess.
- Install: `uv pip install -r requirements.lock` (or `pip install -r requirements.lock`)
- Run tests: `pytest tests/` (or `.\.venv\Scripts\pytest tests/`)
- Run CLI: `python src/cli.py -i input.fa -o report.csv --format all`
- Run app: `streamlit run app.py`

## Definition of done (project)
Every task in `docs/TASKS.md` is done or explicitly blocked; all tests pass (existing 18 plus new leakage, permutation, parity and robustness tests); every reported number reproducible from a versioned command; no unverified license or fabricated data; all gates approved.
