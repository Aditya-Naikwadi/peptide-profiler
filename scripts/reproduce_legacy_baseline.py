"""Reproduce and verify legacy baseline metrics on canonical sample peptides.

Executes deterministic inference on examples/sample_peptides.fa without safety gating
to reproduce the exact legacy scores, desirability, and rankings stored in
results/legacy_baseline_metrics.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregator import calculate_desirability, profile_multiple_sequences
from src.parser import parse_fasta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reproduce_baseline(input_fasta: Path, baseline_json: Path) -> bool:
    logger.info(f"Loading input sequences from {input_fasta}...")
    records = parse_fasta(input_fasta)
    logger.info(f"Parsed {len(records)} sequence records.")

    # Profiling without Phase 4 safety gates to extract raw legacy predictions
    profiles = profile_multiple_sequences(
        records,
        candidate_type="vaccine",
        enforce_safety_gates=False,
        rank_candidates=False,
    )

    baseline_data = json.loads(baseline_json.read_text(encoding="utf-8"))
    stored_candidates = {c["id"]: c for c in baseline_data["candidates"]}

    computed = []
    for p in profiles:
        cid = p["id"]
        length = len(p["sequence"])
        raw_tox = p["toxicity"]["score"]
        raw_ant = p["antigenicity"]["score"]
        raw_alg = p["allergenicity"]["score"]

        # Legacy formulas (pre-Phase 4, uncalibrated raw score geometric multiplication)
        vac_score = calculate_desirability(raw_ant, raw_tox, raw_alg, candidate_type="vaccine")
        thr_score = calculate_desirability(raw_ant, raw_tox, raw_alg, candidate_type="therapeutic")

        computed.append({
            "id": cid,
            "length": length,
            "raw_tox": raw_tox,
            "raw_ant": raw_ant,
            "raw_alg": raw_alg,
            "vac_score": vac_score,
            "thr_score": thr_score,
        })

    # Compute ranks
    computed.sort(key=lambda x: x["vac_score"], reverse=True)
    for rank, c in enumerate(computed, 1):
        c["vac_rank"] = rank

    computed.sort(key=lambda x: x["thr_score"], reverse=True)
    for rank, c in enumerate(computed, 1):
        c["thr_rank"] = rank

    computed.sort(key=lambda x: x["vac_rank"])

    all_match = True
    print("\n" + "=" * 90)
    print(f"{'Candidate ID':<32} {'Len':<4} {'Tox':<6} {'Ant':<6} {'Alg':<6} {'VacRank (Score)':<18} {'ThrRank (Score)':<18} {'Status'}")
    print("=" * 90)

    for c in computed:
        cid = c["id"]
        s = stored_candidates.get(cid)
        if not s:
            print(f"Missing in baseline: {cid}")
            all_match = False
            continue

        match = (
            round(s["toxicity"]["score"], 3) == round(c["raw_tox"], 3) and
            round(s["antigenicity"]["score"], 3) == round(c["raw_ant"], 3) and
            round(s["allergenicity"]["score"], 3) == round(c["raw_alg"], 3) and
            round(s["desirability"]["score"], 4) == round(c["vac_score"], 4) and
            s["desirability"]["rank"] == c["vac_rank"]
        )
        if not match:
            all_match = False

        status = "MATCH" if match else "MISMATCH"
        print(
            f"{cid:<32} {c['length']:<4} {c['raw_tox']:<6.3f} {c['raw_ant']:<6.3f} {c['raw_alg']:<6.3f} "
            f"#{c['vac_rank']} ({c['vac_score']:.4f})     #{c['thr_rank']} ({c['thr_score']:.4f})     [{status}]"
        )
    print("=" * 90)
    print(f"Overall Reproduction Status: {'SUCCESS (100% Match)' if all_match else 'FAILED'}\n")
    return all_match


if __name__ == "__main__":
    fa_path = PROJECT_ROOT / "examples" / "sample_peptides.fa"
    json_path = PROJECT_ROOT / "results" / "legacy_baseline_metrics.json"
    success = reproduce_baseline(fa_path, json_path)
    if not success:
        sys.exit(1)
