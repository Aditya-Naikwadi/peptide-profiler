"""Script to evaluate ranking sanity benchmark on Vaccine and Therapeutic profiles."""

import json
import sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.aggregator import profile_multiple_sequences, to_dataframe

SANITY_PATH = BASE_DIR / "data" / "ranking_sanity_set.json"
RESULTS_PATH = BASE_DIR / "results" / "ranking_sanity_results.json"


def evaluate_profile(candidates, profile_name: str):
    profiles = profile_multiple_sequences(
        candidates,
        candidate_type=profile_name,
        rank_candidates=True,
        enforce_safety_gates=True,
    )

    label_key = f"true_{profile_name}_label"
    
    # Extract ranking ranks and binary ground truths
    ranks = []
    scores = []
    labels = []
    results_list = []

    for p in profiles:
        cand_id = p["id"]
        # Find matching candidate
        orig = next(c for c in candidates if c["id"] == cand_id)
        true_label_str = orig[label_key]
        binary_label = 1 if true_label_str == "good" else 0
        rank = p["desirability"]["rank"]
        score = p["desirability"]["conservative_rank_score"]

        ranks.append(rank)
        scores.append(score)
        labels.append(binary_label)

        results_list.append({
            "rank": rank,
            "id": cand_id,
            "description": orig["description"],
            "true_label": true_label_str,
            "candidate_status": p["candidate_status"],
            "reason_codes": p.get("reason_codes", []),
            "abstentions": p.get("abstentions", []),
            "raw_antigenicity": p["antigenicity"]["score"],
            "calibrated_antigenicity": p["antigenicity"].get("calibrated_probability", p["antigenicity"]["score"]),
            "raw_toxicity": p["toxicity"]["score"],
            "calibrated_toxicity": p["toxicity"].get("calibrated_probability", p["toxicity"]["score"]),
            "raw_allergenicity": p["allergenicity"]["score"],
            "calibrated_allergenicity": p["allergenicity"].get("calibrated_probability", p["allergenicity"]["score"]),
            "desirability": p["desirability"]["score"],
            "conservative_rank_score": score,
            "expected_cost_usd": p["desirability"].get("expected_cost_usd", 0.0),
        })

    # Sort results_list by rank
    results_list.sort(key=lambda x: x["rank"])

    # Number of good candidates
    n_good = sum(labels)
    top_k = n_good
    top_k_candidates = results_list[:top_k]
    top_k_hits = sum(1 for c in top_k_candidates if c["true_label"] == "good")
    top_k_hit_rate = top_k_hits / max(1, top_k)

    # Spearman rank correlation between rank order and ground truth (reversed rank since rank 1 is highest score)
    # Higher score should correspond to label 1.
    spearman_corr, spearman_pval = spearmanr(scores, labels)
    
    # AUROC of scores predicting binary ground truth
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = None

    # Hard safety gating accuracy: fraction of toxic/allergenic controls excluded
    known_toxins_allergens = [
        c for c in results_list 
        if "Toxin" in c["id"] or "Allergen" in c["id"] or "Venom" in c["id"]
    ]
    tox_alg_excluded = sum(1 for c in known_toxins_allergens if c["candidate_status"] == "EXCLUDED")
    safety_exclusion_rate = tox_alg_excluded / max(1, len(known_toxins_allergens))

    return {
        "profile": profile_name,
        "n_candidates": len(candidates),
        "n_good": n_good,
        "n_bad": len(candidates) - n_good,
        "top_k": top_k,
        "top_k_hits": top_k_hits,
        "top_k_hit_rate": top_k_hit_rate,
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_pval,
        "auroc": auroc,
        "safety_exclusion_rate": safety_exclusion_rate,
        "ranked_candidates": results_list,
    }


def main():
    with open(SANITY_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    vaccine_results = evaluate_profile(candidates, "vaccine")
    therapeutic_results = evaluate_profile(candidates, "therapeutic")

    summary = {
        "benchmark_set": "ranking_sanity_set_v1",
        "n_total": len(candidates),
        "profiles": {
            "vaccine": vaccine_results,
            "therapeutic": therapeutic_results,
        }
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== RANKING SANITY BENCHMARK EVALUATION ===")
    for pname in ["vaccine", "therapeutic"]:
        res = summary["profiles"][pname]
        print(f"\n--- PROFILE: {pname.upper()} ---")
        print(f"Top-{res['top_k']} Hit Rate: {res['top_k_hits']}/{res['top_k']} ({res['top_k_hit_rate']:.1%})")
        print(f"Spearman Rank Correlation: {res['spearman_correlation']:.4f} (p={res['spearman_pvalue']:.4e})")
        print(f"AUROC: {res['auroc']:.4f}")
        print(f"Safety Gate Exclusion Rate for Toxins/Allergens: {res['safety_exclusion_rate']:.1%}")
        print("\nRankings:")
        for c in res["ranked_candidates"]:
            status_str = f"[{c['candidate_status']}]"
            reasons = f" ({', '.join(c['reason_codes'])})" if c["reason_codes"] else ""
            print(f"#{c['rank']:2d} | Score: {c['conservative_rank_score']:.4f} | Status: {status_str:10s} | True: {c['true_label']:4s} | ID: {c['id']:25s}{reasons}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
