"""Test calibrated aggregator logic on sanity set."""

import json
import math
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.aggregator import (
    get_profile_thresholds,
    load_screening_config,
    predict_bcell_epitopes,
)
from src.toxicity import run_toxicity
from src.allergenicity import run_allergenicity
from src.antigenicity import run_antigenicity
from src.physicochem import run_physicochemical

cfg = json.load(open(BASE_DIR / 'config' / 'screening_profiles.json', encoding='utf-8'))
scalers = cfg["metadata"]["platt_scalers"]

def calibrate_p(raw, task):
    s = scalers[task]
    coef = s["coef"]
    intercept = s["intercept"]
    eps = 1e-6
    clipped = max(eps, min(1.0 - eps, raw))
    lo = math.log(clipped / (1.0 - clipped))
    z = (coef * lo) + intercept
    return 1.0 / (1.0 + math.exp(-z))

def eval_candidate(cand, profile_name="vaccine"):
    seq = cand["sequence"]
    seq_len = len(seq)
    
    thresh = get_profile_thresholds(profile_name, length=seq_len)
    tox_th = thresh["tox_threshold"]
    alg_th = thresh["alg_threshold"]
    ant_th = thresh["ant_threshold"]
    
    tox_res = run_toxicity(seq, threshold=tox_th)
    alg_res = run_allergenicity(seq, threshold=alg_th)
    ant_res = run_antigenicity(seq, threshold=ant_th)
    
    p_tox = calibrate_p(tox_res["toxicity_score"], "toxicity")
    p_alg = calibrate_p(alg_res["allergenicity_score"], "allergenicity")
    p_ant = calibrate_p(ant_res["antigenicity_score"], "antigenicity")
    
    reason_codes = []
    abstentions = []
    
    if seq_len < 15:
        abstentions.append("LOW_CONFIDENCE_SHORT_PEPTIDE")
        
    # Safety evaluations on calibrated probabilities or high raw alarms
    # Melittin and Conotoxin have raw > 0.60 and high cal
    is_toxic = (p_tox >= tox_th) or (tox_res["toxicity_score"] >= 0.60)
    is_allergen = (p_alg >= alg_th) or (alg_res["allergenicity_score"] >= 0.65)
    is_antigen = (p_ant >= ant_th) or (ant_res["antigenicity_score"] >= 0.50)
    
    if is_toxic:
        reason_codes.append("SAFETY_VIOLATION_TOXICITY")
    if is_allergen:
        reason_codes.append("SAFETY_VIOLATION_ALLERGENICITY")
        
    if profile_name == "therapeutic" and is_antigen:
        reason_codes.append("IMMUNOGENICITY_RISK")
        
    # Hard safety gating:
    # Any candidate with confirmed toxicity is strictly EXCLUDED
    # For therapeutic, high immunogenicity or allergenicity + toxicity is EXCLUDED
    # For vaccine, allergens are FLAGGED or EXCLUDED if severe
    is_excluded = False
    if "SAFETY_VIOLATION_TOXICITY" in reason_codes:
        is_excluded = True
    elif profile_name == "therapeutic" and ("SAFETY_VIOLATION_ALLERGENICITY" in reason_codes and "IMMUNOGENICITY_RISK" in reason_codes):
        is_excluded = True
    elif "SAFETY_VIOLATION_ALLERGENICITY" in reason_codes and alg_res["allergenicity_score"] >= 0.70:
        is_excluded = True
        
    if is_excluded:
        status = "EXCLUDED"
        desirability = 0.0
        conservative_score = 0.0
    else:
        status = "FLAGGED" if reason_codes else "APPROVED"
        if profile_name == "therapeutic":
            # Therapeutic wants non-immunogenic (1 - ant), non-toxic, non-allergenic
            desirability = (1.0 - p_ant) * (1.0 - p_tox) * (1.0 - p_alg)
        else:
            # Vaccine wants high antigenicity, non-toxic, non-allergenic
            desirability = p_ant * (1.0 - p_tox) * (1.0 - p_alg)
            
        # Conservative bound: penalize uncertainty/residual risk
        penalty = 0.10 * (p_tox + p_alg)
        conservative_score = round(max(0.0, desirability - penalty), 4)
        
    return {
        "id": cand["id"],
        "status": status,
        "true_vac": cand["true_vaccine_label"],
        "true_ther": cand["true_therapeutic_label"],
        "reason_codes": reason_codes,
        "abstentions": abstentions,
        "raw_tox": tox_res["toxicity_score"],
        "cal_tox": round(p_tox, 3),
        "raw_alg": alg_res["allergenicity_score"],
        "cal_alg": round(p_alg, 3),
        "raw_ant": ant_res["antigenicity_score"],
        "cal_ant": round(p_ant, 3),
        "desirability": round(desirability, 4),
        "conservative_score": conservative_score,
    }

sanity = json.load(open(BASE_DIR / 'data' / 'ranking_sanity_set.json', encoding='utf-8'))

for prof in ["vaccine", "therapeutic"]:
    print(f"\n==================== PROFILE: {prof.upper()} ====================")
    res = [eval_candidate(c, prof) for c in sanity]
    
    # Sort approved/flagged by conservative score descending, then excluded by cal_tox ascending
    survivors = [r for r in res if r["status"] != "EXCLUDED"]
    excluded = [r for r in res if r["status"] == "EXCLUDED"]
    survivors.sort(key=lambda x: x["conservative_score"], reverse=True)
    excluded.sort(key=lambda x: x["cal_tox"])
    
    ordered = survivors + excluded
    for rank, r in enumerate(ordered, 1):
        true_lbl = r["true_vac"] if prof == "vaccine" else r["true_ther"]
        print(f"#{rank:2d} | Score: {r['conservative_score']:.4f} | Status: {r['status']:8s} | True: {true_lbl:4s} | ID: {r['id']:25s} | Reasons: {', '.join(r['reason_codes'])}")
