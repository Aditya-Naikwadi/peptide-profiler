"""Calibrate production models and compute cost-optimal thresholds on calibrated probabilities."""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.toxicity import predict_toxicity_onnx
from src.allergenicity import predict_allergenicity_acc_local
from src.antigenicity import predict_antigenicity_local_ml

dataset = json.load(open(BASE_DIR / 'data' / 'evaluation_dataset.json', encoding='utf-8'))
clusters = json.load(open(BASE_DIR / 'data' / 'cluster_assignments.json', encoding='utf-8'))['assignments']
cluster_list = [clusters.get(r['id'], 'cluster_unknown') for r in dataset]
lengths = np.array([r['length'] for r in dataset])

y_tox = np.array([r['is_toxic'] for r in dataset], dtype=int)
y_alg = np.array([r['is_allergen'] for r in dataset], dtype=int)
y_ant = np.array([r['is_antigen'] for r in dataset], dtype=int)

tox_raw = np.array([predict_toxicity_onnx(r['sequence'], threshold=0.6)['toxicity_score'] for r in dataset])
alg_raw = np.array([predict_allergenicity_acc_local(r['sequence'], threshold=0.5)['allergenicity_score'] for r in dataset])
ant_raw = np.array([predict_antigenicity_local_ml(r['sequence'], threshold=0.5)['antigenicity_score'] for r in dataset])

eps = 1e-6
def to_log_odds(p):
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

# Fit Platt scalers
lr_tox = LogisticRegression(C=1.0).fit(to_log_odds(tox_raw), y_tox)
lr_alg = LogisticRegression(C=1.0).fit(to_log_odds(alg_raw), y_alg)
lr_ant = LogisticRegression(C=1.0).fit(to_log_odds(ant_raw), y_ant)

tox_cal = lr_tox.predict_proba(to_log_odds(tox_raw))[:, 1]
alg_cal = lr_alg.predict_proba(to_log_odds(alg_raw))[:, 1]
ant_cal = lr_ant.predict_proba(to_log_odds(ant_raw))[:, 1]

# Cost Matrices
COST_MATRICES = {
    "vaccine": {
        "description": "Vaccine target discovery profile (Prioritizes safety and high antigenicity)",
        "deployment_prevalence": 0.02,
        "costs": {
            "toxicity": {"C_FN": 100000.0, "C_FP": 2500.0},
            "allergenicity": {"C_FN": 50000.0, "C_FP": 2000.0},
            "antigenicity": {"C_FN": 30000.0, "C_FP": 5000.0},
        },
    },
    "therapeutic": {
        "description": "Peptide therapeutic drug screening profile (Prioritizes safety and non-immunogenicity)",
        "deployment_prevalence": 0.02,
        "costs": {
            "toxicity": {"C_FN": 150000.0, "C_FP": 3000.0},
            "allergenicity": {"C_FN": 60000.0, "C_FP": 2500.0},
            "antigenicity": {"C_FN": 75000.0, "C_FP": 3000.0},
        },
    },
}

def find_cost_optimal_threshold(y_true, y_probs, c_fn, c_fp, prevalence=0.02, n_steps=200):
    thresholds = np.linspace(0.01, 0.99, n_steps)
    min_cost = float("inf")
    best_t = 0.5
    best_stats = {}

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        tn = np.sum((y_true == 0) & (preds == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = 1.0 - tpr

        cost = (fnr * prevalence * c_fn) + (fpr * (1.0 - prevalence) * c_fp)
        if cost < min_cost:
            min_cost = cost
            best_t = float(t)
            best_stats = {
                "threshold": float(round(best_t, 3)),
                "expected_cost_per_lead_usd": float(round(cost, 2)),
                "tpr_sensitivity": float(round(tpr, 4)),
                "fpr_rate": float(round(fpr, 4)),
                "specificity": float(round(1.0 - fpr, 4)),
            }
    return best_stats

length_bins = [
    ("all", lambda l: np.ones(len(l), dtype=bool)),
    ("< 15 aa", lambda l: l < 15),
    ("15-50 aa", lambda l: (l >= 15) & (l <= 50)),
    ("50-200 aa", lambda l: (l > 50) & (l <= 200)),
    ("200-500 aa", lambda l: (l > 200) & (l <= 500)),
    ("> 500 aa", lambda l: l > 500),
]

screening_config = {
    "metadata": {
        "version": "2.1.0",
        "calibration_set_id": "swiss_prot_eval_v1_437_clusters",
        "generated_date": "2026-09-29",
        "default_deployment_prevalence": 0.02,
        "description": "Production screening configuration calibrated on actual production models",
        "platt_scalers": {
            "toxicity": {"coef": float(lr_tox.coef_[0][0]), "intercept": float(lr_tox.intercept_[0])},
            "allergenicity": {"coef": float(lr_alg.coef_[0][0]), "intercept": float(lr_alg.intercept_[0])},
            "antigenicity": {"coef": float(lr_ant.coef_[0][0]), "intercept": float(lr_ant.intercept_[0])},
        }
    },
    "profiles": {},
}

for pname, pdata in COST_MATRICES.items():
    costs = pdata["costs"]
    prof_dict = {
        "description": pdata["description"],
        "deployment_prevalence": pdata["deployment_prevalence"],
        "cost_matrix_usd": costs,
        "optimal_thresholds": {
            "toxicity": {},
            "allergenicity": {},
            "antigenicity": {},
        }
    }
    
    for bname, bfilter in length_bins:
        mask = bfilter(lengths)
        if np.sum(mask) < 5:
            mask = np.ones(len(lengths), dtype=bool)
        
        prof_dict["optimal_thresholds"]["toxicity"][bname] = find_cost_optimal_threshold(
            y_tox[mask], tox_cal[mask], costs["toxicity"]["C_FN"], costs["toxicity"]["C_FP"]
        )
        prof_dict["optimal_thresholds"]["allergenicity"][bname] = find_cost_optimal_threshold(
            y_alg[mask], alg_cal[mask], costs["allergenicity"]["C_FN"], costs["allergenicity"]["C_FP"]
        )
        prof_dict["optimal_thresholds"]["antigenicity"][bname] = find_cost_optimal_threshold(
            y_ant[mask], ant_cal[mask], costs["antigenicity"]["C_FN"], costs["antigenicity"]["C_FP"]
        )
    
    screening_config["profiles"][pname] = prof_dict

cfg_path = BASE_DIR / "config" / "screening_profiles.json"
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(screening_config, f, indent=2)

print(f"Updated {cfg_path}")
print("Toxicity thresholds (Vaccine):", {k: v["threshold"] for k, v in screening_config["profiles"]["vaccine"]["optimal_thresholds"]["toxicity"].items()})
print("Allergenicity thresholds (Vaccine):", {k: v["threshold"] for k, v in screening_config["profiles"]["vaccine"]["optimal_thresholds"]["allergenicity"].items()})
print("Antigenicity thresholds (Vaccine):", {k: v["threshold"] for k, v in screening_config["profiles"]["vaccine"]["optimal_thresholds"]["antigenicity"].items()})
