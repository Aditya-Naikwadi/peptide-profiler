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
y_tox = np.array([r['is_toxic'] for r in dataset], dtype=int)
y_alg = np.array([r['is_allergen'] for r in dataset], dtype=int)
y_ant = np.array([r['is_antigen'] for r in dataset], dtype=int)

tox_raw = np.array([predict_toxicity_onnx(r['sequence'], threshold=0.6)['toxicity_score'] for r in dataset])
alg_raw = np.array([predict_allergenicity_acc_local(r['sequence'], threshold=0.5)['allergenicity_score'] for r in dataset])
ant_raw = np.array([predict_antigenicity_local_ml(r['sequence'], threshold=0.5)['antigenicity_score'] for r in dataset])

# Fit Platt scalers (logistic regression on log-odds)
eps = 1e-6
def fit_platt(p_raw, y):
    lo = np.log(np.clip(p_raw, eps, 1-eps) / (1 - np.clip(p_raw, eps, 1-eps))).reshape(-1, 1)
    lr = LogisticRegression(C=1.0)
    lr.fit(lo, y)
    return lr

platt_tox = fit_platt(tox_raw, y_tox)
platt_alg = fit_platt(alg_raw, y_alg)
platt_ant = fit_platt(ant_raw, y_ant)

def cal_prob(lr, p_raw):
    lo = np.log(np.clip(p_raw, eps, 1-eps) / (1 - np.clip(p_raw, eps, 1-eps))).reshape(-1, 1)
    return lr.predict_proba(lo)[:, 1]

# Now test on sanity set
sanity = json.load(open(BASE_DIR / 'data' / 'ranking_sanity_set.json', encoding='utf-8'))
header = f"{'Candidate':25s} | {'Vac':4s} | {'Ther':4s} | {'ToxRaw':6s} | {'ToxCal':6s} | {'AlgRaw':6s} | {'AlgCal':6s} | {'AntRaw':6s} | {'AntCal':6s}"
print(header)
print("-" * len(header))
for s in sanity:
    seq = s['sequence']
    t_raw = predict_toxicity_onnx(seq, threshold=0.6)['toxicity_score']
    al_raw = predict_allergenicity_acc_local(seq, threshold=0.5)['allergenicity_score']
    an_raw = predict_antigenicity_local_ml(seq, threshold=0.5)['antigenicity_score']
    
    t_cal = float(cal_prob(platt_tox, np.array([t_raw]))[0])
    al_cal = float(cal_prob(platt_alg, np.array([al_raw]))[0])
    an_cal = float(cal_prob(platt_ant, np.array([an_raw]))[0])
    
    cand_name = s['id'][:25]
    print(f"{cand_name:25s} | {s['true_vaccine_label']:4s} | {s['true_therapeutic_label']:4s} | {t_raw:6.3f} | {t_cal:6.3f} | {al_raw:6.3f} | {al_cal:6.3f} | {an_raw:6.3f} | {an_cal:6.3f}")
