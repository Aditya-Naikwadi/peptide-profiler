"""Audit the actual production models on evaluation_dataset.json and fit true Platt scalers."""

import json
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.toxicity import predict_toxicity_onnx
from src.allergenicity import predict_allergenicity_acc_local
from src.antigenicity import predict_antigenicity_local_ml
from src.calibration import run_group_heldout_calibration

dataset = json.load(open(BASE_DIR / 'data' / 'evaluation_dataset.json', encoding='utf-8'))
clusters = json.load(open(BASE_DIR / 'data' / 'cluster_assignments.json', encoding='utf-8'))['assignments']
cluster_list = [clusters.get(r['id'], 'cluster_unknown') for r in dataset]

y_tox = np.array([r['is_toxic'] for r in dataset], dtype=int)
y_alg = np.array([r['is_allergen'] for r in dataset], dtype=int)
y_ant = np.array([r['is_antigen'] for r in dataset], dtype=int)

print(f"Total dataset entries: {len(dataset)}")
print(f"Toxicity positives: {sum(y_tox)} ({np.mean(y_tox):.1%})")
print(f"Allergenicity positives: {sum(y_alg)} ({np.mean(y_alg):.1%})")
print(f"Antigenicity positives: {sum(y_ant)} ({np.mean(y_ant):.1%})")

# Measure performance of production models directly
tox_raw = np.array([predict_toxicity_onnx(r['sequence'], threshold=0.6)['toxicity_score'] for r in dataset])
alg_raw = np.array([predict_allergenicity_acc_local(r['sequence'], threshold=0.5)['allergenicity_score'] for r in dataset])
ant_raw = np.array([predict_antigenicity_local_ml(r['sequence'], threshold=0.5)['antigenicity_score'] for r in dataset])

from sklearn.metrics import roc_auc_score, average_precision_score

print("\n--- PRODUCTION MODELS DISCRIMINATION ON EVALUATION DATASET ---")
print(f"Toxicity (ONNX RF):      AUROC = {roc_auc_score(y_tox, tox_raw):.4f}, PR-AUC = {average_precision_score(y_tox, tox_raw):.4f}")
print(f"Allergenicity (RF ACC):  AUROC = {roc_auc_score(y_alg, alg_raw):.4f}, PR-AUC = {average_precision_score(y_alg, alg_raw):.4f}")
print(f"Antigenicity (RF ML):    AUROC = {roc_auc_score(y_ant, ant_raw):.4f}, PR-AUC = {average_precision_score(y_ant, ant_raw):.4f}")
