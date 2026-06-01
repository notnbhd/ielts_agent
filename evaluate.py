"""
Evaluate IELTS band score predictions against ground-truth labels.

Metrics: QWK, Pearson, Accuracy, MAE, RMSE

Usage:
    uv run evaluate.py
"""

import csv
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)


def round_to_half(x: float) -> float:
    """Round to the nearest 0.5 step."""
    return round(x * 2) / 2


# ── Load predictions ──────────────────────────────────────────────────────
pred_map: dict[str, float] = {}
with open("sgrade_predictions.csv", "r") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip():
            continue
        pred_map[row[0].strip()] = float(row[1].strip())

# ── Load ground-truth ─────────────────────────────────────────────────────
gt_map: dict[str, float] = {}
with open("IELTS-writing-task-2-evaluation/test.csv", "r") as f:
    for idx, row in enumerate(csv.DictReader(f)):
        sid = f"D_Ielts_Writing_Task_2_Dataset_test_{idx}"
        band_raw = row["band"].strip()
        # "<4" → 3.5
        gt_map[sid] = float(band_raw[1:]) - 0.5 if band_raw.startswith("<") else float(band_raw)

# ── Align ─────────────────────────────────────────────────────────────────
common = sorted(set(pred_map) & set(gt_map))
y_pred = np.array([pred_map[s] for s in common])
y_true = np.array([gt_map[s] for s in common])
n = len(common)

# Discretize to 0.5 steps for QWK & Accuracy (as string labels for sklearn)
y_pred_d = [str(round_to_half(v)) for v in y_pred]
y_true_d = [str(round_to_half(v)) for v in y_true]

# ── Compute metrics ──────────────────────────────────────────────────────
qwk      = cohen_kappa_score(y_true_d, y_pred_d, weights="quadratic")
r, _     = pearsonr(y_pred, y_true)
acc      = accuracy_score(y_true_d, y_pred_d)
mae      = mean_absolute_error(y_true, y_pred)
rmse     = np.sqrt(mean_squared_error(y_true, y_pred))

# ── Print ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  IELTS Band Score Evaluation  (n = {n})")
print(f"{'='*50}\n")
print(f"  {'Metric':<10s}  {'Value':>10s}")
print(f"  {'─'*10}  {'─'*10}")
print(f"  {'QWK':<10s}  {qwk:>10.4f}")
print(f"  {'Pearson':<10s}  {r:>10.4f}")
print(f"  {'Accuracy':<10s}  {acc:>9.1%}")
print(f"  {'MAE':<10s}  {mae:>10.4f}")
print(f"  {'RMSE':<10s}  {rmse:>10.4f}")
print()
