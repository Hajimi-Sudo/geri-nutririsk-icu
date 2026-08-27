#!/usr/bin/env python3
"""V2b external-domain recalibration for the MIMIC->eICU model.

Uses the server-only eICU V2 feature table. A deterministic 20% eICU calibration
split fits calibration-in-the-large and logistic recalibration; the remaining
80% is held out for external test metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


V2_DIR = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v2_vitals_model")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v2b_recalibration")
METRICS = V2_DIR / "v2_vitals_external_validation_metrics.json"
EICU_FEATURES = V2_DIR / "eicu_v2_features_server_only.csv"


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_bins(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        out.append({
            "bin": i + 1,
            "n": int(mask.sum()),
            "pred_mean": round(float(p[mask].mean()), 4),
            "obs_rate": round(float(y[mask].mean()), 4),
        })
    return out


def metric_block(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": round(float(y.mean()), 4),
        "auc": round(auc(y, p), 4),
        "brier": round(brier(y, p), 4),
        "calibration_bins": calibration_bins(y, p),
    }


def fit_calibrator(score, y, slope=True, lr=0.05, epochs=5000, l2=0.0):
    if slope:
        X = np.column_stack([np.ones(len(score)), score])
    else:
        X = np.ones((len(score), 1))
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        if slope and l2:
            grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def apply_calibrator(score, w):
    if len(w) == 1:
        return sigmoid(score + w[0])
    return sigmoid(w[0] + w[1] * score)


def load_eicu_predictions():
    m = json.loads(METRICS.read_text())
    features = m["features"]
    coef = m["coefficients"]
    impute = m["imputation_from_mimic"]
    scaler = m["standardization_from_mimic"]
    w = np.array([coef["intercept"]] + [coef[f] for f in features], dtype=float)
    ids = []
    y = []
    raw = []
    with EICU_FEATURES.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            y.append(float(row["y"]))
            vals = []
            for feat in features:
                val = row.get(feat, "")
                if val == "":
                    val = impute[feat]
                else:
                    val = float(val)
                mu, sd = scaler[feat]
                vals.append((val - mu) / sd)
            raw.append(vals)
    X = np.column_stack([np.ones(len(raw)), np.array(raw, dtype=float)])
    y = np.array(y, dtype=float)
    p = sigmoid(X @ w)
    s = logit(p)
    return ids, y, p, s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, y, p, s = load_eicu_predictions()
    # Deterministic split by row index: 20% calibration, 80% held-out external test.
    idx = np.arange(len(y))
    cal = idx % 5 == 0
    test = ~cal
    w_intercept = fit_calibrator(s[cal], y[cal], slope=False, lr=0.05, epochs=3000)
    w_platt = fit_calibrator(s[cal], y[cal], slope=True, lr=0.03, epochs=5000, l2=0.001)
    p_intercept = apply_calibrator(s, w_intercept)
    p_platt = apply_calibrator(s, w_platt)
    res = {
        "source_model": str(METRICS),
        "feature_table_server_only": str(EICU_FEATURES),
        "split": {"calibration": int(cal.sum()), "heldout_test": int(test.sum()), "rule": "row_index_mod_5_equals_0"},
        "calibration_parameters": {
            "intercept_only_delta": float(w_intercept[0]),
            "platt_intercept": float(w_platt[0]),
            "platt_slope": float(w_platt[1]),
        },
        "calibration_split": {
            "original": metric_block(y[cal], p[cal]),
            "intercept_recalibrated": metric_block(y[cal], p_intercept[cal]),
            "platt_recalibrated": metric_block(y[cal], p_platt[cal]),
        },
        "heldout_external_test": {
            "original": metric_block(y[test], p[test]),
            "intercept_recalibrated": metric_block(y[test], p_intercept[test]),
            "platt_recalibrated": metric_block(y[test], p_platt[test]),
        },
    }
    (OUT / "v2b_recalibration_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [
        "# V2b External Recalibration Results",
        "",
        "Status date: 2026-08-26",
        "",
        "## Design",
        "",
        "The MIMIC-IV trained V2 model was applied to eICU. A deterministic 20% eICU calibration split was used for recalibration, and the remaining 80% was held out for external test evaluation.",
        "",
        "## Held-Out eICU Test Metrics",
        "",
        f"- Original V2: AUROC={res['heldout_external_test']['original']['auc']}; Brier={res['heldout_external_test']['original']['brier']}",
        f"- Intercept-only recalibration: AUROC={res['heldout_external_test']['intercept_recalibrated']['auc']}; Brier={res['heldout_external_test']['intercept_recalibrated']['brier']}",
        f"- Logistic recalibration: AUROC={res['heldout_external_test']['platt_recalibrated']['auc']}; Brier={res['heldout_external_test']['platt_recalibrated']['brier']}",
        "",
        "## Calibration Parameters",
        "",
        f"- Intercept-only delta: {res['calibration_parameters']['intercept_only_delta']:.4f}",
        f"- Logistic recalibration intercept: {res['calibration_parameters']['platt_intercept']:.4f}",
        f"- Logistic recalibration slope: {res['calibration_parameters']['platt_slope']:.4f}",
        "",
        "## Interpretation",
        "",
        "This analysis tests whether a light external-domain calibration layer can repair cross-database probability shift while preserving discrimination. It should be framed as model updating, not as pure untouched external validation.",
    ]
    (OUT / "V2B_EXTERNAL_RECALIBRATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "heldout_original_auc": res["heldout_external_test"]["original"]["auc"],
        "heldout_original_brier": res["heldout_external_test"]["original"]["brier"],
        "heldout_platt_auc": res["heldout_external_test"]["platt_recalibrated"]["auc"],
        "heldout_platt_brier": res["heldout_external_test"]["platt_recalibrated"]["brier"],
        "platt_slope": res["calibration_parameters"]["platt_slope"],
    }, indent=2))


if __name__ == "__main__":
    main()
