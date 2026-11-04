#!/usr/bin/env python3
"""V4 robustness package for manuscript-grade reporting.

Uses server-only V3 feature tables to create aggregate internal validation,
external validation, subgroup metrics, feature-family ablations, decision-curve
tables, and plot-ready calibration/ROC data.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import run_v1_labs_external_validation as v1
import run_v2_vitals_external_validation as v2
import run_v3_severity_external_validation as v3


V3_DIR = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v3_severity_model")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v4_robustness")

FEATURE_SETS = {
    "V0_demographics_bmi_nutrition_support": ["age", "male", "bmi", "enteral_24h", "parenteral_24h"],
    "V1_plus_labs": v1.FEATURES,
    "V2_plus_vitals": v2.FEATURES,
    "V3_plus_severity": v3.FEATURES,
    "nutrition_metabolic_core": [
        "age", "male", "bmi", "enteral_24h", "parenteral_24h",
        "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
        "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
        "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    ],
    "severity_only_baseline": ["age", "male", "heart_rate", "resp_rate", "spo2", "map", "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h"] + [f"{x}_missing" for x in ["heart_rate", "resp_rate", "spo2", "map", "urine_24h_ml", "gcs_24h"]],
}


def load_table(path: Path) -> list[dict]:
    out = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {"id": row["id"], "y": int(float(row["y"]))}
            for k, v in row.items():
                if k in {"id", "y"}:
                    continue
                rec[k] = float(v) if v != "" else None
            out.append(rec)
    return out


def matrix(data: list[dict], features: list[str], impute=None, scaler=None):
    if impute is None:
        impute = {}
        for feat in features:
            vals = [r.get(feat) for r in data if r.get(feat) is not None]
            impute[feat] = float(np.median(vals)) if vals else 0.0
    raw = np.array([[r.get(feat) if r.get(feat) is not None else impute[feat] for feat in features] for r in data], dtype=float)
    if scaler is None:
        scaler = {}
        for j, feat in enumerate(features):
            mu = float(raw[:, j].mean())
            sd = float(raw[:, j].std()) or 1.0
            scaler[feat] = (mu, sd)
    X = raw.copy()
    for j, feat in enumerate(features):
        mu, sd = scaler[feat]
        X[:, j] = (X[:, j] - mu) / sd
    return np.column_stack([np.ones(len(data)), X]), np.array([r["y"] for r in data], dtype=float), impute, scaler


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))

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
            "pred_mean": round(float(p[mask].mean()), 5),
            "obs_rate": round(float(y[mask].mean()), 5),
        })
    return out


def metric_block(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": round(float(y.mean()), 5),
        "auc": round(auc(y, p), 5),
        "brier": round(brier(y, p), 5),
        "calibration_bins": calibration_bins(y, p),
    }


def fit_logistic(X, y, lr=0.03, l2=0.02, epochs=5000):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def fit_platt(score, y, lr=0.03, epochs=5000):
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w


def apply_platt(score, w):
    return sigmoid(w[0] + w[1] * score)


def roc_points(y, p, n=101):
    thresholds = np.quantile(p, np.linspace(1, 0, n))
    pts = []
    positives = y.sum()
    negatives = len(y) - positives
    for th in thresholds:
        pred = p >= th
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        pts.append({
            "threshold": float(th),
            "tpr": tp / positives if positives else 0.0,
            "fpr": fp / negatives if negatives else 0.0,
        })
    return pts


def decision_curve(y, p, thresholds=None):
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.02, 0.31, 0.02)]
    out = []
    n = len(y)
    prevalence = float(y.mean())
    for pt in thresholds:
        pred = p >= pt
        tp = float(((pred == 1) & (y == 1)).sum()) / n
        fp = float(((pred == 1) & (y == 0)).sum()) / n
        nb_model = tp - fp * pt / (1 - pt)
        nb_all = prevalence - (1 - prevalence) * pt / (1 - pt)
        out.append({
            "threshold": float(pt),
            "net_benefit_model": round(nb_model, 6),
            "net_benefit_treat_all": round(nb_all, 6),
            "net_benefit_treat_none": 0.0,
        })
    return out


def subgroup_metrics(data, y, p):
    subgroups = {}
    definitions = {
        "sex_male": lambda r: "male" if r.get("male") == 1 else "female_or_other",
        "age_group": lambda r: "65-74" if r.get("age", 0) < 75 else ("75-84" if r.get("age", 0) < 85 else "85+"),
        "albumin_available": lambda r: "available" if r.get("albumin") is not None else "missing",
        "ventilation_24h": lambda r: "yes" if r.get("ventilation_24h") == 1 else "no",
        "enteral_24h": lambda r: "yes" if r.get("enteral_24h") == 1 else "no",
    }
    for name, fn in definitions.items():
        groups = defaultdict(list)
        for i, row in enumerate(data):
            groups[fn(row)].append(i)
        subgroups[name] = {}
        for g, idx in groups.items():
            if len(idx) < 500:
                continue
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            subgroups[name][g] = {
                "n": int(len(idx)),
                "events": int(yy.sum()),
                "event_rate": round(float(yy.mean()), 5),
                "auc": round(auc(yy, p[idx]), 5),
                "brier": round(brier(yy, p[idx]), 5),
            }
    return subgroups


def run_feature_set(name, features, mimic, eicu, train_mask=None):
    if train_mask is None:
        train_mask = np.ones(len(mimic), dtype=bool)
    train = [mimic[i] for i in np.where(train_mask)[0]]
    internal_test = [mimic[i] for i in np.where(~train_mask)[0]]
    Xm_train, ym_train, impute, scaler = matrix(train, features)
    w = fit_logistic(Xm_train, ym_train)
    out = {"features": features}
    if internal_test:
        Xm_test, ym_test, _, _ = matrix(internal_test, features, impute=impute, scaler=scaler)
        pm_test = sigmoid(Xm_test @ w)
        out["mimic_internal_test"] = metric_block(ym_test, pm_test)
    Xm_all, ym_all, _, _ = matrix(mimic, features, impute=impute, scaler=scaler)
    Xe, ye, _, _ = matrix(eicu, features, impute=impute, scaler=scaler)
    pm_all = sigmoid(Xm_all @ w)
    pe = sigmoid(Xe @ w)
    out["mimic_apparent_or_all"] = metric_block(ym_all, pm_all)
    out["eicu_external"] = metric_block(ye, pe)
    return out, pe, ye


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    train_mask = np.array([(i % 5) != 0 for i in range(len(mimic))], dtype=bool)

    feature_results = {}
    external_predictions = {}
    ye = None
    for name, feats in FEATURE_SETS.items():
        print(f"running feature set {name}", flush=True)
        res, pe, y_ext = run_feature_set(name, feats, mimic, eicu, train_mask=train_mask)
        feature_results[name] = res
        external_predictions[name] = pe
        ye = y_ext

    v3_p = external_predictions["V3_plus_severity"]
    score = logit(v3_p)
    idx = np.arange(len(ye))
    cal = idx % 5 == 0
    test = ~cal
    platt = fit_platt(score[cal], ye[cal])
    p_platt = apply_platt(score, platt)

    result = {
        "design": {
            "mimic_internal_split": "row_index_mod_5_not_equal_0 train; row_index_mod_5_equal_0 internal test",
            "eicu_domain_calibration": "row_index_mod_5_equal_0 calibration; remaining 80% held-out external test",
            "patient_level_tables": "server-only",
        },
        "feature_set_results": feature_results,
        "v3_external_full": metric_block(ye, v3_p),
        "v3_eicu_calibration_split": metric_block(ye[cal], v3_p[cal]),
        "v3_eicu_heldout_original": metric_block(ye[test], v3_p[test]),
        "v3_eicu_heldout_platt": metric_block(ye[test], p_platt[test]),
        "v3_platt_parameters": {"intercept": float(platt[0]), "slope": float(platt[1])},
        "v3_eicu_subgroups_original": subgroup_metrics(eicu, ye, v3_p),
        "v3_eicu_subgroups_platt": subgroup_metrics(eicu, ye, p_platt),
        "plot_data": {
            "v3_eicu_roc": roc_points(ye, v3_p),
            "v3_eicu_calibration_original": calibration_bins(ye, v3_p),
            "v3_eicu_calibration_platt_heldout": calibration_bins(ye[test], p_platt[test]),
            "v3_eicu_decision_curve_original": decision_curve(ye, v3_p),
            "v3_eicu_decision_curve_platt_heldout": decision_curve(ye[test], p_platt[test]),
        },
    }
    (OUT / "v4_robustness_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    ablation_lines = [
        "| Model | MIMIC internal test AUROC | eICU external AUROC | eICU external Brier |",
        "|---|---:|---:|---:|",
    ]
    for name, res in feature_results.items():
        internal = res.get("mimic_internal_test", {})
        external = res["eicu_external"]
        ablation_lines.append(
            f"| {name} | {internal.get('auc', 'NA')} | {external['auc']} | {external['brier']} |"
        )

    lines = [
        "# V4 Robustness Package",
        "",
        "Status date: 2026-08-26",
        "",
        "## Design",
        "",
        "V4 reuses the server-only V3 feature tables to generate manuscript-grade robustness checks: MIMIC internal validation, eICU external validation, feature-family ablation, subgroup performance, calibration, and decision-curve data.",
        "",
        "## Feature-Family Ablation",
        "",
        *ablation_lines,
        "",
        "## Current Main Model",
        "",
        f"- V3 full eICU external AUROC: {result['v3_external_full']['auc']}",
        f"- V3 full eICU external Brier: {result['v3_external_full']['brier']}",
        f"- V3 held-out eICU Platt AUROC: {result['v3_eicu_heldout_platt']['auc']}",
        f"- V3 held-out eICU Platt Brier: {result['v3_eicu_heldout_platt']['brier']}",
        "",
        "## Outputs",
        "",
        "- `v4_robustness_metrics.json`: full aggregate metrics, calibration bins, ROC points, decision-curve data, and subgroup results.",
        "",
        "Patient-level data remain on the server only.",
    ]
    (OUT / "V4_ROBUSTNESS_PACKAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "v3_eicu_auc": result["v3_external_full"]["auc"],
        "v3_eicu_brier": result["v3_external_full"]["brier"],
        "v3_platt_heldout_auc": result["v3_eicu_heldout_platt"]["auc"],
        "v3_platt_heldout_brier": result["v3_eicu_heldout_platt"]["brier"],
        "models": {name: feature_results[name]["eicu_external"]["auc"] for name in feature_results},
    }, indent=2))


if __name__ == "__main__":
    main()
