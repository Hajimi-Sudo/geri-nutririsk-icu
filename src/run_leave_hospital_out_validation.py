#!/usr/bin/env python3
"""EXP-001: Strict Leave-Hospital-Out External Validation Protocol.

GERI-NutriRisk-ICU: an APACHE-contextualized, hospital-robust, and nutrition-aware
domain-updated framework for mortality risk stratification in older ICU patients.
Frontiers in Nutrition - R1 Revision Experiment EXP-001.

========================================================================================
CRITICAL REPRODUCIBILITY NOTICE (Methodological Differentiation):
========================================================================================
This script provides the TRUE center-level leave-hospital-out isolation requested
by Reviewer 2 (Comment 2):
  - In eICU-CRD, 207 unique hospitals are identified.
  - Using random permutation (seed 20260901):
      * 20 hospitals (10%, n = 9,949 stays) -> eICU Domain Updating subset
      * 20 hospitals (10%, n = 4,376 stays) -> eICU Platt Calibration subset
      * 167 hospitals (80%, n = 57,536 stays) -> STRICTLY UNSEEN Held-Out Test Centers
  - ZERO patient stays from any of the 167 held-out hospitals are ever exposed
    to feature imputation, model training, domain updating, or Platt calibration.

NOTE ON HISTORICAL SCRIPT DIFFERENCE:
  - 'run_v16_hospital_cluster_robustness.py' is the historical pre-revision script
    evaluating cluster-bootstrap robustness within the PRIMARY patient-level stratified split.
  - This script ('run_leave_hospital_out_validation.py') executes the STRICT center-level
    partition where entire hospitals are held out before any modeling or calibration.
========================================================================================
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

# Path configurations (auto-detect server or local paths)
SERVER_ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
EICU_DIR = Path("/root/autodl-tmp/eicu/base")
V13_DIR = SERVER_ROOT / "v13_strong_clinical_feature_upgrade"
OUT_DIR = SERVER_ROOT / "revision_experiments"

# Metric calculation helpers
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    total = 0.0
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)

def fit_platt(score: np.ndarray, y: np.ndarray, lr: float = 0.035, epochs: int = 4500) -> np.ndarray:
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w

def apply_platt(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(w[0] + w[1] * logit(p))

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    z = logit(p)
    lr_cal = LogisticRegression(solver="lbfgs", max_iter=1000)
    try:
        lr_cal.fit(z.reshape(-1, 1), y)
        slope = float(lr_cal.coef_[0, 0])
        intercept = float(lr_cal.intercept_[0])
    except Exception:
        slope = float("nan")
        intercept = float("nan")
    
    events = int(y.sum())
    total_n = int(len(y))
    oe_ratio = float(events / max(p.sum(), 1e-6))
    
    return {
        "n": total_n,
        "events": events,
        "event_rate": float(events / max(total_n, 1)),
        "auc": float(auc(y, p)),
        "brier": float(brier(y, p)),
        "ece_q10": float(ece(y, p, bins=10)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "oe_ratio": oe_ratio,
    }

def fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 20261901, n_estimators: int = 1400, lr: float = 0.02) -> lgb.LGBMClassifier:
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_samples=80,
        reg_lambda=2.0,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X, y)
    return model

def load_hospitals(eicu_dir: Path) -> dict[str, dict[str, str]]:
    out = {}
    path = eicu_dir / "patient.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out[row["patientunitstayid"]] = {
                    "hospitalid": row.get("hospitalid", ""),
                    "uniquepid": row.get("uniquepid", "") or row.get("patienthealthsystemstayid", "") or row["patientunitstayid"]
                }
    return out

def load_apache(eicu_dir: Path) -> dict[str, dict[str, float]]:
    out = {}
    path = eicu_dir / "apachePatientResult.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("apacheversion")
                p = row.get("predictedhospitalmortality")
                try:
                    p = float(p)
                except Exception:
                    continue
                if 0 <= p <= 1:
                    out.setdefault(row["patientunitstayid"], {})[v] = p
    return out

def load_features(path: Path):
    ids, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        features = [c for c in reader.fieldnames if c not in {"id", "y"}]
        for row in reader:
            ids.append(row["id"])
            y.append(int(float(row["y"])))
            X.append([float(row[c]) if row.get(c, "") != "" else np.nan for c in features])
    return ids, np.asarray(X, dtype=float), np.asarray(y, dtype=int), features

def impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)

def impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X

def run():
    print("=" * 80)
    print("EXP-001: Strict Leave-Hospital-Out External Validation Protocol")
    print("=" * 80)
    t0 = time.time()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mimic_csv = V13_DIR / "mimic_v13_features_server_only.csv"
    eicu_csv = V13_DIR / "eicu_v13_features_server_only.csv"
    
    if not mimic_csv.exists() or not eicu_csv.exists():
        raise FileNotFoundError(f"Required feature CSVs not found in {V13_DIR}")
        
    print("Loading cohort feature matrices...")
    eicu_ids, X_e_raw, y_e, features = load_features(eicu_csv)
    mimic_ids, X_m_raw, y_m, _ = load_features(mimic_csv)
    print(f"  MIMIC-IV cohort: {len(y_m):,} stays (deaths: {y_m.sum():,}, {y_m.mean():.2%})")
    print(f"  eICU-CRD cohort: {len(y_e):,} stays (deaths: {y_e.sum():,}, {y_e.mean():.2%})")
    print(f"  Total clinical features: {len(features)}")
    
    # Hospital ID extraction
    print("\nLoading hospital identifiers from eICU...")
    h_map = load_hospitals(EICU_DIR)
    hospitals = np.array([h_map.get(str(pid), {}).get("hospitalid", "unknown") for pid in eicu_ids])
    unique_hosp = sorted(set(hospitals) - {"unknown", ""})
    print(f"  Total unique valid hospitals identified in eICU cohort: {len(unique_hosp)}")
    
    # ----------------------------------------------------------------------------------
    # STRICT HOSPITAL-LEVEL SPLIT (EXP-001)
    # ----------------------------------------------------------------------------------
    print("\nExecuting Strict Leave-Hospital-Out Partitioning (seed=20260901)...")
    rng = np.random.default_rng(20260901)
    shuffled_hosp = rng.permutation(unique_hosp)
    n_h = len(shuffled_hosp)
    n_up_h = max(1, int(0.10 * n_h))    # 20 hospitals (10%)
    n_cal_h = max(1, int(0.10 * n_h))   # 20 hospitals (10%)
    up_hosp = set(shuffled_hosp[:n_up_h])
    cal_hosp = set(shuffled_hosp[n_up_h : n_up_h + n_cal_h])
    held_hosp = set(shuffled_hosp[n_up_h + n_cal_h :])  # 167 hospitals (80%)
    
    mask_h_up = np.isin(hospitals, list(up_hosp))
    mask_h_cal = np.isin(hospitals, list(cal_hosp))
    mask_h_held = np.isin(hospitals, list(held_hosp))
    
    print(f"  [1] Domain-Updating Centers: {len(up_hosp):>3} hospitals | {mask_h_up.sum():>6,} patient stays (deaths: {y_e[mask_h_up].sum():,})")
    print(f"  [2] Platt-Calibration Centers: {len(cal_hosp):>3} hospitals | {mask_h_cal.sum():>6,} patient stays (deaths: {y_e[mask_h_cal].sum():,})")
    print(f"  [3] Held-Out Test Centers:   {len(held_hosp):>3} hospitals | {mask_h_held.sum():>6,} patient stays (deaths: {y_e[mask_h_held].sum():,})")
    assert len(held_hosp) == 167, f"Expected 167 heldout hospitals, got {len(held_hosp)}"
    assert mask_h_held.sum() == 57536, f"Expected 57536 heldout stays, got {mask_h_held.sum()}"
    
    # Feature Imputation: fit ONLY on training sets (MIMIC-IV development + update hospitals)
    print("\nFitting baseline median imputer on development + update cohort...")
    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    dev_pool_raw = np.vstack([X_m_raw[m_train], X_e_raw[mask_h_up]])
    med = impute_fit(dev_pool_raw)
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)
    
    # Model Training: MIMIC-IV development + eICU update hospitals (seed=20261901)
    print("Training GERI-NutriRisk-ICU LightGBM on MIMIC-IV + 20 update hospitals...")
    X_train_comb = np.vstack([X_m[m_train], X_e[mask_h_up]])
    y_train_comb = np.concatenate([y_m[m_train], y_e[mask_h_up]])
    m_lho = fit_lgbm(X_train_comb, y_train_comb, seed=20261901)
    
    # Calibration on 20 calibration hospitals
    print("Calibrating via Platt scaling on 20 disjoint calibration hospitals...")
    p_lho_raw = m_lho.predict_proba(X_e)[:, 1]
    w_lho = fit_platt(logit(p_lho_raw[mask_h_cal]), y_e[mask_h_cal])
    p_lho_platt = apply_platt(p_lho_raw, w_lho)
    print(f"  Platt scaling parameters: intercept = {w_lho[0]:.4f}, slope = {w_lho[1]:.4f}")
    
    # Primary Evaluation on 167 Unseen Held-Out Centers
    print("\n" + "=" * 50)
    print("PRIMARY RESULT: 167 UNSEEN HELD-OUT HOSPITALS")
    print("=" * 50)
    y_test = y_e[mask_h_held]
    p_test = p_lho_platt[mask_h_held]
    primary_metrics = metrics(y_test, p_test)
    
    print(f"  Stays (N):                 {primary_metrics['n']:,}")
    print(f"  In-Hospital Deaths:        {primary_metrics['events']:,} ({primary_metrics['event_rate']:.2%})")
    print(f"  AUROC:                     {primary_metrics['auc']:.4f}")
    print(f"  Brier Score:               {primary_metrics['brier']:.4f}")
    print(f"  Expected Calib Error (ECE):{primary_metrics['ece_q10']:.4f}")
    print(f"  Calibration Slope:         {primary_metrics['calibration_slope']:.4f}")
    print(f"  Calibration Intercept:     {primary_metrics['calibration_intercept']:.4f}")
    print(f"  O/E Ratio:                 {primary_metrics['oe_ratio']:.4f}")
    
    # Sensitivity analyses on held-out hospitals
    print("\nExecuting sensitivity analyses on held-out hospitals...")
    held_hospitals_arr = hospitals[mask_h_held]
    h_counts = Counter(held_hospitals_arr)
    top10_held = {h for h, _ in h_counts.most_common(10)}
    not_top10_mask = np.isin(held_hospitals_arr, list(top10_held), invert=True)
    n_ge_100_mask = np.array([h_counts[h] >= 100 for h in held_hospitals_arr])
    n_ge_200_mask = np.array([h_counts[h] >= 200 for h in held_hospitals_arr])
    
    sens_rows = [
        {"subset": "All 167 Unseen Held-Out Hospitals", **primary_metrics},
        {"subset": "Excluding Top-10 Volume Hospitals", **metrics(y_test[not_top10_mask], p_test[not_top10_mask])},
        {"subset": "Hospitals with N >= 100", **metrics(y_test[n_ge_100_mask], p_test[n_ge_100_mask])},
        {"subset": "Hospitals with N >= 200", **metrics(y_test[n_ge_200_mask], p_test[n_ge_200_mask])},
    ]
    
    # Per-hospital performance breakdown
    print("Calculating per-hospital performance across 167 held-out centers...")
    hosp_perf_rows = []
    for h in sorted(held_hosp):
        h_sub = held_hospitals_arr == h
        nh = int(h_sub.sum())
        evth = int(y_test[h_sub].sum())
        ph_sub = p_test[h_sub]
        yh_sub = y_test[h_sub]
        auc_val = auc(yh_sub, ph_sub) if nh >= 50 and 0 < evth < nh else np.nan
        brier_val = brier(yh_sub, ph_sub)
        oe_val = evth / max(ph_sub.sum(), 1e-6)
        hosp_perf_rows.append({
            "hospitalid": h,
            "n_stays": nh,
            "events": evth,
            "mortality_rate": float(evth / max(nh, 1)),
            "auroc": float(auc_val) if np.isfinite(auc_val) else "",
            "brier": float(brier_val),
            "oe_ratio": float(oe_val),
        })
    
    # Save artifacts
    lho_summary = {
        "protocol": "Strict Leave-Hospital-Out Multicenter Validation (EXP-001)",
        "update_hospitals": len(up_hosp),
        "calibration_hospitals": len(cal_hosp),
        "heldout_hospitals": len(held_hosp),
        "update_stays": int(mask_h_up.sum()),
        "calibration_stays": int(mask_h_cal.sum()),
        "heldout_stays": int(mask_h_held.sum()),
        "platt_parameters": {"intercept": float(w_lho[0]), "slope": float(w_lho[1])},
        "leave_hospital_out_metrics": primary_metrics,
        "sensitivity_analysis": sens_rows,
        "execution_time_seconds": float(time.time() - t0),
    }
    
    out_json = OUT_DIR / "exp001_leave_hospital_out.json"
    out_json.write_text(json.dumps(lho_summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON -> {out_json}")
    
    out_sens_csv = OUT_DIR / "table_exp001_leave_hospital_out_sensitivity.csv"
    with out_sens_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sens_rows)
    print(f"Saved sensitivity table -> {out_sens_csv}")
    
    out_hosp_csv = OUT_DIR / "table_exp001_hospital_level_performance.csv"
    with out_hosp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hosp_perf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hosp_perf_rows)
    print(f"Saved per-hospital metrics table -> {out_hosp_csv}")
    
    print(f"\n=== All EXP-001 computations completed successfully in {time.time() - t0:.1f}s ===")

if __name__ == "__main__":
    run()
