#!/usr/bin/env python3
"""V12 review-repair package for Frontiers in Nutrition triage risks.

This script is intentionally not another open-ended model search. It creates
reviewer-facing evidence for the main external-review blockers: patient-level
splitting, clinical/nutrition baseline comparisons, paired incremental value,
calibration summaries, and reproducibility/log manifests.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
V3_DIR = ROOT / "v3_severity_model"
OUT = ROOT / "v12_review_repair_package"

FEATURES = [
    "age", "male", "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    "heart_rate", "resp_rate", "temperature_c", "spo2", "map",
    "heart_rate_missing", "resp_rate_missing", "temperature_c_missing", "spo2_missing", "map_missing",
    "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h",
    "urine_24h_ml_missing", "ventilation_24h_missing", "dialysis_24h_missing", "gcs_24h_missing",
]

LABS = [
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
]
VITALS = [
    "heart_rate", "resp_rate", "temperature_c", "spo2", "map",
    "heart_rate_missing", "resp_rate_missing", "temperature_c_missing", "spo2_missing", "map_missing",
]
SEVERITY = [
    "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h",
    "urine_24h_ml_missing", "ventilation_24h_missing", "dialysis_24h_missing", "gcs_24h_missing",
]
NUTRITION_CONSTRUCT = ["bmi", "enteral_24h", "parenteral_24h"] + LABS


def gz_rows(path: Path, fields: list[str] | None = None):
    with gzip.open(path, "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def load_feature_table(path: Path, features: list[str] = FEATURES):
    rows, X, y = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            y.append(int(float(row["y"])))
            vals = []
            for feat in features:
                raw = row.get(feat, "")
                vals.append(float(raw) if raw != "" else np.nan)
            X.append(vals)
    return rows, np.array(X, dtype=float), np.array(y, dtype=int)


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


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
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    total = 0.0
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def net_benefit(y, p, threshold):
    pred = p >= threshold
    n = len(y)
    tp = float(((pred == 1) & (y == 1)).sum()) / n
    fp = float(((pred == 1) & (y == 0)).sum()) / n
    return float(tp - fp * threshold / (1 - threshold))


def metric_block(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc(y, p),
        "brier": brier(y, p),
        "ece_q10": ece(y, p),
        "net_benefit_0_10": net_benefit(y, p, 0.10),
        "net_benefit_0_20": net_benefit(y, p, 0.20),
    }


def fit_platt(score, y, lr=0.035, epochs=4500):
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w


def apply_platt(p, w):
    return sigmoid(w[0] + w[1] * logit(p))


def calibration_intercept_slope(y, p, lr=0.035, epochs=4500):
    score = logit(p)
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        pp = sigmoid(X @ w)
        grad = X.T @ (pp - y) / len(y)
        w -= lr * grad
    return {"intercept": float(w[0]), "slope": float(w[1])}


def paired_bootstrap_diff(y, p_full, p_other, n_resamples=500, seed=20260826):
    rng = np.random.default_rng(seed)
    metrics = {
        "auc_diff": lambda yy, a, b: auc(yy, a) - auc(yy, b),
        "brier_diff": lambda yy, a, b: brier(yy, a) - brier(yy, b),
        "ece_q10_diff": lambda yy, a, b: ece(yy, a) - ece(yy, b),
        "net_benefit_0_10_diff": lambda yy, a, b: net_benefit(yy, a, 0.10) - net_benefit(yy, b, 0.10),
        "net_benefit_0_20_diff": lambda yy, a, b: net_benefit(yy, a, 0.20) - net_benefit(yy, b, 0.20),
    }
    out = {}
    n = len(y)
    for name, fn in metrics.items():
        point = fn(y, p_full, p_other)
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            vals.append(fn(yy, p_full[idx], p_other[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = {"estimate": float(point), "ci_95": [float(lo), float(hi)], "n_resamples": len(vals)}
    return out


def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X, med):
    out = X.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def fit_lightgbm(X_train, y_train):
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=16,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=100,
        reg_lambda=3.0,
        objective="binary",
        random_state=20260826,
        n_jobs=8,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def mimic_id_map():
    out = {}
    for r in gz_rows(MIMIC / "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "first_careunit"]):
        out[r["stay_id"]] = r
    return out


def eicu_id_map():
    out = {}
    fields = [
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid",
        "hospitalid", "hospitaladmitoffset", "unittype",
    ]
    for r in gz_rows(EICU / "patient.csv.gz", fields):
        out[r["patientunitstayid"]] = r
    return out


def cohort_split_audit(mimic_rows, eicu_rows):
    mid = mimic_id_map()
    eid = eicu_id_map()
    mimic_subjects = [mid.get(r["id"], {}).get("subject_id", "") for r in mimic_rows]
    mimic_hadm = [mid.get(r["id"], {}).get("hadm_id", "") for r in mimic_rows]
    eicu_unique = [eid.get(r["id"], {}).get("uniquepid", "") or eid.get(r["id"], {}).get("patientunitstayid", "") for r in eicu_rows]
    eicu_health = [eid.get(r["id"], {}).get("patienthealthsystemstayid", "") for r in eicu_rows]
    eicu_hosp = [eid.get(r["id"], {}).get("hospitalid", "") for r in eicu_rows]
    mimic_train = [(i % 5) != 0 for i in range(len(mimic_rows))]
    eicu_cal = [(i % 5) == 0 for i in range(len(eicu_rows))]
    def overlap(ids, mask_a):
        a = {x for x, m in zip(ids, mask_a) if m and x}
        b = {x for x, m in zip(ids, mask_a) if (not m) and x}
        return len(a & b)
    return {
        "mimic_rows": len(mimic_rows),
        "mimic_unique_subjects": len(set(x for x in mimic_subjects if x)),
        "mimic_unique_hadm": len(set(x for x in mimic_hadm if x)),
        "mimic_subject_overlap_train_internal_test": overlap(mimic_subjects, mimic_train),
        "eicu_rows": len(eicu_rows),
        "eicu_unique_uniquepid": len(set(x for x in eicu_unique if x)),
        "eicu_unique_healthsystem_stays": len(set(x for x in eicu_health if x)),
        "eicu_unique_hospitals": len(set(x for x in eicu_hosp if x)),
        "eicu_uniquepid_overlap_calibration_heldout": overlap(eicu_unique, eicu_cal),
        "eicu_hospital_top10": Counter(x for x in eicu_hosp if x).most_common(10),
        "interpretation": (
            "Feature tables contain the first older ICU/unit stay per MIMIC subject_id "
            "and per eICU uniquepid, so the current row-index internal/calibration splits "
            "are patient-level under this cohort construction."
        ),
    }


def build_eicu_apache_baselines(eicu_rows, y_eicu, held):
    wanted = {r["id"] for r in eicu_rows}
    versions = {}
    for r in gz_rows(EICU / "apachePatientResult.csv.gz"):
        sid = r.get("patientunitstayid", "")
        if sid not in wanted:
            continue
        ver = r.get("apacheversion", "") or "unknown"
        pred = fnum(r.get("predictedhospitalmortality", ""))
        score = fnum(r.get("apachescore", ""))
        if pred is not None and 0 <= pred <= 1:
            versions.setdefault(ver, {})[sid] = {"pred": pred, "score": score}
    out = {}
    for ver, vals in sorted(versions.items()):
        idx = []
        pred = []
        score = []
        for i, row in enumerate(eicu_rows):
            v = vals.get(row["id"])
            if not v:
                continue
            idx.append(i)
            pred.append(v["pred"])
            score.append(v["score"] if v["score"] is not None else v["pred"])
        idx = np.array(idx, dtype=int)
        pred = np.array(pred, dtype=float)
        score = np.array(score, dtype=float)
        if len(idx) < 100:
            continue
        yy = y_eicu[idx]
        out[f"APACHE_{ver}_predicted_hospital_mortality"] = {
            **metric_block(yy, pred),
            "coverage_n": int(len(idx)),
            "coverage_fraction": float(len(idx) / len(eicu_rows)),
            "calibration_intercept_slope": calibration_intercept_slope(yy, pred),
        }
        cal_sub = np.array([(i % 5) == 0 for i in idx])
        if cal_sub.sum() > 20 and (~cal_sub).sum() > 20:
            w = fit_platt(score[cal_sub], yy[cal_sub])
            pp = sigmoid(w[0] + w[1] * score)
            out[f"APACHE_{ver}_score_platt_heldout"] = {
                **metric_block(yy[~cal_sub], pp[~cal_sub]),
                "coverage_n": int((~cal_sub).sum()),
                "coverage_fraction": float((~cal_sub).sum() / len(eicu_rows)),
                "platt_intercept": float(w[0]),
                "platt_slope": float(w[1]),
            }
    return out


def sofa_lite_score(row):
    score = 0.0
    spo2 = fnum(row.get("spo2", ""))
    vent = fnum(row.get("ventilation_24h", ""))
    rr = fnum(row.get("resp_rate", ""))
    if vent and vent > 0:
        score += 2
    if spo2 is not None:
        if spo2 < 85:
            score += 3
        elif spo2 < 90:
            score += 2
        elif spo2 < 94:
            score += 1
    if rr is not None and rr >= 30:
        score += 1
    mapv = fnum(row.get("map", ""))
    if mapv is not None and mapv < 70:
        score += 1
    creat = fnum(row.get("creatinine", ""))
    dialysis = fnum(row.get("dialysis_24h", ""))
    urine = fnum(row.get("urine_24h_ml", ""))
    if dialysis and dialysis > 0:
        score += 4
    elif creat is not None:
        if creat >= 5:
            score += 4
        elif creat >= 3.5:
            score += 3
        elif creat >= 2:
            score += 2
        elif creat >= 1.2:
            score += 1
    if urine is not None:
        if urine < 200:
            score += 2
        elif urine < 500:
            score += 1
    bili = fnum(row.get("bilirubin", ""))
    if bili is not None:
        if bili >= 12:
            score += 4
        elif bili >= 6:
            score += 3
        elif bili >= 2:
            score += 2
        elif bili >= 1.2:
            score += 1
    gcs = fnum(row.get("gcs_24h", ""))
    if gcs is not None:
        if gcs < 6:
            score += 4
        elif gcs <= 9:
            score += 3
        elif gcs <= 12:
            score += 2
        elif gcs <= 14:
            score += 1
    return score


def eicu_comorbidity_counts(wanted):
    patterns = {
        "cancer": ("cancer", "neoplasm", "lymphoma", "leukemia", "metast"),
        "copd": ("copd", "chronic obstructive", "emphysema"),
        "renal": ("renal failure", "kidney", "dialysis"),
        "liver": ("cirrhosis", "hepatic", "liver failure"),
        "diabetes": ("diabetes", "diabetic"),
        "heart_failure": ("heart failure", "congestive"),
    }
    out = defaultdict(set)
    for r in gz_rows(EICU / "diagnosis.csv.gz", ["patientunitstayid", "diagnosisstring", "icd9code"]):
        sid = r["patientunitstayid"]
        if sid not in wanted:
            continue
        text = (r.get("diagnosisstring", "") + " " + r.get("icd9code", "")).lower()
        for name, terms in patterns.items():
            if any(t in text for t in terms):
                out[sid].add(name)
    return {k: len(v) for k, v in out.items()}


def eicu_score_baselines(eicu_rows, y_eicu):
    eid = eicu_id_map()
    wanted = {r["id"] for r in eicu_rows}
    comorb = eicu_comorbidity_counts(wanted)
    sofa = np.array([sofa_lite_score(r) for r in eicu_rows], dtype=float)
    nutric = []
    for row, s in zip(eicu_rows, sofa):
        pid = row["id"]
        age = fnum(row.get("age", "")) or 65
        raw = eid.get(pid, {})
        off = fnum(raw.get("hospitaladmitoffset", ""))
        hosp_days = max(0.0, -(off or 0.0) / 1440.0)
        score = 0.0
        score += 1 if age >= 75 else 0
        score += 1 if age >= 85 else 0
        score += 1 if s >= 6 else 0
        score += 1 if s >= 10 else 0
        score += 1 if comorb.get(pid, 0) >= 2 else 0
        score += 1 if hosp_days >= 1 else 0
        nutric.append(score)
    nutric = np.array(nutric, dtype=float)
    out = {}
    cal = np.array([(i % 5) == 0 for i in range(len(y_eicu))])
    held = ~cal
    for name, score in {"SOFA_lite_proxy": sofa, "mNUTRIC_lite_proxy": nutric}.items():
        w = fit_platt(score[cal], y_eicu[cal])
        p = sigmoid(w[0] + w[1] * score)
        out[name] = {
            **metric_block(y_eicu[held], p[held]),
            "raw_score_summary": {
                "mean": float(score.mean()),
                "median": float(np.median(score)),
                "p25": float(np.quantile(score, 0.25)),
                "p75": float(np.quantile(score, 0.75)),
            },
            "platt_intercept": float(w[0]),
            "platt_slope": float(w[1]),
            "proxy_warning": "Not an exact published score; included as a harmonized, auditable proxy because exact score reconstruction is limited by cross-database variable availability.",
        }
    return out


def run_lightgbm_feature_set(name, features):
    _, X_mimic, y_mimic = load_feature_table(V3_DIR / "mimic_v3_features_server_only.csv", features)
    _, X_eicu, y_eicu = load_feature_table(V3_DIR / "eicu_v3_features_server_only.csv", features)
    train = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    cal = np.array([(i % 5) == 0 for i in range(len(y_eicu))])
    held = ~cal
    model = fit_lightgbm(X_mimic[train], y_mimic[train])
    p_eicu = model.predict_proba(X_eicu)[:, 1]
    w = fit_platt(logit(p_eicu[cal]), y_eicu[cal])
    p_platt = apply_platt(p_eicu, w)
    return {
        "name": name,
        "features": features,
        "feature_count": len(features),
        "eicu_original": metric_block(y_eicu, p_eicu),
        "eicu_platt_heldout": metric_block(y_eicu[held], p_platt[held]),
        "calibration_intercept_slope_heldout": calibration_intercept_slope(y_eicu[held], p_platt[held]),
        "platt_parameters": {"intercept": float(w[0]), "slope": float(w[1])},
        "predictions_heldout": p_platt[held],
        "y_heldout": y_eicu[held],
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_manifest():
    rows = []
    for path in sorted(ROOT.glob("**/*")):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".csv", ".log"}:
            rows.append({
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mimic_rows, _, _ = load_feature_table(V3_DIR / "mimic_v3_features_server_only.csv")
    eicu_rows, _, y_eicu = load_feature_table(V3_DIR / "eicu_v3_features_server_only.csv")
    held = np.array([(i % 5) != 0 for i in range(len(y_eicu))])

    print("running cohort/split audit", flush=True)
    split_audit = cohort_split_audit(mimic_rows, eicu_rows)

    print("running clinical score baselines", flush=True)
    clinical = {}
    clinical.update(build_eicu_apache_baselines(eicu_rows, y_eicu, held))
    clinical.update(eicu_score_baselines(eicu_rows, y_eicu))

    print("running LightGBM incremental feature-set comparisons", flush=True)
    feature_sets = {
        "full_lightgbm": FEATURES,
        "without_nutrition_support": [f for f in FEATURES if f not in {"enteral_24h", "parenteral_24h"}],
        "without_nutrition_metabolic_labs": [f for f in FEATURES if f not in LABS],
        "without_all_nutrition_construct": [f for f in FEATURES if f not in set(NUTRITION_CONSTRUCT)],
        "non_nutrition_clinical_baseline": ["age", "male"] + VITALS + SEVERITY,
        "severity_only": SEVERITY,
    }
    model_results = {name: run_lightgbm_feature_set(name, feats) for name, feats in feature_sets.items()}
    full = model_results["full_lightgbm"]
    comparisons = {}
    for name, res in model_results.items():
        if name == "full_lightgbm":
            continue
        comparisons[f"full_minus_{name}"] = paired_bootstrap_diff(
            full["y_heldout"], full["predictions_heldout"], res["predictions_heldout"],
            n_resamples=500, seed=20260826 + len(name),
        )

    for res in model_results.values():
        res.pop("predictions_heldout", None)
        res.pop("y_heldout", None)

    manifest = file_manifest()
    log_files = [r for r in manifest if r["relative_path"].endswith(".log")]
    result = {
        "protocol": "V12_REVIEW_REPAIR_PACKAGE",
        "status_date": "2026-08-26",
        "purpose": "Address external-review blockers before Frontiers in Nutrition packaging.",
        "cohort_split_audit": split_audit,
        "clinical_and_nutrition_proxy_baselines": clinical,
        "lightgbm_feature_set_results": model_results,
        "paired_incremental_value": comparisons,
        "reproducibility_manifest": {
            "file_count": len(manifest),
            "log_file_count": len(log_files),
            "log_files": log_files,
            "all_files_table": "table_v12_reproducibility_manifest.csv",
        },
        "claim_boundary": (
            "This repair package supports nutrition-risk-relevant mortality stratification, "
            "not malnutrition diagnosis, nutrition adequacy assessment, nutrition-treatment "
            "response, or prospective clinical utility."
        ),
    }
    (OUT / "v12_review_repair_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(OUT / "table_v12_reproducibility_manifest.csv", manifest)
    write_csv(OUT / "table_v12_model_feature_set_results.csv", [
        {
            "model": name,
            "feature_count": res["feature_count"],
            "eicu_original_auc": res["eicu_original"]["auc"],
            "eicu_original_brier": res["eicu_original"]["brier"],
            "eicu_platt_heldout_auc": res["eicu_platt_heldout"]["auc"],
            "eicu_platt_heldout_brier": res["eicu_platt_heldout"]["brier"],
            "eicu_platt_heldout_ece_q10": res["eicu_platt_heldout"]["ece_q10"],
            "net_benefit_0_10": res["eicu_platt_heldout"]["net_benefit_0_10"],
            "net_benefit_0_20": res["eicu_platt_heldout"]["net_benefit_0_20"],
        }
        for name, res in model_results.items()
    ])
    write_csv(OUT / "table_v12_clinical_proxy_baselines.csv", [
        {
            "baseline": name,
            "n": res.get("n"),
            "events": res.get("events"),
            "auc": res.get("auc"),
            "brier": res.get("brier"),
            "ece_q10": res.get("ece_q10"),
            "net_benefit_0_10": res.get("net_benefit_0_10"),
            "net_benefit_0_20": res.get("net_benefit_0_20"),
            "coverage_fraction": res.get("coverage_fraction", 1.0),
        }
        for name, res in clinical.items()
    ])
    write_csv(OUT / "table_v12_paired_incremental_value.csv", [
        {
            "contrast": contrast,
            "metric": metric,
            "estimate": vals["estimate"],
            "ci95_low": vals["ci_95"][0],
            "ci95_high": vals["ci_95"][1],
            "n_resamples": vals["n_resamples"],
        }
        for contrast, block in comparisons.items()
        for metric, vals in block.items()
    ])
    md = [
        "# V12 Review Repair Package",
        "",
        "Status date: 2026-08-26",
        "",
        "## Purpose",
        "",
        "This package addresses external-review blockers for Frontiers in Nutrition: patient-level partitioning, clinical and nutrition proxy baselines, paired incremental value, calibration summaries, and reproducibility/log evidence.",
        "",
        "## Split Audit",
        "",
        f"- MIMIC rows: {split_audit['mimic_rows']}; unique subject IDs: {split_audit['mimic_unique_subjects']}.",
        f"- MIMIC subject overlap between development and internal test: {split_audit['mimic_subject_overlap_train_internal_test']}.",
        f"- eICU rows: {split_audit['eicu_rows']}; unique uniquepid values: {split_audit['eicu_unique_uniquepid']}; hospitals: {split_audit['eicu_unique_hospitals']}.",
        f"- eICU uniquepid overlap between calibration and held-out evaluation: {split_audit['eicu_uniquepid_overlap_calibration_heldout']}.",
        "",
        "## Main LightGBM Feature-Set Results",
        "",
        "| Model | Features | Held-out recalibrated AUROC | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, res in model_results.items():
        m = res["eicu_platt_heldout"]
        md.append(f"| {name} | {res['feature_count']} | {m['auc']:.3f} | {m['brier']:.3f} | {m['ece_q10']:.3f} |")
    md += [
        "",
        "## Interpretation Boundary",
        "",
        result["claim_boundary"],
        "",
        "Exact mNUTRIC, SOFA, GNRI, and PNI reconstruction is limited by cross-database variable availability. The reported clinical/nutrition scores are therefore marked as proxies unless exact source variables were available.",
    ]
    (OUT / "V12_REVIEW_REPAIR_PACKAGE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "split_audit": split_audit,
        "full_lightgbm_platt_heldout": model_results["full_lightgbm"]["eicu_platt_heldout"],
        "log_file_count": len(log_files),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
