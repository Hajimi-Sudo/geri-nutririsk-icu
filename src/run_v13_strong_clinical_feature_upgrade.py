#!/usr/bin/env python3
"""V13 strong clinical-feature upgrade.

Goal: test whether a clean, cross-database, non-APACHE-output feature upgrade
can approach or exceed eICU APACHE-IV/IVa predicted mortality while preserving
the GERI-NutriRisk-ICU claim boundary.

Red lines:
- do not use APACHE predicted mortality, APACHE score, or acute physiology score
  as model inputs;
- keep MIMIC source-only external validation separate from eICU domain-updated
  held-out evaluation;
- retain a nutrition-construct ablation so performance gains do not erase the
  target-journal rationale.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
V3_DIR = ROOT / "v3_severity_model"
OUT = ROOT / "v13_strong_clinical_feature_upgrade"
MIMIC_EXTRA_LAB_SUBSET = OUT / "mimic_extra_labevents_24h_source_subset.csv"

V3_FEATURES = [
    "age", "male", "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    "heart_rate", "resp_rate", "temperature_c", "spo2", "map",
    "heart_rate_missing", "resp_rate_missing", "temperature_c_missing", "spo2_missing", "map_missing",
    "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h",
    "urine_24h_ml_missing", "ventilation_24h_missing", "dialysis_24h_missing", "gcs_24h_missing",
]

EXTRA_LABS = [
    "lactate", "platelet", "potassium", "chloride", "bicarbonate", "calcium",
    "magnesium", "inr", "pt", "ptt", "ph", "pco2", "pao2", "fio2", "base_excess",
    "anion_gap", "hemoglobin",
]
EXTRA_FLAGS = [
    "pf_ratio",
    "icu_type_cardiac", "icu_type_surgical", "icu_type_neuro", "icu_type_trauma",
    "icu_type_medical", "icu_type_med_surg",
    "admit_emergency_or_ed", "admit_transfer", "admit_elective",
]
EXTRA_FEATURES = EXTRA_LABS + [f"{x}_missing" for x in EXTRA_LABS] + EXTRA_FLAGS
FEATURES = V3_FEATURES + EXTRA_FEATURES
NUTRITION_CONSTRUCT = {
    "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
}

MIMIC_EXTRA_ITEMIDS = {
    "lactate": {"50813"},
    "platelet": {"51265"},
    "potassium": {"50822", "50971", "52610"},
    "chloride": {"50806", "50902"},
    "bicarbonate": {"50882"},
    "calcium": {"50808", "50960", "50893"},
    "magnesium": {"50960"},
    "inr": {"51237", "51675"},
    "pt": {"51274"},
    "ptt": {"51275"},
    "ph": {"50820"},
    "pco2": {"50818"},
    "pao2": {"50821"},
    "fio2": {"50816"},
    "base_excess": {"50802"},
    "anion_gap": {"50868"},
    "hemoglobin": {"50811", "51222", "51640"},
}

EICU_LAB_PATTERNS = {
    "lactate": re.compile(r"^lactate$", re.I),
    "platelet": re.compile(r"platelet", re.I),
    "potassium": re.compile(r"^potassium$", re.I),
    "chloride": re.compile(r"^chloride$", re.I),
    "bicarbonate": re.compile(r"bicarbonate|total co2|hco3", re.I),
    "calcium": re.compile(r"^calcium$", re.I),
    "magnesium": re.compile(r"^magnesium$", re.I),
    "inr": re.compile(r"inr", re.I),
    "pt": re.compile(r"^pt$|prothrombin", re.I),
    "ptt": re.compile(r"ptt|partial thromboplastin", re.I),
    "ph": re.compile(r"^ph$|arterial ph", re.I),
    "pco2": re.compile(r"pco2", re.I),
    "pao2": re.compile(r"pao2|po2", re.I),
    "fio2": re.compile(r"fio2", re.I),
    "base_excess": re.compile(r"base excess", re.I),
    "anion_gap": re.compile(r"anion gap", re.I),
    "hemoglobin": re.compile(r"hemoglobin|hgb", re.I),
}

VALID_RANGES = {
    "lactate": (0.1, 30.0),
    "platelet": (1.0, 2000.0),
    "potassium": (1.0, 10.0),
    "chloride": (50.0, 150.0),
    "bicarbonate": (2.0, 60.0),
    "calcium": (1.0, 25.0),
    "magnesium": (0.2, 15.0),
    "inr": (0.5, 20.0),
    "pt": (5.0, 150.0),
    "ptt": (5.0, 250.0),
    "ph": (6.5, 8.0),
    "pco2": (5.0, 200.0),
    "pao2": (10.0, 700.0),
    "fio2": (0.20, 1.0),
    "base_excess": (-40.0, 40.0),
    "anion_gap": (0.0, 60.0),
    "hemoglobin": (2.0, 25.0),
}

def gz_rows(path: Path, fields: list[str] | None = None):
    with gzip.open(path, "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def plain_rows(path: Path, fields: list[str] | None = None):
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(str(x).strip())
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def valid(name, value):
    if value is None:
        return None
    if name == "fio2" and value > 1.0 and value <= 100.0:
        value /= 100.0
    if name == "platelet" and value < 5:
        value *= 1000.0
    lo, hi = VALID_RANGES.get(name, (-math.inf, math.inf))
    return value if lo <= value <= hi else None


def dt(x):
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None


def median(values):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def load_base(path: Path):
    rows = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {"id": row["id"], "y": int(float(row["y"]))}
            for feat in V3_FEATURES:
                raw = row.get(feat, "")
                out[feat] = fnum(raw) if raw != "" else None
            for feat in EXTRA_LABS:
                out[feat] = None
                out[f"{feat}_missing"] = 1
            for feat in EXTRA_FLAGS:
                out[feat] = 0.0
            rows.append(out)
    return rows


def mimic_maps():
    stays = {}
    for r in gz_rows(MIMIC / "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "intime", "first_careunit"]):
        t = dt(r["intime"])
        if t:
            stays[r["stay_id"]] = {**r, "intime_dt": t}
    admissions = {}
    for r in gz_rows(MIMIC / "hosp/admissions.csv.gz", ["hadm_id", "admission_type", "admission_location"]):
        admissions[r["hadm_id"]] = r
    return stays, admissions


def eicu_map():
    fields = [
        "patientunitstayid", "uniquepid", "hospitalid", "unittype", "apacheadmissiondx",
        "hospitaladmitsource", "unitadmitsource", "hospitaladmitoffset", "unitstaytype",
    ]
    return {r["patientunitstayid"]: r for r in gz_rows(EICU / "patient.csv.gz", fields)}


def set_icu_and_admission_flags_mimic(rows):
    stays, admissions = mimic_maps()
    for row in rows:
        s = stays.get(row["id"], {})
        adm = admissions.get(s.get("hadm_id", ""), {})
        care = (s.get("first_careunit", "") or "").lower()
        adt = (adm.get("admission_type", "") + " " + adm.get("admission_location", "")).lower()
        row["icu_type_cardiac"] = float("cardiac" in care or "coronary" in care or "cicu" in care or "cvicu" in care)
        row["icu_type_surgical"] = float("surgical" in care or "surgery" in care or "sicu" in care)
        row["icu_type_neuro"] = float("neuro" in care)
        row["icu_type_trauma"] = float("trauma" in care)
        row["icu_type_med_surg"] = float("medical/surgical" in care or "micu/sicu" in care)
        row["icu_type_medical"] = float("medical" in care and not row["icu_type_med_surg"])
        row["admit_emergency_or_ed"] = float("emer" in adt or "emergency" in adt)
        row["admit_transfer"] = float("transfer" in adt)
        row["admit_elective"] = float("elective" in adt)
    return stays


def set_icu_and_admission_flags_eicu(rows):
    emap = eicu_map()
    for row in rows:
        e = emap.get(row["id"], {})
        unit = (e.get("unittype", "") + " " + e.get("apacheadmissiondx", "")).lower()
        src = (e.get("hospitaladmitsource", "") + " " + e.get("unitadmitsource", "") + " " + e.get("unitstaytype", "")).lower()
        row["icu_type_cardiac"] = float("cardiac" in unit or "ccu" in unit or "cticu" in unit or "cardiovascular" in unit)
        row["icu_type_surgical"] = float("surgical" in unit or "surgery" in unit or "operative" in unit or "post-op" in unit)
        row["icu_type_neuro"] = float("neuro" in unit or "stroke" in unit or "neurologic" in unit)
        row["icu_type_trauma"] = float("trauma" in unit)
        row["icu_type_med_surg"] = float("med-surg" in unit or "med surg" in unit)
        row["icu_type_medical"] = float("medical" in unit and not row["icu_type_med_surg"])
        row["admit_emergency_or_ed"] = float("emergency" in src or "ed" in src or "er" in src)
        row["admit_transfer"] = float("transfer" in src or "floor" in src or "ward" in src)
        row["admit_elective"] = float("elective" in src or "scheduled" in src)
    return emap


def ensure_mimic_extra_lab_subset():
    if MIMIC_EXTRA_LAB_SUBSET.exists() and MIMIC_EXTRA_LAB_SUBSET.stat().st_size > 100:
        print(f"using existing MIMIC extra lab subset: {MIMIC_EXTRA_LAB_SUBSET}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    itemids = sorted({x for ids in MIMIC_EXTRA_ITEMIDS.values() for x in ids})
    awk_items = " ".join(itemids)
    cmd = (
        "zcat "
        + str(MIMIC / "hosp/labevents.csv.gz")
        + " | awk -F, 'BEGIN{print \"hadm_id,itemid,charttime,valuenum\"; "
        + f"split(\"{awk_items}\",a,\" \"); for(i in a) ids[a[i]]=1}} "
        + "NR>1 && ($5 in ids) {print $3\",\"$5\",\"$7\",\"$10}' > "
        + str(MIMIC_EXTRA_LAB_SUBSET)
    )
    print("creating MIMIC extra lab subset", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=True)


def add_mimic_extra_labs(rows, stays):
    ensure_mimic_extra_lab_subset()
    by_stay = {r["id"]: r for r in rows}
    hadm_to_stays = defaultdict(list)
    for sid, meta in stays.items():
        if sid in by_stay:
            hadm_to_stays[meta["hadm_id"]].append(sid)
    item_to_lab = {item: lab for lab, ids in MIMIC_EXTRA_ITEMIDS.items() for item in ids}
    vals = {sid: defaultdict(list) for sid in by_stay}
    for r in plain_rows(MIMIC_EXTRA_LAB_SUBSET, ["hadm_id", "itemid", "charttime", "valuenum"]):
        lab = item_to_lab.get(r["itemid"])
        if not lab:
            continue
        value = valid(lab, fnum(r["valuenum"]))
        if value is None:
            continue
        t = dt(r["charttime"])
        for sid in hadm_to_stays.get(r["hadm_id"], []):
            intime = stays[sid]["intime_dt"]
            if t and timedelta(0) <= t - intime <= timedelta(hours=24):
                vals[sid][lab].append(value)
    for sid, row in by_stay.items():
        for lab in EXTRA_LABS:
            if vals[sid].get(lab):
                row[lab] = median(vals[sid][lab])
                row[f"{lab}_missing"] = 0
        add_derived_features(row)


def add_eicu_extra_labs(rows):
    by_id = {r["id"]: r for r in rows}
    ids = set(by_id)
    vals = {sid: defaultdict(list) for sid in ids}
    print("adding eICU apacheApsVar raw physiology", flush=True)
    aps_map = {"ph": "ph", "pco2": "pco2", "pao2": "pao2", "fio2": "fio2"}
    for r in gz_rows(EICU / "apacheApsVar.csv.gz"):
        sid = r.get("patientunitstayid", "")
        if sid not in ids:
            continue
        for lab, col in aps_map.items():
            value = valid(lab, fnum(r.get(col, "")))
            if value is not None:
                vals[sid][lab].append(value)
    print("scanning eICU lab 24h extra variables", flush=True)
    for r in gz_rows(EICU / "lab.csv.gz", ["patientunitstayid", "labresultoffset", "labname", "labresult"]):
        sid = r["patientunitstayid"]
        if sid not in ids:
            continue
        off = fnum(r["labresultoffset"])
        if off is None or off < 0 or off > 1440:
            continue
        name = r["labname"]
        for lab, pat in EICU_LAB_PATTERNS.items():
            if pat.search(name):
                value = valid(lab, fnum(r["labresult"]))
                if value is not None:
                    vals[sid][lab].append(value)
                break
    for sid, row in by_id.items():
        for lab in EXTRA_LABS:
            if vals[sid].get(lab):
                row[lab] = median(vals[sid][lab])
                row[f"{lab}_missing"] = 0
        add_derived_features(row)


def add_derived_features(row):
    pao2 = row.get("pao2")
    fio2 = row.get("fio2")
    if pao2 is not None and fio2 is not None and fio2 > 0:
        row["pf_ratio"] = float(max(20.0, min(700.0, pao2 / fio2)))
    sodium = row.get("sodium")
    chloride = row.get("chloride")
    bicarbonate = row.get("bicarbonate")
    if row.get("anion_gap") is None and sodium is not None and chloride is not None and bicarbonate is not None:
        ag = sodium - chloride - bicarbonate
        row["anion_gap"] = valid("anion_gap", ag)
        row["anion_gap_missing"] = int(row["anion_gap"] is None)


def matrix(rows, features=FEATURES, impute=None):
    X = []
    y = []
    for row in rows:
        y.append(int(row["y"]))
        X.append([row.get(f) if row.get(f) is not None else np.nan for f in features])
    X = np.asarray(X, dtype=float)
    if impute is None:
        impute = np.nanmedian(X, axis=0)
        impute = np.where(np.isnan(impute), 0.0, impute)
    X_i = X.copy()
    inds = np.where(np.isnan(X_i))
    X_i[inds] = np.take(impute, inds[1])
    return X_i, np.asarray(y, dtype=int), impute


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


def metric_block(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc(y, p),
        "brier": brier(y, p),
        "ece_q10": ece(y, p),
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


def fit_lgbm(X, y, seed=20260826, stronger=False):
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    params = {
        "n_estimators": 1400 if stronger else 900,
        "learning_rate": 0.02 if stronger else 0.03,
        "num_leaves": 31 if stronger else 20,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_samples": 80,
        "reg_lambda": 2.0,
        "objective": "binary",
        "random_state": seed,
        "n_jobs": 8,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def get_apache_predictions():
    preds = {}
    for r in gz_rows(EICU / "apachePatientResult.csv.gz", ["patientunitstayid", "apacheversion", "predictedhospitalmortality"]):
        pred = fnum(r["predictedhospitalmortality"])
        if pred is None or not (0 <= pred <= 1):
            continue
        preds.setdefault(r["patientunitstayid"], {})[r["apacheversion"]] = pred
    return preds


def same_subset_metrics(eicu_rows, y, held_mask, predictions):
    apache = get_apache_predictions()
    out = {}
    for version in ["IV", "IVa"]:
        idx = []
        ap = []
        for i, row in enumerate(eicu_rows):
            if not held_mask[i]:
                continue
            val = apache.get(row["id"], {}).get(version)
            if val is None:
                continue
            idx.append(i)
            ap.append(val)
        idx = np.asarray(idx, dtype=int)
        ap = np.asarray(ap, dtype=float)
        if len(idx) < 100:
            continue
        block = {"APACHE": metric_block(y[idx], ap), "coverage_n": int(len(idx))}
        for name, p in predictions.items():
            block[name] = metric_block(y[idx], p[idx])
            block[f"{name}_minus_APACHE_auc"] = block[name]["auc"] - block["APACHE"]["auc"]
        out[f"APACHE_{version}_heldout_same_subset"] = block
    return out


def paired_bootstrap_auc_diff(y, p_a, p_b, n_resamples=500, seed=20260826):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(auc(yy, p_a[idx]) - auc(yy, p_b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"estimate": float(auc(y, p_a) - auc(y, p_b)), "ci_95": [float(lo), float(hi)], "n_resamples": len(vals)}


def write_features(path, rows):
    fields = ["id", "y"] + FEATURES
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def nonmissing_summary(rows):
    return {f: int(sum(r.get(f) is not None for r in rows)) for f in EXTRA_LABS}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading V3 base feature tables", flush=True)
    mimic = load_base(V3_DIR / "mimic_v3_features_server_only.csv")
    eicu = load_base(V3_DIR / "eicu_v3_features_server_only.csv")

    print("adding ICU/admission flags", flush=True)
    mimic_stays = set_icu_and_admission_flags_mimic(mimic)
    set_icu_and_admission_flags_eicu(eicu)

    print("adding extra labs and blood gas features", flush=True)
    add_mimic_extra_labs(mimic, mimic_stays)
    add_eicu_extra_labs(eicu)

    write_features(OUT / "mimic_v13_features_server_only.csv", mimic)
    write_features(OUT / "eicu_v13_features_server_only.csv", eicu)

    X_mimic, y_mimic, impute = matrix(mimic)
    X_eicu, y_eicu, _ = matrix(eicu, impute=impute)
    train = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    test = ~train
    cal = np.array([(i % 5) == 0 for i in range(len(y_eicu))])
    held = ~cal

    print("training V13 source-only LightGBM", flush=True)
    source = fit_lgbm(X_mimic[train], y_mimic[train], stronger=True)
    p_mimic_test = source.predict_proba(X_mimic[test])[:, 1]
    p_eicu_source = source.predict_proba(X_eicu)[:, 1]
    platt = fit_platt(logit(p_eicu_source[cal]), y_eicu[cal])
    p_eicu_source_platt = apply_platt(p_eicu_source, platt)

    print("training V13 eICU calibration-only update head", flush=True)
    eicu_update = fit_lgbm(X_eicu[cal], y_eicu[cal], seed=20260827, stronger=False)
    p_eicu_update = eicu_update.predict_proba(X_eicu)[:, 1]

    print("training V13 combined MIMIC+eICU calibration update head", flush=True)
    X_combined = np.vstack([X_mimic[train], X_eicu[cal]])
    y_combined = np.concatenate([y_mimic[train], y_eicu[cal]])
    combined = fit_lgbm(X_combined, y_combined, seed=20260828, stronger=True)
    p_eicu_combined = combined.predict_proba(X_eicu)[:, 1]

    print("running nutrition ablation for V13 source-only", flush=True)
    features_no_nutrition = [f for f in FEATURES if f not in NUTRITION_CONSTRUCT]
    X_mimic_non, y_mimic_non, impute_non = matrix(mimic, features_no_nutrition)
    X_eicu_non, y_eicu_non, _ = matrix(eicu, features_no_nutrition, impute=impute_non)
    source_non = fit_lgbm(X_mimic_non[train], y_mimic_non[train], seed=20260829, stronger=True)
    p_eicu_non = source_non.predict_proba(X_eicu_non)[:, 1]
    platt_non = fit_platt(logit(p_eicu_non[cal]), y_eicu_non[cal])
    p_eicu_non_platt = apply_platt(p_eicu_non, platt_non)

    held_predictions = {
        "V13_source_platt": p_eicu_source_platt,
        "V13_eicu_calibration_update": p_eicu_update,
        "V13_combined_domain_update": p_eicu_combined,
    }
    result = {
        "protocol": "V13_STRONG_CLINICAL_FEATURE_UPGRADE",
        "status_date": "2026-08-26",
        "red_lines": [
            "No APACHE predicted mortality, APACHE score, or acute physiology score used as model input.",
            "Source-only validation and eICU domain-updated held-out validation are reported separately.",
            "Nutrition-construct ablation retained.",
        ],
        "feature_count": len(FEATURES),
        "extra_features": EXTRA_FEATURES,
        "extra_lab_nonmissing": {"mimic": nonmissing_summary(mimic), "eicu": nonmissing_summary(eicu)},
        "source_only_lightgbm": {
            "mimic_internal_test": metric_block(y_mimic[test], p_mimic_test),
            "eicu_full_original": metric_block(y_eicu, p_eicu_source),
            "eicu_heldout_platt": metric_block(y_eicu[held], p_eicu_source_platt[held]),
            "platt_parameters": {"intercept": float(platt[0]), "slope": float(platt[1])},
        },
        "eicu_calibration_only_update_head": {
            "eicu_heldout": metric_block(y_eicu[held], p_eicu_update[held]),
            "interpretation": "Model fitted only on the prespecified eICU calibration split; not a source-only external validation model.",
        },
        "combined_mimic_eicu_calibration_update_head": {
            "eicu_heldout": metric_block(y_eicu[held], p_eicu_combined[held]),
            "interpretation": "Model fitted on MIMIC development plus the prespecified eICU calibration split; evaluated on held-out eICU.",
        },
        "source_only_nutrition_construct_ablation": {
            "full_v13_eicu_heldout_platt": metric_block(y_eicu[held], p_eicu_source_platt[held]),
            "without_nutrition_construct_eicu_heldout_platt": metric_block(y_eicu_non[held], p_eicu_non_platt[held]),
            "paired_auc_diff_full_minus_without_nutrition": paired_bootstrap_auc_diff(
                y_eicu[held], p_eicu_source_platt[held], p_eicu_non_platt[held]
            ),
        },
        "apache_same_heldout_subset": same_subset_metrics(eicu, y_eicu, held, held_predictions),
        "claim_boundary": (
            "If domain-updated models exceed APACHE on held-out eICU, this is a local model-updating result, "
            "not source-only transport superiority. If source-only remains below APACHE, APACHE should remain "
            "a contextual clinical benchmark."
        ),
    }
    (OUT / "v13_strong_clinical_feature_upgrade_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for name, block in [
        ("source_only_eicu_full_original", result["source_only_lightgbm"]["eicu_full_original"]),
        ("source_only_eicu_heldout_platt", result["source_only_lightgbm"]["eicu_heldout_platt"]),
        ("eicu_calibration_only_update_heldout", result["eicu_calibration_only_update_head"]["eicu_heldout"]),
        ("combined_domain_update_heldout", result["combined_mimic_eicu_calibration_update_head"]["eicu_heldout"]),
        ("without_nutrition_construct_source_only_platt", result["source_only_nutrition_construct_ablation"]["without_nutrition_construct_eicu_heldout_platt"]),
    ]:
        rows.append({"model": name, **block})
    with (OUT / "table_v13_main_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# V13 Strong Clinical-Feature Upgrade",
        "",
        "Status date: 2026-08-26",
        "",
        "## Main Results",
        "",
        f"- Source-only V13 held-out recalibrated eICU AUROC: {result['source_only_lightgbm']['eicu_heldout_platt']['auc']:.3f}.",
        f"- eICU calibration-only update held-out AUROC: {result['eicu_calibration_only_update_head']['eicu_heldout']['auc']:.3f}.",
        f"- Combined MIMIC plus eICU calibration update held-out AUROC: {result['combined_mimic_eicu_calibration_update_head']['eicu_heldout']['auc']:.3f}.",
        f"- Source-only V13 without nutrition construct held-out AUROC: {result['source_only_nutrition_construct_ablation']['without_nutrition_construct_eicu_heldout_platt']['auc']:.3f}.",
        "",
        "## Interpretation",
        "",
        result["claim_boundary"],
    ]
    (OUT / "V13_STRONG_CLINICAL_FEATURE_UPGRADE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "source_only_platt_auc": result["source_only_lightgbm"]["eicu_heldout_platt"]["auc"],
        "eicu_update_auc": result["eicu_calibration_only_update_head"]["eicu_heldout"]["auc"],
        "combined_update_auc": result["combined_mimic_eicu_calibration_update_head"]["eicu_heldout"]["auc"],
        "apache_same_subset": result["apache_same_heldout_subset"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
