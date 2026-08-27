#!/usr/bin/env python3
"""V0 MIMIC-IV to eICU external-validation experiment.

This intentionally uses a compact, interpretable feature set and only numpy.
Patient-level feature files remain on the server; only aggregate metrics should
be copied back to the RH workspace.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v0_model")

FEATURES = ["age", "male", "bmi", "enteral_24h", "parenteral_24h"]

ENTERAL_RE = re.compile(r"enteral|tube feed|tube feeding|feeding tube|jevity|promote|nepro|osmolite|glucerna|twocal|vital|nutren", re.I)
PN_RE = re.compile(r"parenteral|tpn|total parenteral|lipid|amino acid", re.I)


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", errors="replace")


def rows(path: Path, fields: list[str] | None = None):
    with open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def fnum(x: str) -> float | None:
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def dt(x: str) -> datetime | None:
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None


def eicu_age(x: str) -> int | None:
    x = (x or "").strip()
    if not x:
        return None
    if x.startswith(">"):
        return 90
    try:
        return int(float(x))
    except Exception:
        return None


def mimic_first_older_stays() -> dict[str, dict]:
    age = {}
    gender = {}
    for r in rows(MIMIC / "hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age"]):
        a = fnum(r["anchor_age"])
        if a is not None:
            age[r["subject_id"]] = a
        gender[r["subject_id"]] = r["gender"]
    mort = {}
    for r in rows(MIMIC / "hosp/admissions.csv.gz", ["hadm_id", "hospital_expire_flag"]):
        mort[r["hadm_id"]] = int(r["hospital_expire_flag"] == "1")

    best = {}
    for r in rows(MIMIC / "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "intime"]):
        sid = r["subject_id"]
        if age.get(sid, -1) < 65:
            continue
        t = dt(r["intime"])
        if t is None:
            continue
        old = best.get(sid)
        if old is None or t < old["intime"]:
            best[sid] = {
                "id": r["stay_id"],
                "subject_id": sid,
                "hadm_id": r["hadm_id"],
                "intime": t,
                "age": age[sid],
                "male": int(gender.get(sid) == "M"),
                "y": mort.get(r["hadm_id"], 0),
                "bmi": None,
                "enteral_24h": 0,
                "parenteral_24h": 0,
            }
    return {v["id"]: v for v in best.values()}


def mimic_bmi_by_subject(stays: dict[str, dict]) -> dict[str, float]:
    subject_ids = {v["subject_id"] for v in stays.values()}
    bmi_by_subject = {}
    weight_kg = {}
    height_cm = {}
    for r in rows(MIMIC / "hosp/omr.csv.gz", ["subject_id", "result_name", "result_value"]):
        sid = r["subject_id"]
        if sid not in subject_ids:
            continue
        name = r["result_name"].lower()
        val = fnum(r["result_value"])
        if val is None:
            continue
        if "bmi" in name and 8 <= val <= 100:
            bmi_by_subject.setdefault(sid, val)
        elif "weight" in name:
            # OMR often stores pounds.
            kg = val * 0.453592 if val > 120 else val
            if 20 <= kg <= 300:
                weight_kg.setdefault(sid, kg)
        elif "height" in name:
            cm = val * 2.54 if val < 100 else val
            if 100 <= cm <= 230:
                height_cm.setdefault(sid, cm)
    for sid in subject_ids:
        if sid not in bmi_by_subject and sid in weight_kg and sid in height_cm:
            bmi_by_subject[sid] = weight_kg[sid] / ((height_cm[sid] / 100) ** 2)
    return bmi_by_subject


def mimic_nutrition_itemids() -> tuple[set[str], set[str]]:
    enteral = set()
    pn = set()
    for r in rows(MIMIC / "icu/d_items.csv.gz"):
        text = " | ".join(str(v) for v in r.values())
        if ENTERAL_RE.search(text):
            enteral.add(r["itemid"])
        if PN_RE.search(text):
            pn.add(r["itemid"])
    return enteral, pn


def add_mimic_nutrition_24h(stays: dict[str, dict]) -> None:
    enteral_ids, pn_ids = mimic_nutrition_itemids()
    stay_ids = set(stays)
    for rel in ["icu/inputevents.csv.gz", "icu/ingredientevents.csv.gz"]:
        for r in rows(MIMIC / rel, ["stay_id", "starttime", "itemid", "amount"]):
            stay_id = r["stay_id"]
            if stay_id not in stay_ids:
                continue
            item = r["itemid"]
            if item not in enteral_ids and item not in pn_ids:
                continue
            t = dt(r["starttime"])
            if t is None:
                continue
            delta = t - stays[stay_id]["intime"]
            if delta < timedelta(0) or delta > timedelta(hours=24):
                continue
            amount = fnum(r["amount"])
            if amount is not None and amount <= 0:
                continue
            if item in enteral_ids:
                stays[stay_id]["enteral_24h"] = 1
            if item in pn_ids:
                stays[stay_id]["parenteral_24h"] = 1


def build_mimic() -> list[dict]:
    stays = mimic_first_older_stays()
    bmi = mimic_bmi_by_subject(stays)
    for v in stays.values():
        v["bmi"] = bmi.get(v["subject_id"])
    add_mimic_nutrition_24h(stays)
    return list(stays.values())


def build_eicu() -> list[dict]:
    best = {}
    for r in rows(EICU / "patient.csv.gz"):
        pid = r.get("uniquepid") or r["patientunitstayid"]
        age = eicu_age(r.get("age", ""))
        if age is None or age < 65:
            continue
        if pid in best:
            continue
        h = fnum(r.get("admissionheight", ""))
        w = fnum(r.get("admissionweight", ""))
        bmi = w / ((h / 100) ** 2) if h and w and 100 <= h <= 230 and 20 <= w <= 300 else None
        best[r["patientunitstayid"]] = {
            "id": r["patientunitstayid"],
            "age": age,
            "male": int(r.get("gender", "").lower() == "male"),
            "bmi": bmi,
            "enteral_24h": 0,
            "parenteral_24h": 0,
            "y": int(r.get("hospitaldischargestatus", "").lower() == "expired"),
        }
    ids = set(best)
    text_specs = [
        ("intakeOutput.csv.gz", "intakeoutputoffset", ["cellpath", "celllabel", "cellvaluetext"]),
        ("treatment.csv.gz", "treatmentoffset", ["treatmentstring"]),
        ("medication.csv.gz", "drugstartoffset", ["drugname", "routeadmin"]),
    ]
    for fname, offset_col, text_cols in text_specs:
        for r in rows(EICU / fname):
            pid = r.get("patientunitstayid", "")
            if pid not in ids:
                continue
            off = fnum(r.get(offset_col, ""))
            if off is None or off < 0 or off > 1440:
                continue
            text = " | ".join(r.get(c, "") for c in text_cols)
            if ENTERAL_RE.search(text):
                best[pid]["enteral_24h"] = 1
            if PN_RE.search(text):
                best[pid]["parenteral_24h"] = 1
    return list(best.values())


def cohort_summary(data: list[dict]) -> dict:
    return {
        "n": len(data),
        "events": int(sum(r["y"] for r in data)),
        "event_rate": round(sum(r["y"] for r in data) / len(data), 4) if data else None,
        "feature_nonmissing": {f: int(sum(r.get(f) is not None for r in data)) for f in FEATURES},
        "feature_positive": {f: int(sum(r.get(f) == 1 for r in data)) for f in ["male", "enteral_24h", "parenteral_24h"]},
    }


def matrix(data: list[dict], impute: dict[str, float] | None = None, scaler: dict[str, tuple[float, float]] | None = None):
    X = []
    y = []
    if impute is None:
        impute = {}
        for f in FEATURES:
            vals = [r[f] for r in data if r.get(f) is not None]
            impute[f] = float(np.median(vals)) if vals else 0.0
    raw = np.array([[r[f] if r.get(f) is not None else impute[f] for f in FEATURES] for r in data], dtype=float)
    if scaler is None:
        scaler = {}
        for j, f in enumerate(FEATURES):
            mu = float(raw[:, j].mean())
            sd = float(raw[:, j].std())
            if sd == 0:
                sd = 1.0
            scaler[f] = (mu, sd)
    X = raw.copy()
    for j, f in enumerate(FEATURES):
        mu, sd = scaler[f]
        X[:, j] = (X[:, j] - mu) / sd
    X = np.column_stack([np.ones(len(data)), X])
    y = np.array([r["y"] for r in data], dtype=float)
    return X, y, impute, scaler


def fit_logistic(X, y, lr=0.05, l2=0.01, epochs=3000):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        grad = X.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict(X, w):
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


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


def subgroup_auc(data, p, key):
    groups = defaultdict(list)
    for i, r in enumerate(data):
        groups[r[key]].append(i)
    out = {}
    y = np.array([r["y"] for r in data], dtype=float)
    for g, idx in groups.items():
        if len(idx) < 500:
            continue
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out[str(g)] = {"n": len(idx), "auc": round(auc(yy, p[idx]), 4), "event_rate": round(float(yy.mean()), 4)}
    return out


def save_patient_level_server_only(name: str, data: list[dict]) -> None:
    path = OUT / name
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "age", "male", "bmi", "enteral_24h", "parenteral_24h", "y"])
        writer.writeheader()
        for r in data:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mimic = build_mimic()
    eicu = build_eicu()
    save_patient_level_server_only("mimic_v0_features_server_only.csv", mimic)
    save_patient_level_server_only("eicu_v0_features_server_only.csv", eicu)

    Xm, ym, impute, scaler = matrix(mimic)
    Xe, ye, _, _ = matrix(eicu, impute=impute, scaler=scaler)
    w = fit_logistic(Xm, ym)
    pm = predict(Xm, w)
    pe = predict(Xe, w)
    res = {
        "features": FEATURES,
        "mimic_summary": cohort_summary(mimic),
        "eicu_summary": cohort_summary(eicu),
        "imputation_from_mimic": impute,
        "standardization_from_mimic": scaler,
        "coefficients": {"intercept": float(w[0]), **{f: float(w[i + 1]) for i, f in enumerate(FEATURES)}},
        "mimic_internal_apparent": {"auc": round(auc(ym, pm), 4), "brier": round(brier(ym, pm), 4), "calibration_bins": calibration_bins(ym, pm)},
        "eicu_external_validation": {"auc": round(auc(ye, pe), 4), "brier": round(brier(ye, pe), 4), "calibration_bins": calibration_bins(ye, pe)},
        "eicu_subgroup_auc_by_male": subgroup_auc(eicu, pe, "male"),
    }
    (OUT / "v0_external_validation_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [
        "# V0 External Validation Results",
        "",
        "Status date: 2026-08-26",
        "",
        "## Model",
        "",
        "Logistic regression trained in MIMIC-IV and externally validated in eICU. Features: " + ", ".join(FEATURES) + ".",
        "",
        "## Cohorts",
        "",
        f"- MIMIC-IV training cohort: n={res['mimic_summary']['n']}, events={res['mimic_summary']['events']}, event_rate={res['mimic_summary']['event_rate']}",
        f"- eICU external cohort: n={res['eicu_summary']['n']}, events={res['eicu_summary']['events']}, event_rate={res['eicu_summary']['event_rate']}",
        "",
        "## Metrics",
        "",
        f"- MIMIC apparent AUROC: {res['mimic_internal_apparent']['auc']}; Brier: {res['mimic_internal_apparent']['brier']}",
        f"- eICU external AUROC: {res['eicu_external_validation']['auc']}; Brier: {res['eicu_external_validation']['brier']}",
        "",
        "## Interpretation",
        "",
        "This is a deliberately minimal transportability smoke test. It demonstrates that a leakage-controlled cross-database pipeline can run, but manuscript-grade performance will require adding first-window laboratory and severity features.",
        "",
        "Patient-level feature CSVs were kept on the server only and should not be copied into the manuscript workspace.",
    ]
    (OUT / "V0_EXTERNAL_VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "mimic_n": res["mimic_summary"]["n"],
        "eicu_n": res["eicu_summary"]["n"],
        "mimic_auc": res["mimic_internal_apparent"]["auc"],
        "eicu_auc": res["eicu_external_validation"]["auc"],
    }, indent=2))


if __name__ == "__main__":
    main()
