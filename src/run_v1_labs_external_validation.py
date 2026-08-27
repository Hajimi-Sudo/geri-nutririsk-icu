#!/usr/bin/env python3
"""V1 MIMIC-IV to eICU external validation with early laboratory features.

The V1 experiment extends the V0 transportability smoke test by adding
nutrition/metabolic and severity-adjacent laboratory features available in both
MIMIC-IV and eICU. Patient-level outputs remain on the server.
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

import numpy as np


MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v1_labs_model")
LAB_SUBSET = OUT / "mimic_target_labevents_server_only.csv"

LABS = ["albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc"]
BASE_FEATURES = ["age", "male", "bmi", "enteral_24h", "parenteral_24h"]
FEATURES = BASE_FEATURES + LABS + [f"{x}_missing" for x in LABS]

ENTERAL_RE = re.compile(r"enteral|tube feed|tube feeding|feeding tube|jevity|promote|nepro|osmolite|glucerna|twocal|vital|nutren", re.I)
PN_RE = re.compile(r"parenteral|tpn|total parenteral|lipid|amino acid", re.I)

MIMIC_LAB_ITEMIDS = {
    "albumin": {"50862"},
    "glucose": {"50809", "50931"},
    "creatinine": {"50912"},
    "bun": {"51006", "52647"},
    "sodium": {"50824", "50983"},
    "bilirubin": {"50885"},
    "hematocrit": {"50810", "51221", "52028"},
    "wbc": {"51300", "51301", "51755", "51756"},
}

VALID_RANGES = {
    "albumin": (0.5, 8.0),
    "glucose": (20.0, 1000.0),
    "creatinine": (0.1, 30.0),
    "bun": (1.0, 300.0),
    "sodium": (90.0, 190.0),
    "bilirubin": (0.0, 80.0),
    "hematocrit": (5.0, 75.0),
    "wbc": (0.1, 200.0),
    "bmi": (8.0, 100.0),
}


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


def plain_rows(path: Path, fields: list[str] | None = None):
    with path.open("rt", newline="", errors="replace") as f:
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


def valid(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    lo, hi = VALID_RANGES.get(name, (-math.inf, math.inf))
    if lo <= value <= hi:
        return value
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


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def summarize_numeric(values: list[float | None]) -> dict:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return round(vals[idx], 4)
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
    }


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
        if "bmi" in name:
            val = valid("bmi", val)
            if val is not None:
                bmi_by_subject.setdefault(sid, val)
        elif "weight" in name:
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


def add_mimic_labs_24h(stays: dict[str, dict]) -> None:
    ensure_mimic_lab_subset()
    hadm_to_stays = defaultdict(list)
    for stay_id, s in stays.items():
        hadm_to_stays[s["hadm_id"]].append(stay_id)
    item_to_lab = {itemid: lab for lab, ids in MIMIC_LAB_ITEMIDS.items() for itemid in ids}
    values = {stay_id: defaultdict(list) for stay_id in stays}
    for r in plain_rows(LAB_SUBSET, ["hadm_id", "itemid", "charttime", "valuenum"]):
        candidates = hadm_to_stays.get(r["hadm_id"])
        if not candidates:
            continue
        lab = item_to_lab.get(r["itemid"])
        if lab is None:
            continue
        val = valid(lab, fnum(r["valuenum"]))
        if val is None:
            continue
        t = dt(r["charttime"])
        if t is None:
            continue
        for stay_id in candidates:
            delta = t - stays[stay_id]["intime"]
            if timedelta(0) <= delta <= timedelta(hours=24):
                values[stay_id][lab].append(val)
    for stay_id, s in stays.items():
        for lab in LABS:
            s[lab] = median(values[stay_id].get(lab, []))
            s[f"{lab}_missing"] = int(s[lab] is None)


def ensure_mimic_lab_subset() -> None:
    if LAB_SUBSET.exists() and LAB_SUBSET.stat().st_size > 100:
        print(f"using existing lab subset: {LAB_SUBSET}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    itemids = sorted({itemid for ids in MIMIC_LAB_ITEMIDS.values() for itemid in ids})
    awk_items = " ".join(itemids)
    cmd = (
        "zcat "
        + str(MIMIC / "hosp/labevents.csv.gz")
        + " | awk -F, 'BEGIN{print \"hadm_id,itemid,charttime,valuenum\"; "
        + f"split(\"{awk_items}\",a,\" \"); for(i in a) ids[a[i]]=1}} "
        + "NR>1 && ($5 in ids) {print $3\",\"$5\",\"$7\",\"$10}' > "
        + str(LAB_SUBSET)
    )
    print("creating MIMIC target lab subset with zcat/awk", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=True)
    print(f"created lab subset: {LAB_SUBSET} ({LAB_SUBSET.stat().st_size} bytes)", flush=True)


def build_mimic() -> list[dict]:
    print("building MIMIC base cohort", flush=True)
    stays = mimic_first_older_stays()
    print(f"MIMIC base stays: {len(stays)}", flush=True)
    bmi = mimic_bmi_by_subject(stays)
    for v in stays.values():
        v["bmi"] = bmi.get(v["subject_id"])
    print("adding MIMIC nutrition support signals", flush=True)
    add_mimic_nutrition_24h(stays)
    print("adding MIMIC 24h laboratory features", flush=True)
    add_mimic_labs_24h(stays)
    return list(stays.values())


def build_eicu() -> list[dict]:
    print("building eICU cohort and nutrition/lab features", flush=True)
    best_by_pid = {}
    for r in rows(EICU / "patient.csv.gz"):
        pid = r.get("uniquepid") or r["patientunitstayid"]
        age = eicu_age(r.get("age", ""))
        if age is None or age < 65:
            continue
        old = best_by_pid.get(pid)
        if old is not None:
            old_offset = fnum(old.get("_hospitaladmitoffset", ""))
            new_offset = fnum(r.get("hospitaladmitoffset", ""))
            if old_offset is not None and new_offset is not None and old_offset <= new_offset:
                continue
        h = fnum(r.get("admissionheight", ""))
        w = fnum(r.get("admissionweight", ""))
        bmi = w / ((h / 100) ** 2) if h and w and 100 <= h <= 230 and 20 <= w <= 300 else None
        best_by_pid[pid] = {
            "_hospitaladmitoffset": r.get("hospitaladmitoffset", ""),
            "id": r["patientunitstayid"],
            "age": age,
            "male": int(r.get("gender", "").lower() == "male"),
            "bmi": bmi,
            "enteral_24h": 0,
            "parenteral_24h": 0,
            "y": int(r.get("hospitaldischargestatus", "").lower() == "expired"),
        }
    by_stay = {r["id"]: r for r in best_by_pid.values()}
    ids = set(by_stay)
    for fname, offset_col, text_cols in [
        ("intakeOutput.csv.gz", "intakeoutputoffset", ["cellpath", "celllabel", "cellvaluetext"]),
        ("treatment.csv.gz", "treatmentoffset", ["treatmentstring"]),
        ("medication.csv.gz", "drugstartoffset", ["drugname", "routeadmin"]),
    ]:
        for r in rows(EICU / fname):
            pid = r.get("patientunitstayid", "")
            if pid not in ids:
                continue
            off = fnum(r.get(offset_col, ""))
            if off is None or off < 0 or off > 1440:
                continue
            text = " | ".join(r.get(c, "") for c in text_cols)
            if ENTERAL_RE.search(text):
                by_stay[pid]["enteral_24h"] = 1
            if PN_RE.search(text):
                by_stay[pid]["parenteral_24h"] = 1
    for r in rows(EICU / "apacheApsVar.csv.gz"):
        pid = r.get("patientunitstayid", "")
        if pid not in ids:
            continue
        target = by_stay[pid]
        mapping = {
            "albumin": "albumin",
            "glucose": "glucose",
            "creatinine": "creatinine",
            "bun": "bun",
            "sodium": "sodium",
            "bilirubin": "bilirubin",
            "hematocrit": "hematocrit",
            "wbc": "wbc",
        }
        for lab, col in mapping.items():
            val = valid(lab, fnum(r.get(col, "")))
            if val is not None and target.get(lab) is None:
                target[lab] = val
    for s in by_stay.values():
        s.pop("_hospitaladmitoffset", None)
        for lab in LABS:
            s.setdefault(lab, None)
            s[f"{lab}_missing"] = int(s[lab] is None)
    return list(by_stay.values())


def cohort_summary(data: list[dict]) -> dict:
    return {
        "n": len(data),
        "events": int(sum(r["y"] for r in data)),
        "event_rate": round(sum(r["y"] for r in data) / len(data), 4) if data else None,
        "feature_nonmissing": {f: int(sum(r.get(f) is not None for r in data)) for f in BASE_FEATURES + LABS},
        "feature_positive": {f: int(sum(r.get(f) == 1 for r in data)) for f in ["male", "enteral_24h", "parenteral_24h"]},
        "lab_summaries": {lab: summarize_numeric([r.get(lab) for r in data]) for lab in LABS},
    }


def matrix(data: list[dict], impute=None, scaler=None):
    if impute is None:
        impute = {}
        for f in FEATURES:
            vals = [r.get(f) for r in data if r.get(f) is not None]
            impute[f] = float(np.median(vals)) if vals else 0.0
    raw = np.array([[r.get(f) if r.get(f) is not None else impute[f] for f in FEATURES] for r in data], dtype=float)
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


def fit_logistic(X, y, lr=0.03, l2=0.02, epochs=5000):
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


def metric_block(y, p):
    return {
        "auc": round(auc(y, p), 4),
        "brier": round(brier(y, p), 4),
        "calibration_bins": calibration_bins(y, p),
    }


def subgroup_auc(data, p, key):
    groups = defaultdict(list)
    for i, r in enumerate(data):
        groups[r.get(key)].append(i)
    y = np.array([r["y"] for r in data], dtype=float)
    out = {}
    for g, idx in groups.items():
        if len(idx) < 500:
            continue
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out[str(g)] = {
            "n": len(idx),
            "auc": round(auc(yy, p[idx]), 4),
            "event_rate": round(float(yy.mean()), 4),
        }
    return out


def write_server_only_features(name: str, data: list[dict]) -> None:
    path = OUT / name
    fields = ["id", "y"] + FEATURES
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data:
            writer.writerow({k: r.get(k, "") for k in fields})


def write_report(res: dict) -> None:
    lines = [
        "# V1 Laboratory-Enhanced External Validation Results",
        "",
        "Status date: 2026-08-26",
        "",
        "## Model",
        "",
        "MIMIC-IV trained logistic regression with early nutrition/metabolic laboratory features and external validation in eICU.",
        "",
        "Features: " + ", ".join(FEATURES) + ".",
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
        "## Feature Availability",
        "",
        "### MIMIC-IV non-missing counts",
        "",
    ]
    for k, v in res["mimic_summary"]["feature_nonmissing"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "### eICU non-missing counts", ""]
    for k, v in res["eicu_summary"]["feature_nonmissing"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Interpretation",
        "",
        "V1 tests whether adding shared laboratory markers improves transportability beyond the V0 demographic/anthropometric/nutrition-support smoke test. Patient-level feature files remain on the server only.",
    ]
    (OUT / "V1_LABS_EXTERNAL_VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mimic = build_mimic()
    print("MIMIC complete", flush=True)
    eicu = build_eicu()
    print("eICU complete", flush=True)
    write_server_only_features("mimic_v1_features_server_only.csv", mimic)
    write_server_only_features("eicu_v1_features_server_only.csv", eicu)

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
        "mimic_internal_apparent": metric_block(ym, pm),
        "eicu_external_validation": metric_block(ye, pe),
        "eicu_subgroup_auc_by_male": subgroup_auc(eicu, pe, "male"),
    }
    (OUT / "v1_labs_external_validation_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res)
    print(json.dumps({
        "out": str(OUT),
        "mimic_n": res["mimic_summary"]["n"],
        "eicu_n": res["eicu_summary"]["n"],
        "mimic_auc": res["mimic_internal_apparent"]["auc"],
        "eicu_auc": res["eicu_external_validation"]["auc"],
        "eicu_brier": res["eicu_external_validation"]["brier"],
    }, indent=2))


if __name__ == "__main__":
    main()
