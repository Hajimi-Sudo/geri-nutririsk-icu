#!/usr/bin/env python3
"""V2 external validation with nutrition/metabolic labs plus early vital signs."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

import run_v1_labs_external_validation as v1


OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v2_vitals_model")
MIMIC_VITAL_SUBSET = OUT / "mimic_target_chartevents_vitals_server_only.csv"
EICU_VITAL_SUBSET = OUT / "eicu_vitalperiodic_24h_server_only.csv"

VITALS = ["heart_rate", "resp_rate", "temperature_c", "spo2", "map"]
FEATURES = v1.FEATURES + VITALS + [f"{x}_missing" for x in VITALS]

MIMIC_VITAL_ITEMIDS = {
    "heart_rate": {"220045"},
    "resp_rate": {"220210"},
    "temperature_c": {"223761", "223762"},
    "spo2": {"220277"},
    "map": {"220052", "220181"},
}

VITAL_RANGES = {
    "heart_rate": (20, 250),
    "resp_rate": (3, 80),
    "temperature_c": (25, 45),
    "spo2": (30, 100),
    "map": (20, 200),
}


def valid_vital(name: str, value: float | None, itemid: str | None = None) -> float | None:
    if value is None:
        return None
    if name == "temperature_c" and itemid == "223761":
        value = (value - 32) * 5 / 9
    lo, hi = VITAL_RANGES[name]
    if lo <= value <= hi:
        return value
    return None


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2


def ensure_mimic_vital_subset() -> None:
    if MIMIC_VITAL_SUBSET.exists() and MIMIC_VITAL_SUBSET.stat().st_size > 100:
        print(f"using existing vital subset: {MIMIC_VITAL_SUBSET}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    itemids = sorted({x for ids in MIMIC_VITAL_ITEMIDS.values() for x in ids})
    awk_items = " ".join(itemids)
    cmd = (
        "zcat "
        + str(v1.MIMIC / "icu/chartevents.csv.gz")
        + " | awk -F, 'BEGIN{print \"stay_id,itemid,charttime,valuenum\"; "
        + f"split(\"{awk_items}\",a,\" \"); for(i in a) ids[a[i]]=1}} "
        + "NR>1 && ($7 in ids) {print $3\",\"$7\",\"$5\",\"$9}' > "
        + str(MIMIC_VITAL_SUBSET)
    )
    print("creating MIMIC target vital subset with zcat/awk", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=True)
    print(f"created vital subset: {MIMIC_VITAL_SUBSET} ({MIMIC_VITAL_SUBSET.stat().st_size} bytes)", flush=True)


def ensure_eicu_vital_subset() -> None:
    if EICU_VITAL_SUBSET.exists() and EICU_VITAL_SUBSET.stat().st_size > 100:
        print(f"using existing eICU vital subset: {EICU_VITAL_SUBSET}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = (
        "zcat "
        + str(v1.EICU / "vitalPeriodic.csv.gz")
        + " | awk -F, 'BEGIN{print \"patientunitstayid,observationoffset,temperature,sao2,heartrate,respiration,systemicmean\"} "
        + "NR>1 && $3 >= 0 && $3 <= 1440 {print $2\",\"$3\",\"$4\",\"$5\",\"$6\",\"$7\",\"$12}' > "
        + str(EICU_VITAL_SUBSET)
    )
    print("creating eICU 24h vital subset with zcat/awk", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=True)
    print(f"created eICU vital subset: {EICU_VITAL_SUBSET} ({EICU_VITAL_SUBSET.stat().st_size} bytes)", flush=True)


def add_mimic_vitals_24h(data: list[dict]) -> None:
    ensure_mimic_vital_subset()
    stays = {r["id"]: r for r in data}
    item_to_vital = {itemid: vital for vital, ids in MIMIC_VITAL_ITEMIDS.items() for itemid in ids}
    values = {sid: defaultdict(list) for sid in stays}
    with MIMIC_VITAL_SUBSET.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay_id = row["stay_id"]
            if stay_id not in stays:
                continue
            vital = item_to_vital.get(row["itemid"])
            if vital is None:
                continue
            value = valid_vital(vital, v1.fnum(row["valuenum"]), row["itemid"])
            if value is None:
                continue
            t = v1.dt(row["charttime"])
            if t is None:
                continue
            delta = t - stays[stay_id]["intime"]
            if timedelta(0) <= delta <= timedelta(hours=24):
                values[stay_id][vital].append(value)
    for row in data:
        for vital in VITALS:
            row[vital] = median(values[row["id"]].get(vital, []))
            row[f"{vital}_missing"] = int(row[vital] is None)


def add_eicu_vitals_24h(data: list[dict]) -> None:
    ensure_eicu_vital_subset()
    by_id = {r["id"]: r for r in data}
    values = {pid: defaultdict(list) for pid in by_id}
    with EICU_VITAL_SUBSET.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patientunitstayid"]
            if pid not in by_id:
                continue
            mapping = {
                "heart_rate": row["heartrate"],
                "resp_rate": row["respiration"],
                "temperature_c": row["temperature"],
                "spo2": row["sao2"],
                "map": row["systemicmean"],
            }
            for vital, raw in mapping.items():
                value = valid_vital(vital, v1.fnum(raw))
                if value is not None:
                    values[pid][vital].append(value)
    for row in data:
        for vital in VITALS:
            row[vital] = median(values[row["id"]].get(vital, []))
            row[f"{vital}_missing"] = int(row[vital] is None)


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


def cohort_summary(data: list[dict]) -> dict:
    feature_base = v1.BASE_FEATURES + v1.LABS + VITALS
    return {
        "n": len(data),
        "events": int(sum(r["y"] for r in data)),
        "event_rate": round(sum(r["y"] for r in data) / len(data), 4) if data else None,
        "feature_nonmissing": {f: int(sum(r.get(f) is not None for r in data)) for f in feature_base},
        "feature_positive": {f: int(sum(r.get(f) == 1 for r in data)) for f in ["male", "enteral_24h", "parenteral_24h"]},
        "vital_summaries": {x: v1.summarize_numeric([r.get(x) for r in data]) for x in VITALS},
    }


def write_server_only_features(name: str, data: list[dict]) -> None:
    fields = ["id", "y"] + FEATURES
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_report(res: dict) -> None:
    lines = [
        "# V2 Vital-Enhanced External Validation Results",
        "",
        "Status date: 2026-08-26",
        "",
        "## Model",
        "",
        "MIMIC-IV trained logistic regression with early nutrition/metabolic laboratory features plus first-24h vital signs; externally validated in eICU.",
        "",
        "Feature families: demographics, BMI, early enteral/parenteral nutrition support signals, albumin/glucose/renal/metabolic labs, and vital signs.",
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
        "V2 evaluates whether a cross-database nutrition-risk framework gains transportable discrimination after adding shared physiological severity signals. Patient-level files remain on the server only.",
    ]
    (OUT / "V2_VITALS_EXTERNAL_VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("building V1 MIMIC features", flush=True)
    mimic = v1.build_mimic()
    print("adding MIMIC vitals", flush=True)
    add_mimic_vitals_24h(mimic)
    print("building V1 eICU features", flush=True)
    eicu = v1.build_eicu()
    print("adding eICU vitals", flush=True)
    add_eicu_vitals_24h(eicu)
    write_server_only_features("mimic_v2_features_server_only.csv", mimic)
    write_server_only_features("eicu_v2_features_server_only.csv", eicu)

    Xm, ym, impute, scaler = matrix(mimic)
    Xe, ye, _, _ = matrix(eicu, impute=impute, scaler=scaler)
    w = v1.fit_logistic(Xm, ym, lr=0.03, l2=0.02, epochs=5000)
    pm = v1.predict(Xm, w)
    pe = v1.predict(Xe, w)
    res = {
        "features": FEATURES,
        "mimic_summary": cohort_summary(mimic),
        "eicu_summary": cohort_summary(eicu),
        "imputation_from_mimic": impute,
        "standardization_from_mimic": scaler,
        "coefficients": {"intercept": float(w[0]), **{f: float(w[i + 1]) for i, f in enumerate(FEATURES)}},
        "mimic_internal_apparent": v1.metric_block(ym, pm),
        "eicu_external_validation": v1.metric_block(ye, pe),
        "eicu_subgroup_auc_by_male": v1.subgroup_auc(eicu, pe, "male"),
    }
    (OUT / "v2_vitals_external_validation_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
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
