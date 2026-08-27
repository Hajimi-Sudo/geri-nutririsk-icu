#!/usr/bin/env python3
"""V3 external validation with organ-support and neurologic severity features."""

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
import run_v2_vitals_external_validation as v2


V2_DIR = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v2_vitals_model")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v3_severity_model")
MIMIC_GCS_SUBSET = OUT / "mimic_target_chartevents_gcs_server_only.csv"

SEVERITY = ["urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h"]
FEATURES = v2.FEATURES + SEVERITY + [f"{x}_missing" for x in SEVERITY]

MIMIC_URINE_ITEMIDS = {"226559", "226560", "226566", "226627", "226631", "226713", "227489"}
MIMIC_VENT_PROC_ITEMIDS = {"225792", "225794"}
MIMIC_DIALYSIS_PROC_ITEMIDS = {"225441", "225802", "225803", "225805", "225809", "225955"}
MIMIC_GCS_ITEMIDS = {
    "220739": "eye",
    "223900": "verbal",
    "223901": "motor",
    "227013": "total",
}


def plain_rows(path: Path, fields: list[str] | None = None):
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def load_feature_table(path: Path) -> list[dict]:
    data = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {"id": row["id"], "y": int(float(row["y"]))}
            for feat in v2.FEATURES:
                val = row.get(feat, "")
                out[feat] = v1.fnum(val) if val != "" else None
            for feat in SEVERITY:
                out[feat] = None
                out[f"{feat}_missing"] = 1
            data.append(out)
    return data


def mimic_stay_intimes() -> dict[str, v1.datetime]:
    out = {}
    for r in v1.rows(v1.MIMIC / "icu/icustays.csv.gz", ["stay_id", "intime"]):
        t = v1.dt(r["intime"])
        if t is not None:
            out[r["stay_id"]] = t
    return out


def valid_urine(value: float | None) -> float | None:
    if value is None:
        return None
    if 0 <= value <= 50000:
        return value
    return None


def add_mimic_urine_24h(data: list[dict], intimes: dict) -> None:
    by_id = {r["id"]: r for r in data}
    urine = defaultdict(float)
    seen = set()
    for r in v1.rows(v1.MIMIC / "icu/outputevents.csv.gz", ["stay_id", "charttime", "itemid", "value"]):
        sid = r["stay_id"]
        if sid not in by_id or r["itemid"] not in MIMIC_URINE_ITEMIDS:
            continue
        t = v1.dt(r["charttime"])
        if t is None or sid not in intimes:
            continue
        delta = t - intimes[sid]
        if delta < timedelta(0) or delta > timedelta(hours=24):
            continue
        val = valid_urine(v1.fnum(r["value"]))
        if val is None:
            continue
        urine[sid] += val
        seen.add(sid)
    for sid, row in by_id.items():
        if sid in seen:
            row["urine_24h_ml"] = urine[sid]
            row["urine_24h_ml_missing"] = 0


def add_mimic_proc_flags_24h(data: list[dict], intimes: dict) -> None:
    by_id = {r["id"]: r for r in data}
    for r in v1.rows(v1.MIMIC / "icu/procedureevents.csv.gz", ["stay_id", "starttime", "itemid"]):
        sid = r["stay_id"]
        if sid not in by_id or sid not in intimes:
            continue
        item = r["itemid"]
        if item not in MIMIC_VENT_PROC_ITEMIDS and item not in MIMIC_DIALYSIS_PROC_ITEMIDS:
            continue
        t = v1.dt(r["starttime"])
        if t is None:
            continue
        delta = t - intimes[sid]
        if delta < timedelta(0) or delta > timedelta(hours=24):
            continue
        if item in MIMIC_VENT_PROC_ITEMIDS:
            by_id[sid]["ventilation_24h"] = 1
            by_id[sid]["ventilation_24h_missing"] = 0
        if item in MIMIC_DIALYSIS_PROC_ITEMIDS:
            by_id[sid]["dialysis_24h"] = 1
            by_id[sid]["dialysis_24h_missing"] = 0
    for row in data:
        for feat in ["ventilation_24h", "dialysis_24h"]:
            if row[feat] is None:
                row[feat] = 0
                row[f"{feat}_missing"] = 0


def ensure_mimic_gcs_subset() -> None:
    if MIMIC_GCS_SUBSET.exists() and MIMIC_GCS_SUBSET.stat().st_size > 100:
        print(f"using existing GCS subset: {MIMIC_GCS_SUBSET}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    itemids = " ".join(sorted(MIMIC_GCS_ITEMIDS))
    cmd = (
        "zcat "
        + str(v1.MIMIC / "icu/chartevents.csv.gz")
        + " | awk -F, 'BEGIN{print \"stay_id,itemid,charttime,valuenum\"; "
        + f"split(\"{itemids}\",a,\" \"); for(i in a) ids[a[i]]=1}} "
        + "NR>1 && ($7 in ids) {print $3\",\"$7\",\"$5\",\"$9}' > "
        + str(MIMIC_GCS_SUBSET)
    )
    print("creating MIMIC GCS subset with zcat/awk", flush=True)
    subprocess.run(["bash", "-lc", cmd], check=True)
    print(f"created GCS subset: {MIMIC_GCS_SUBSET} ({MIMIC_GCS_SUBSET.stat().st_size} bytes)", flush=True)


def add_mimic_gcs_24h(data: list[dict], intimes: dict) -> None:
    ensure_mimic_gcs_subset()
    by_id = {r["id"]: r for r in data}
    parts = {sid: defaultdict(list) for sid in by_id}
    with MIMIC_GCS_SUBSET.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r["stay_id"]
            if sid not in by_id or sid not in intimes:
                continue
            t = v1.dt(r["charttime"])
            if t is None:
                continue
            delta = t - intimes[sid]
            if delta < timedelta(0) or delta > timedelta(hours=24):
                continue
            val = v1.fnum(r["valuenum"])
            if val is None:
                continue
            component = MIMIC_GCS_ITEMIDS.get(r["itemid"])
            if component == "total" and 3 <= val <= 15:
                parts[sid]["total"].append(val)
            elif component == "eye" and 1 <= val <= 4:
                parts[sid]["eye"].append(val)
            elif component == "verbal" and 1 <= val <= 5:
                parts[sid]["verbal"].append(val)
            elif component == "motor" and 1 <= val <= 6:
                parts[sid]["motor"].append(val)
    for sid, row in by_id.items():
        p = parts[sid]
        gcs = None
        if p.get("total"):
            gcs = min(p["total"])
        elif p.get("eye") and p.get("verbal") and p.get("motor"):
            gcs = min(p["eye"]) + min(p["verbal"]) + min(p["motor"])
        if gcs is not None and 3 <= gcs <= 15:
            row["gcs_24h"] = gcs
            row["gcs_24h_missing"] = 0


def add_eicu_aps_features(data: list[dict]) -> None:
    by_id = {r["id"]: r for r in data}
    for r in v1.rows(v1.EICU / "apacheApsVar.csv.gz", ["patientunitstayid", "vent", "dialysis", "urine", "eyes", "motor", "verbal"]):
        sid = r["patientunitstayid"]
        if sid not in by_id:
            continue
        row = by_id[sid]
        vent = v1.fnum(r["vent"])
        dialysis = v1.fnum(r["dialysis"])
        urine = valid_urine(v1.fnum(r["urine"]))
        eyes = v1.fnum(r["eyes"])
        motor = v1.fnum(r["motor"])
        verbal = v1.fnum(r["verbal"])
        if vent is not None:
            row["ventilation_24h"] = int(vent > 0)
            row["ventilation_24h_missing"] = 0
        if dialysis is not None:
            row["dialysis_24h"] = int(dialysis > 0)
            row["dialysis_24h_missing"] = 0
        if urine is not None:
            row["urine_24h_ml"] = urine
            row["urine_24h_ml_missing"] = 0
        if eyes is not None and motor is not None and verbal is not None:
            gcs = eyes + motor + verbal
            if 3 <= gcs <= 15:
                row["gcs_24h"] = gcs
                row["gcs_24h_missing"] = 0
    for row in data:
        for feat in ["ventilation_24h", "dialysis_24h"]:
            if row[feat] is None:
                row[feat] = 0
                row[f"{feat}_missing"] = 0


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
    return np.column_stack([np.ones(len(data)), X]), np.array([r["y"] for r in data], dtype=float), impute, scaler


def cohort_summary(data: list[dict]) -> dict:
    return {
        "n": len(data),
        "events": int(sum(r["y"] for r in data)),
        "event_rate": round(sum(r["y"] for r in data) / len(data), 4),
        "severity_nonmissing": {f: int(sum(r.get(f) is not None for r in data)) for f in SEVERITY},
        "severity_positive": {f: int(sum(r.get(f) == 1 for r in data)) for f in ["ventilation_24h", "dialysis_24h"]},
        "urine_summary": v1.summarize_numeric([r.get("urine_24h_ml") for r in data]),
        "gcs_summary": v1.summarize_numeric([r.get("gcs_24h") for r in data]),
    }


def write_server_only_features(name: str, data: list[dict]) -> None:
    fields = ["id", "y"] + FEATURES
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data:
            writer.writerow({k: r.get(k, "") for k in fields})


def calibration_update(y, p):
    s = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    idx = np.arange(len(y))
    cal = idx % 5 == 0
    test = ~cal
    X = np.column_stack([np.ones(int(cal.sum())), s[cal]])
    w = np.zeros(2)
    for _ in range(5000):
        pred = v1.predict(X, w)
        grad = X.T @ (pred - y[cal]) / len(X)
        grad[1] += 0.001 * w[1]
        w -= 0.03 * grad
    Xtest = np.column_stack([np.ones(int(test.sum())), s[test]])
    pcal = v1.predict(Xtest, w)
    return {
        "split": {"calibration": int(cal.sum()), "heldout_test": int(test.sum())},
        "platt_intercept": float(w[0]),
        "platt_slope": float(w[1]),
        "heldout_original": v1.metric_block(y[test], p[test]),
        "heldout_platt": v1.metric_block(y[test], pcal),
    }


def write_report(res: dict) -> None:
    lines = [
        "# V3 Severity-Enhanced External Validation Results",
        "",
        "Status date: 2026-08-26",
        "",
        "## Model",
        "",
        "V3 extends V2 by adding first-24h organ-support and neurologic severity features: urine output, ventilation, dialysis, and GCS.",
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
        f"- V3 held-out eICU Platt-recalibrated AUROC: {res['eicu_domain_recalibration']['heldout_platt']['auc']}; Brier: {res['eicu_domain_recalibration']['heldout_platt']['brier']}",
        "",
        "## Interpretation",
        "",
        "V3 is the current candidate main model if severity features improve external discrimination without making the nutrition-risk framing disappear. Domain recalibration should be reported as a model-updating analysis.",
    ]
    (OUT / "V3_SEVERITY_EXTERNAL_VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading V2 server-only feature tables", flush=True)
    mimic = load_feature_table(V2_DIR / "mimic_v2_features_server_only.csv")
    eicu = load_feature_table(V2_DIR / "eicu_v2_features_server_only.csv")
    print(f"loaded MIMIC={len(mimic)} eICU={len(eicu)}", flush=True)
    intimes = mimic_stay_intimes()
    print("adding MIMIC urine/procedure/GCS severity features", flush=True)
    add_mimic_urine_24h(mimic, intimes)
    add_mimic_proc_flags_24h(mimic, intimes)
    add_mimic_gcs_24h(mimic, intimes)
    print("adding eICU APACHE severity features", flush=True)
    add_eicu_aps_features(eicu)
    write_server_only_features("mimic_v3_features_server_only.csv", mimic)
    write_server_only_features("eicu_v3_features_server_only.csv", eicu)
    Xm, ym, impute, scaler = matrix(mimic)
    Xe, ye, _, _ = matrix(eicu, impute=impute, scaler=scaler)
    w = v1.fit_logistic(Xm, ym, lr=0.03, l2=0.02, epochs=5000)
    pm = v1.predict(Xm, w)
    pe = v1.predict(Xe, w)
    res = {
        "features": FEATURES,
        "mimic_summary": cohort_summary(mimic),
        "eicu_summary": cohort_summary(eicu),
        "coefficients": {"intercept": float(w[0]), **{f: float(w[i + 1]) for i, f in enumerate(FEATURES)}},
        "imputation_from_mimic": impute,
        "standardization_from_mimic": scaler,
        "mimic_internal_apparent": v1.metric_block(ym, pm),
        "eicu_external_validation": v1.metric_block(ye, pe),
        "eicu_domain_recalibration": calibration_update(ye, pe),
    }
    (OUT / "v3_severity_external_validation_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res)
    print(json.dumps({
        "out": str(OUT),
        "mimic_auc": res["mimic_internal_apparent"]["auc"],
        "eicu_auc": res["eicu_external_validation"]["auc"],
        "eicu_brier": res["eicu_external_validation"]["brier"],
        "eicu_platt_auc": res["eicu_domain_recalibration"]["heldout_platt"]["auc"],
        "eicu_platt_brier": res["eicu_domain_recalibration"]["heldout_platt"]["brier"],
    }, indent=2))


if __name__ == "__main__":
    main()
