#!/usr/bin/env python3
"""Build v0 aggregate cohort and feature-mapping report.

No patient-level output is copied out of the server. The script writes aggregate
counts and a candidate shared feature dictionary for the MIMIC-IV/eICU project.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v0_mapping")


LAB_CONCEPTS = {
    "albumin": re.compile(r"\balbumin\b", re.I),
    "glucose": re.compile(r"\bglucose\b", re.I),
    "creatinine": re.compile(r"\bcreatinine\b", re.I),
    "bun": re.compile(r"\burea nitrogen\b|\bbun\b", re.I),
    "sodium": re.compile(r"\bsodium\b", re.I),
    "bilirubin": re.compile(r"\bbilirubin\b", re.I),
    "hematocrit": re.compile(r"\bhematocrit\b", re.I),
    "wbc": re.compile(r"\bwhite blood cells\b|\bwbc\b", re.I),
    "prealbumin": re.compile(r"\bprealbumin\b", re.I),
}

MIMIC_ITEM_TERMS = {
    "enteral_nutrition": re.compile(r"enteral|tube feed|tube feeding|feeding tube|nutren|jevity|promote|nepro|osmolite|glucerna|twocal|vital", re.I),
    "parenteral_nutrition": re.compile(r"parenteral|tpn|total parenteral|lipid|amino acid", re.I),
    "weight": re.compile(r"\bweight\b|admission weight|daily weight", re.I),
    "height": re.compile(r"\bheight\b", re.I),
}

EICU_TEXT_TERMS = {
    "enteral_nutrition": re.compile(r"enteral|tube feed|tube feeding|feeding tube|jevity|promote|nepro|osmolite|glucerna|twocal|vital", re.I),
    "parenteral_nutrition": re.compile(r"parenteral|tpn|total parenteral|lipid|amino acid", re.I),
    "diet_or_oral_intake": re.compile(r"\bdiet\b|oral intake|nutrition|feeding", re.I),
    "dysphagia": re.compile(r"dysphagia|swallow", re.I),
    "malnutrition": re.compile(r"malnutrition|cachexia|undernutrition", re.I),
}


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", errors="replace")


def dict_rows(path: Path, fields: list[str] | None = None):
    with open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def to_float(x: str) -> float | None:
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def age_eicu(x: str) -> int | None:
    x = (x or "").strip()
    if not x:
        return None
    if x.startswith(">"):
        return 90
    try:
        return int(float(x))
    except Exception:
        return None


def summarize_numeric(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return round(vals[idx], 3)
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "min": round(vals[0], 3),
        "max": round(vals[-1], 3),
    }


def mimic_base() -> dict:
    ages = {}
    genders = {}
    for r in dict_rows(MIMIC / "hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age"]):
        age = to_float(r["anchor_age"])
        if age is not None:
            ages[r["subject_id"]] = age
        genders[r["subject_id"]] = r["gender"]

    admissions = {}
    for r in dict_rows(MIMIC / "hosp/admissions.csv.gz", ["hadm_id", "hospital_expire_flag", "race", "insurance"]):
        admissions[r["hadm_id"]] = r

    rows = []
    first_seen = set()
    for r in dict_rows(MIMIC / "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "first_careunit", "los"]):
        sid = r["subject_id"]
        if sid in first_seen:
            continue
        age = ages.get(sid)
        if age is None or age < 65:
            continue
        first_seen.add(sid)
        adm = admissions.get(r["hadm_id"], {})
        rows.append({
            "subject_id": sid,
            "stay_id": r["stay_id"],
            "hadm_id": r["hadm_id"],
            "age": age,
            "gender": genders.get(sid, ""),
            "careunit": r["first_careunit"],
            "los_days": to_float(r["los"]),
            "hospital_mortality": int(adm.get("hospital_expire_flag") == "1"),
            "race": adm.get("race", ""),
            "insurance": adm.get("insurance", ""),
        })
    return {
        "n_first_older_icu": len(rows),
        "age": summarize_numeric([r["age"] for r in rows]),
        "los_days": summarize_numeric([r["los_days"] for r in rows if r["los_days"] is not None]),
        "hospital_mortality_events": sum(r["hospital_mortality"] for r in rows),
        "hospital_mortality_rate": round(sum(r["hospital_mortality"] for r in rows) / len(rows), 4) if rows else None,
        "gender": Counter(r["gender"] for r in rows),
        "careunit_top": Counter(r["careunit"] for r in rows).most_common(10),
        "race_top": Counter(r["race"] for r in rows).most_common(12),
        "insurance": Counter(r["insurance"] for r in rows),
    }


def eicu_base() -> dict:
    rows = []
    seen_pid = set()
    for r in dict_rows(EICU / "patient.csv.gz"):
        pid = r.get("uniquepid") or r["patientunitstayid"]
        if pid in seen_pid:
            continue
        age = age_eicu(r.get("age", ""))
        if age is None or age < 65:
            continue
        seen_pid.add(pid)
        h = to_float(r.get("admissionheight", ""))
        w = to_float(r.get("admissionweight", ""))
        bmi = w / ((h / 100) ** 2) if w and h and 80 <= h <= 230 and 20 <= w <= 300 else None
        rows.append({
            "age": age,
            "gender": r.get("gender", ""),
            "ethnicity": r.get("ethnicity", ""),
            "unit_type": r.get("unittype", ""),
            "height_cm": h,
            "weight_kg": w,
            "bmi": bmi,
            "hospital_mortality": int(r.get("hospitaldischargestatus", "").lower() == "expired"),
            "icu_mortality": int(r.get("unitdischargestatus", "").lower() == "expired"),
        })
    return {
        "n_first_older_unit_stays_by_uniquepid": len(rows),
        "age": summarize_numeric([r["age"] for r in rows]),
        "height_cm": summarize_numeric([r["height_cm"] for r in rows if r["height_cm"] is not None]),
        "weight_kg": summarize_numeric([r["weight_kg"] for r in rows if r["weight_kg"] is not None]),
        "bmi": summarize_numeric([r["bmi"] for r in rows if r["bmi"] is not None]),
        "hospital_mortality_events": sum(r["hospital_mortality"] for r in rows),
        "hospital_mortality_rate": round(sum(r["hospital_mortality"] for r in rows) / len(rows), 4) if rows else None,
        "icu_mortality_events": sum(r["icu_mortality"] for r in rows),
        "icu_mortality_rate": round(sum(r["icu_mortality"] for r in rows) / len(rows), 4) if rows else None,
        "gender": Counter(r["gender"] for r in rows),
        "ethnicity_top": Counter(r["ethnicity"] for r in rows).most_common(12),
        "unit_type_top": Counter(r["unit_type"] for r in rows).most_common(10),
    }


def mimic_lab_itemids() -> dict:
    concepts = defaultdict(list)
    for r in dict_rows(MIMIC / "hosp/d_labitems.csv.gz"):
        label = r.get("label", "")
        for name, pat in LAB_CONCEPTS.items():
            if pat.search(label):
                concepts[name].append(r)
    return dict(concepts)


def mimic_icu_itemids() -> dict:
    concepts = defaultdict(list)
    for r in dict_rows(MIMIC / "icu/d_items.csv.gz"):
        text = " | ".join(str(v) for v in r.values())
        for name, pat in MIMIC_ITEM_TERMS.items():
            if pat.search(text):
                concepts[name].append(r)
    return dict(concepts)


def eicu_lab_concepts() -> dict:
    found = defaultdict(Counter)
    for r in dict_rows(EICU / "lab.csv.gz", ["labname"]):
        lab = r["labname"]
        for name, pat in LAB_CONCEPTS.items():
            if pat.search(lab):
                found[name][lab] += 1
    return {k: v.most_common(12) for k, v in found.items()}


def eicu_text_signal_counts(max_rows_per_table: int = 1000000) -> dict:
    result = {}
    for fname, text_fields in {
        "intakeOutput.csv.gz": ["cellpath", "celllabel", "cellvaluetext"],
        "treatment.csv.gz": ["treatmentstring"],
        "diagnosis.csv.gz": ["diagnosisstring", "icd9code"],
        "nurseCharting.csv.gz": ["nursingchartcelltypecat", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"],
    }.items():
        counts = Counter()
        rows_scanned = 0
        for r in dict_rows(EICU / fname, text_fields):
            rows_scanned += 1
            text = " | ".join(r.values())
            for name, pat in EICU_TEXT_TERMS.items():
                if pat.search(text):
                    counts[name] += 1
            if rows_scanned >= max_rows_per_table:
                break
        result[fname] = {"rows_scanned": rows_scanned, "signal_counts": dict(counts)}
    return result


def shared_feature_dictionary() -> list[dict]:
    return [
        {"domain": "demographics", "feature": "age", "mimic_source": "hosp/patients.anchor_age", "eicu_source": "patient.age", "status": "shared"},
        {"domain": "demographics", "feature": "sex/gender", "mimic_source": "hosp/patients.gender", "eicu_source": "patient.gender", "status": "shared"},
        {"domain": "anthropometrics", "feature": "height", "mimic_source": "hosp/omr result_name/item labels; requires extraction", "eicu_source": "patient.admissionheight", "status": "shared_with_mapping_work"},
        {"domain": "anthropometrics", "feature": "weight", "mimic_source": "icu d_items/input/chartevents or hosp/omr; requires extraction", "eicu_source": "patient.admissionweight", "status": "shared_with_mapping_work"},
        {"domain": "nutrition/metabolic labs", "feature": "albumin", "mimic_source": "hosp/d_labitems + labevents", "eicu_source": "apacheApsVar.albumin and lab.labname", "status": "shared"},
        {"domain": "nutrition/metabolic labs", "feature": "glucose", "mimic_source": "hosp/d_labitems + labevents", "eicu_source": "apacheApsVar.glucose and lab.labname", "status": "shared"},
        {"domain": "renal/metabolic", "feature": "creatinine", "mimic_source": "hosp/d_labitems + labevents", "eicu_source": "apacheApsVar.creatinine and lab.labname", "status": "shared"},
        {"domain": "renal/metabolic", "feature": "BUN", "mimic_source": "hosp/d_labitems + labevents", "eicu_source": "apacheApsVar.bun and lab.labname", "status": "shared"},
        {"domain": "severity", "feature": "ventilation", "mimic_source": "procedureevents/chartevents; requires extraction", "eicu_source": "apacheApsVar.vent", "status": "shared_with_mapping_work"},
        {"domain": "severity", "feature": "dialysis/CRRT", "mimic_source": "procedures/inputevents/diagnoses; requires extraction", "eicu_source": "apacheApsVar.dialysis", "status": "shared_with_mapping_work"},
        {"domain": "outcome", "feature": "hospital mortality", "mimic_source": "hosp/admissions.hospital_expire_flag", "eicu_source": "patient.hospitaldischargestatus or apachePatientResult.actualhospitalmortality", "status": "shared"},
        {"domain": "nutrition support", "feature": "enteral/parenteral nutrition signal", "mimic_source": "icu/inputevents and ingredientevents itemids; requires itemid extraction", "eicu_source": "intakeOutput/treatment/medication/nurseCharting text signals", "status": "candidate_shared_proxy"},
    ]


def write_markdown(res: dict) -> None:
    lines = [
        "# V0 Cohort and Cross-Database Mapping Audit",
        "",
        "Status date: 2026-08-26",
        "",
        "## Cohort Summary",
        "",
        "### MIMIC-IV Development Cohort Candidate",
        "",
    ]
    for k, v in res["mimic_base"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "### eICU External Validation Cohort Candidate", ""]
    for k, v in res["eicu_base"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Candidate Shared Feature Domains",
        "",
        "| Domain | Feature | MIMIC-IV source | eICU source | Status |",
        "|---|---|---|---|---|",
    ]
    for row in res["shared_feature_dictionary"]:
        lines.append(
            f"| {row['domain']} | {row['feature']} | {row['mimic_source']} | {row['eicu_source']} | {row['status']} |"
        )
    lines += [
        "",
        "## Nutrition-Specific Feasibility Signals",
        "",
        f"- MIMIC lab concepts found: {', '.join(sorted(res['mimic_lab_itemids']))}",
        f"- MIMIC ICU item concepts found: {', '.join(sorted(res['mimic_icu_itemids']))}",
        f"- eICU lab concepts found: {', '.join(sorted(res['eicu_lab_concepts']))}",
        "- eICU text signal counts are stored in `v0_cohort_mapping_full.json`.",
        "",
        "## V0 Decision",
        "",
        "Proceed to cohort construction with hospital mortality as the first robust endpoint and a parallel nutrition-support proxy audit.",
        "",
        "The target Frontiers in Nutrition framing should emphasize a transportable nutrition-risk framework rather than a generic ICU mortality model.",
    ]
    (OUT / "V0_COHORT_MAPPING_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "mimic_base": mimic_base(),
        "eicu_base": eicu_base(),
        "mimic_lab_itemids": mimic_lab_itemids(),
        "mimic_icu_itemids": mimic_icu_itemids(),
        "eicu_lab_concepts": eicu_lab_concepts(),
        "eicu_text_signal_counts": eicu_text_signal_counts(),
        "shared_feature_dictionary": shared_feature_dictionary(),
    }
    (OUT / "v0_cohort_mapping_full.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(res)
    print(json.dumps({
        "out": str(OUT),
        "mimic_first_older_icu": res["mimic_base"]["n_first_older_icu"],
        "mimic_hospital_mortality_rate": res["mimic_base"]["hospital_mortality_rate"],
        "eicu_first_older_unit_stays": res["eicu_base"]["n_first_older_unit_stays_by_uniquepid"],
        "eicu_hospital_mortality_rate": res["eicu_base"]["hospital_mortality_rate"],
        "shared_feature_count": len(res["shared_feature_dictionary"]),
    }, indent=2))


if __name__ == "__main__":
    main()
