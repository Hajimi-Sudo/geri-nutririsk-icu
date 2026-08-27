#!/usr/bin/env python3
"""Gate-0 inventory for MIMIC-IV + eICU geriatric nutrition-risk project.

This script uses only the Python standard library so it can run on a minimal
server environment. It writes aggregate-only audit reports and does not export
patient-level records.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/gate0")

CORE_MIMIC = {
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/omr.csv.gz",
    "icu/icustays.csv.gz",
    "icu/d_items.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/ingredientevents.csv.gz",
    "icu/outputevents.csv.gz",
}

CORE_EICU = {
    "patient.csv.gz",
    "diagnosis.csv.gz",
    "lab.csv.gz",
    "intakeOutput.csv.gz",
    "medication.csv.gz",
    "treatment.csv.gz",
    "apachePatientResult.csv.gz",
    "apacheApsVar.csv.gz",
    "vitalPeriodic.csv.gz",
}

NUTRITION_TERMS = [
    "nutrition",
    "nutritional",
    "malnutrition",
    "undernutrition",
    "sarcopen",
    "cachexia",
    "frail",
    "dysphagia",
    "enteral",
    "parenteral",
    "tube feed",
    "tube-feed",
    "feeding",
    "diet",
    "calorie",
    "caloric",
    "protein",
    "albumin",
    "prealbumin",
    "weight",
    "height",
    "bmi",
    "glucose",
    "insulin",
    "nasogastric",
    "gastrostomy",
    "tpn",
]
NUTRITION_RE = re.compile("|".join(re.escape(t) for t in NUTRITION_TERMS), re.I)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", errors="replace")
    return path.open("rt", newline="", errors="replace")


def csv_header(path: Path) -> list[str]:
    with open_text(path) as f:
        return next(csv.reader(f))


def count_rows(path: Path) -> int:
    n = 0
    with open_text(path) as f:
        next(f, None)
        for n, _ in enumerate(f, 1):
            pass
    return n


def maybe_count(path: Path, rel: str, core: set[str]) -> int | None:
    if rel in core:
        return count_rows(path)
    return None


def inventory(root: Path, core: set[str]) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*.csv.gz")):
        rel = str(path.relative_to(root))
        header = csv_header(path)
        rows = maybe_count(path, rel, core)
        records.append(
            {
                "relative_path": rel,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "n_columns": len(header),
                "columns": header,
                "rows_if_counted": rows,
                "counted": rows is not None,
            }
        )
    return records


def read_dict_rows(path: Path, fields: list[str] | None = None) -> Iterable[dict]:
    with open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def mimic_cohort() -> dict:
    patients = MIMIC / "hosp" / "patients.csv.gz"
    admissions = MIMIC / "hosp" / "admissions.csv.gz"
    icustays = MIMIC / "icu" / "icustays.csv.gz"

    age_by_subject = {}
    subject_count = 0
    older_subject_count = 0
    for row in read_dict_rows(patients, ["subject_id", "anchor_age"]):
        subject_count += 1
        try:
            age = int(float(row["anchor_age"]))
        except Exception:
            continue
        age_by_subject[row["subject_id"]] = age
        if age >= 65:
            older_subject_count += 1

    hosp_expire_by_hadm = {}
    admission_count = 0
    for row in read_dict_rows(admissions, ["hadm_id", "hospital_expire_flag"]):
        admission_count += 1
        hosp_expire_by_hadm[row["hadm_id"]] = row.get("hospital_expire_flag", "")

    icu_count = 0
    older_icu_count = 0
    older_hosp_mort = 0
    older_hadm = set()
    older_subjects = set()
    first_icu_by_subject = {}
    for row in read_dict_rows(
        icustays, ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]
    ):
        icu_count += 1
        sid = row["subject_id"]
        age = age_by_subject.get(sid)
        if age is None or age < 65:
            continue
        older_icu_count += 1
        older_subjects.add(sid)
        older_hadm.add(row["hadm_id"])
        if hosp_expire_by_hadm.get(row["hadm_id"]) == "1":
            older_hosp_mort += 1
        current = first_icu_by_subject.get(sid)
        if current is None or row["intime"] < current["intime"]:
            first_icu_by_subject[sid] = row

    return {
        "patients_total": subject_count,
        "patients_age_ge_65": older_subject_count,
        "admissions_total": admission_count,
        "icu_stays_total": icu_count,
        "icu_stays_age_ge_65": older_icu_count,
        "unique_subjects_age_ge_65_with_icu": len(older_subjects),
        "unique_hadm_age_ge_65_with_icu": len(older_hadm),
        "first_icu_stays_age_ge_65_by_subject": len(first_icu_by_subject),
        "hospital_mortality_events_among_older_icu_stays": older_hosp_mort,
        "hospital_mortality_rate_among_older_icu_stays": round(
            older_hosp_mort / older_icu_count, 4
        )
        if older_icu_count
        else None,
    }


def parse_eicu_age(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith(">"):
        return 90
    try:
        return int(float(value))
    except Exception:
        return None


def eicu_cohort() -> dict:
    patient = EICU / "patient.csv.gz"
    total = 0
    older = 0
    older_hosp_mort = 0
    older_icu_mort = 0
    older_with_weight = 0
    older_with_height = 0
    unit_types = Counter()
    for row in read_dict_rows(
        patient,
        [
            "patientunitstayid",
            "age",
            "unittype",
            "admissionweight",
            "admissionheight",
            "hospitaldischargestatus",
            "unitdischargestatus",
        ],
    ):
        total += 1
        age = parse_eicu_age(row.get("age", ""))
        if age is None or age < 65:
            continue
        older += 1
        unit_types[row.get("unittype", "")] += 1
        if row.get("hospitaldischargestatus", "").lower() == "expired":
            older_hosp_mort += 1
        if row.get("unitdischargestatus", "").lower() == "expired":
            older_icu_mort += 1
        if row.get("admissionweight", "").strip():
            older_with_weight += 1
        if row.get("admissionheight", "").strip():
            older_with_height += 1
    return {
        "unit_stays_total": total,
        "unit_stays_age_ge_65": older,
        "hospital_mortality_events_age_ge_65": older_hosp_mort,
        "hospital_mortality_rate_age_ge_65": round(older_hosp_mort / older, 4)
        if older
        else None,
        "icu_mortality_events_age_ge_65": older_icu_mort,
        "icu_mortality_rate_age_ge_65": round(older_icu_mort / older, 4)
        if older
        else None,
        "older_stays_with_admissionweight": older_with_weight,
        "older_stays_with_admissionheight": older_with_height,
        "top_unit_types": unit_types.most_common(10),
    }


def scan_mimic_nutrition_terms() -> dict:
    hits = defaultdict(list)
    targets = [
        MIMIC / "icu" / "d_items.csv.gz",
        MIMIC / "hosp" / "d_labitems.csv.gz",
        MIMIC / "hosp" / "d_icd_diagnoses.csv.gz",
    ]
    for path in targets:
        with open_text(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = " | ".join(str(v) for v in row.values())
                if NUTRITION_RE.search(text):
                    hits[str(path.relative_to(MIMIC))].append(row)
    return {k: v[:250] for k, v in hits.items()}


def scan_eicu_nutrition_terms(max_hits_per_table: int = 250) -> dict:
    hits = defaultdict(list)
    target_names = [
        "lab.csv.gz",
        "intakeOutput.csv.gz",
        "medication.csv.gz",
        "treatment.csv.gz",
        "diagnosis.csv.gz",
        "nurseCharting.csv.gz",
        "nurseCare.csv.gz",
        "physicalExam.csv.gz",
    ]
    for name in target_names:
        path = EICU / name
        if not path.exists():
            continue
        with open_text(path) as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                text = " | ".join(str(v) for v in row.values())
                if not NUTRITION_RE.search(text):
                    continue
                compact = {k: v for k, v in row.items() if NUTRITION_RE.search(k) or NUTRITION_RE.search(str(v))}
                sig = json.dumps(compact, sort_keys=True)[:300]
                if sig in seen:
                    continue
                seen.add(sig)
                hits[name].append(compact or row)
                if len(hits[name]) >= max_hits_per_table:
                    break
    return dict(hits)


def write_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_report(results: dict) -> str:
    mimic_cohort_result = results["mimic_cohort"]
    eicu_cohort_result = results["eicu_cohort"]
    lines = [
        "# Gate-0 Data Inventory",
        "",
        "Status date: 2026-08-26",
        "",
        "## Dataset Paths",
        "",
        f"- MIMIC-IV: `{MIMIC}`",
        f"- eICU-CRD: `{EICU}`",
        "",
        "## Core Cohort Counts",
        "",
        "### MIMIC-IV",
        "",
    ]
    for k, v in mimic_cohort_result.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "### eICU", ""])
    for k, v in eicu_cohort_result.items():
        lines.append(f"- {k}: {v}")
    lines.extend(
        [
            "",
            "## Nutrition-Relevant Signals",
            "",
            f"- MIMIC dictionary/source tables with hits: {', '.join(results['mimic_nutrition_hits'].keys()) or 'none'}",
            f"- eICU source tables with hits: {', '.join(results['eicu_nutrition_hits'].keys()) or 'none'}",
            "",
            "## Preliminary Gate Decision",
            "",
        ]
    )
    go = (
        mimic_cohort_result.get("icu_stays_age_ge_65", 0) > 5000
        and eicu_cohort_result.get("unit_stays_age_ge_65", 0) > 5000
        and bool(results["mimic_nutrition_hits"])
        and bool(results["eicu_nutrition_hits"])
    )
    if go:
        lines.extend(
            [
                "Gate-0 status: GO for cohort construction and feature mapping.",
                "",
                "Rationale: both datasets contain large older-adult ICU cohorts and both contain nutrition-relevant variables or labels.",
            ]
        )
    else:
        lines.extend(
            [
                "Gate-0 status: CAUTION/PIVOT.",
                "",
                "Rationale: one or more required conditions need manual review before model training.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Build a harmonized first-24h or first-48h cohort table.",
            "- Freeze primary endpoint after leakage audit.",
            "- Create a cross-database feature mapping table.",
            "- Run baseline models before any advanced method.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "paths": {"mimic": str(MIMIC), "eicu": str(EICU), "out": str(OUT)},
        "mimic_inventory": inventory(MIMIC, CORE_MIMIC),
        "eicu_inventory": inventory(EICU, CORE_EICU),
        "mimic_cohort": mimic_cohort(),
        "eicu_cohort": eicu_cohort(),
        "mimic_nutrition_hits": scan_mimic_nutrition_terms(),
        "eicu_nutrition_hits": scan_eicu_nutrition_terms(),
    }
    write_json("gate0_inventory_full.json", results)
    (OUT / "GATE0_DATA_INVENTORY.md").write_text(markdown_report(results), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "mimic_older_icu_stays": results["mimic_cohort"]["icu_stays_age_ge_65"],
        "eicu_older_unit_stays": results["eicu_cohort"]["unit_stays_age_ge_65"],
        "mimic_nutrition_tables": list(results["mimic_nutrition_hits"].keys()),
        "eicu_nutrition_tables": list(results["eicu_nutrition_hits"].keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
