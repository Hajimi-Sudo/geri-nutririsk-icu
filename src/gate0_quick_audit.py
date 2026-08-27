#!/usr/bin/env python3
"""Fast Gate-0 audit for the MIMIC-IV + eICU nutrition-risk project.

The quick pass avoids full scans of very large event tables. It gives enough
evidence to decide whether cohort construction and external validation can begin.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MIMIC = Path("/root/autodl-tmp/mimiciv/mimic-iv-3.1")
EICU = Path("/root/autodl-tmp/eicu/base")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/gate0")

NUTRITION_TERMS = [
    "nutrition", "nutritional", "malnutrition", "undernutrition", "sarcopen",
    "cachexia", "frail", "dysphagia", "enteral", "parenteral", "tube feed",
    "feeding", "diet", "calorie", "protein", "albumin", "prealbumin",
    "weight", "height", "bmi", "glucose", "insulin", "tpn",
]
NUTRITION_RE = re.compile("|".join(re.escape(t) for t in NUTRITION_TERMS), re.I)

MIMIC_FILES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "icu/d_items.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/omr.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/ingredientevents.csv.gz",
    "hosp/labevents.csv.gz",
]
EICU_FILES = [
    "patient.csv.gz",
    "lab.csv.gz",
    "intakeOutput.csv.gz",
    "medication.csv.gz",
    "treatment.csv.gz",
    "diagnosis.csv.gz",
    "apachePatientResult.csv.gz",
    "apacheApsVar.csv.gz",
    "vitalPeriodic.csv.gz",
    "nurseCharting.csv.gz",
]


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", errors="replace")


def header(path: Path) -> list[str]:
    with open_text(path) as f:
        return next(csv.reader(f))


def row_count(path: Path) -> int:
    with open_text(path) as f:
        next(f, None)
        return sum(1 for _ in f)


def inventory(root: Path, rels: list[str], count_rels: set[str]) -> list[dict]:
    out = []
    for rel in rels:
        p = root / rel
        if not p.exists():
            out.append({"relative_path": rel, "exists": False})
            continue
        h = header(p)
        out.append({
            "relative_path": rel,
            "exists": True,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "n_columns": len(h),
            "columns": h,
            "rows_if_counted": row_count(p) if rel in count_rels else None,
            "counted": rel in count_rels,
        })
    return out


def dict_rows(path: Path, fields: list[str] | None = None):
    with open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                yield {k: row.get(k, "") for k in fields}
            else:
                yield row


def mimic_cohort() -> dict:
    ages = {}
    for r in dict_rows(MIMIC / "hosp/patients.csv.gz", ["subject_id", "anchor_age"]):
        try:
            ages[r["subject_id"]] = int(float(r["anchor_age"]))
        except Exception:
            pass
    hosp_mort = {}
    for r in dict_rows(MIMIC / "hosp/admissions.csv.gz", ["hadm_id", "hospital_expire_flag"]):
        hosp_mort[r["hadm_id"]] = r["hospital_expire_flag"]
    total = older = older_mort = 0
    subjects = set()
    hadms = set()
    for r in dict_rows(MIMIC / "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id"]):
        total += 1
        if ages.get(r["subject_id"], -1) >= 65:
            older += 1
            subjects.add(r["subject_id"])
            hadms.add(r["hadm_id"])
            older_mort += int(hosp_mort.get(r["hadm_id"], "0") == "1")
    return {
        "patients_total": len(ages),
        "patients_age_ge_65": sum(1 for a in ages.values() if a >= 65),
        "icu_stays_total": total,
        "icu_stays_age_ge_65": older,
        "unique_subjects_age_ge_65_with_icu": len(subjects),
        "unique_hadm_age_ge_65_with_icu": len(hadms),
        "hospital_mortality_events_among_older_icu_stays": older_mort,
        "hospital_mortality_rate_among_older_icu_stays": round(older_mort / older, 4) if older else None,
    }


def parse_age(x: str) -> int | None:
    x = (x or "").strip()
    if not x:
        return None
    if x.startswith(">"):
        return 90
    try:
        return int(float(x))
    except Exception:
        return None


def eicu_cohort() -> dict:
    total = older = hmort = imort = weight = height = 0
    unit_types = Counter()
    for r in dict_rows(EICU / "patient.csv.gz", [
        "age", "unittype", "admissionweight", "admissionheight",
        "hospitaldischargestatus", "unitdischargestatus",
    ]):
        total += 1
        age = parse_age(r["age"])
        if age is None or age < 65:
            continue
        older += 1
        unit_types[r["unittype"]] += 1
        hmort += int(r["hospitaldischargestatus"].lower() == "expired")
        imort += int(r["unitdischargestatus"].lower() == "expired")
        weight += int(bool(r["admissionweight"].strip()))
        height += int(bool(r["admissionheight"].strip()))
    return {
        "unit_stays_total": total,
        "unit_stays_age_ge_65": older,
        "hospital_mortality_events_age_ge_65": hmort,
        "hospital_mortality_rate_age_ge_65": round(hmort / older, 4) if older else None,
        "icu_mortality_events_age_ge_65": imort,
        "icu_mortality_rate_age_ge_65": round(imort / older, 4) if older else None,
        "older_stays_with_admissionweight": weight,
        "older_stays_with_admissionheight": height,
        "top_unit_types": unit_types.most_common(10),
    }


def scan_table(path: Path, max_rows: int = 200000, max_hits: int = 80) -> list[dict]:
    hits = []
    seen = set()
    with open_text(path) as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            text = " | ".join(str(v) for v in r.values())
            if not NUTRITION_RE.search(text):
                continue
            compact = {k: v for k, v in r.items() if NUTRITION_RE.search(k) or NUTRITION_RE.search(str(v))}
            if not compact:
                compact = {k: r[k] for k in list(r)[:6]}
            sig = json.dumps(compact, sort_keys=True)[:400]
            if sig in seen:
                continue
            seen.add(sig)
            hits.append(compact)
            if len(hits) >= max_hits:
                break
    return hits


def nutrition_hits() -> dict:
    out = {"mimic": {}, "eicu": {}}
    for rel in ["icu/d_items.csv.gz", "hosp/d_labitems.csv.gz", "hosp/d_icd_diagnoses.csv.gz", "hosp/omr.csv.gz"]:
        p = MIMIC / rel
        if p.exists():
            hits = scan_table(p, max_rows=300000, max_hits=100)
            if hits:
                out["mimic"][rel] = hits
    for rel in ["lab.csv.gz", "intakeOutput.csv.gz", "medication.csv.gz", "treatment.csv.gz", "diagnosis.csv.gz", "nurseCharting.csv.gz"]:
        p = EICU / rel
        if p.exists():
            hits = scan_table(p, max_rows=300000, max_hits=100)
            if hits:
                out["eicu"][rel] = hits
    return out


def write_report(res: dict) -> None:
    m = res["mimic_cohort"]
    e = res["eicu_cohort"]
    go = (
        m["icu_stays_age_ge_65"] > 5000
        and e["unit_stays_age_ge_65"] > 5000
        and bool(res["nutrition_hits"]["mimic"])
        and bool(res["nutrition_hits"]["eicu"])
    )
    lines = [
        "# Gate-0 Quick Data Inventory",
        "",
        "Status date: 2026-08-26",
        "",
        "## Paths",
        "",
        f"- MIMIC-IV: `{MIMIC}`",
        f"- eICU-CRD: `{EICU}`",
        "",
        "## Cohort Feasibility",
        "",
        "### MIMIC-IV",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in m.items()]
    lines += ["", "### eICU", ""]
    lines += [f"- {k}: {v}" for k, v in e.items()]
    lines += [
        "",
        "## Nutrition-Relevant Table Hits",
        "",
        f"- MIMIC-IV: {', '.join(res['nutrition_hits']['mimic'].keys()) or 'none'}",
        f"- eICU: {', '.join(res['nutrition_hits']['eicu'].keys()) or 'none'}",
        "",
        "## Gate-0 Decision",
        "",
        "GO for cohort construction and cross-database feature mapping." if go else "CAUTION: manual review needed before cohort construction.",
        "",
        "## Immediate Next Experiment",
        "",
        "Build first-24h and first-48h cohort tables with a frozen shared feature dictionary. Use hospital mortality as the initial robust endpoint, while auditing nutrition-specific proxies separately.",
    ]
    (OUT / "GATE0_QUICK_DATA_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "mimic_inventory": inventory(MIMIC, MIMIC_FILES, {"hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz", "icu/d_items.csv.gz", "hosp/d_labitems.csv.gz", "hosp/d_icd_diagnoses.csv.gz"}),
        "eicu_inventory": inventory(EICU, EICU_FILES, {"patient.csv.gz", "apachePatientResult.csv.gz", "apacheApsVar.csv.gz"}),
        "mimic_cohort": mimic_cohort(),
        "eicu_cohort": eicu_cohort(),
        "nutrition_hits": nutrition_hits(),
    }
    (OUT / "gate0_quick_inventory_full.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(res)
    print(json.dumps({
        "out": str(OUT),
        "mimic_older_icu_stays": res["mimic_cohort"]["icu_stays_age_ge_65"],
        "eicu_older_unit_stays": res["eicu_cohort"]["unit_stays_age_ge_65"],
        "mimic_nutrition_tables": list(res["nutrition_hits"]["mimic"]),
        "eicu_nutrition_tables": list(res["nutrition_hits"]["eicu"]),
    }, indent=2))


if __name__ == "__main__":
    main()
