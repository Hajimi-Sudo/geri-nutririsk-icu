# Gate-0 Quick Data Inventory

Status date: 2026-08-26

## Paths

- MIMIC-IV: `/root/autodl-tmp/mimiciv/mimic-iv-3.1`
- eICU-CRD: `/root/autodl-tmp/eicu/base`

## Cohort Feasibility

### MIMIC-IV

- patients_total: 364627
- patients_age_ge_65: 95715
- icu_stays_total: 94458
- icu_stays_age_ge_65: 47444
- unique_subjects_age_ge_65_with_icu: 33671
- unique_hadm_age_ge_65_with_icu: 42766
- hospital_mortality_events_among_older_icu_stays: 7172
- hospital_mortality_rate_among_older_icu_stays: 0.1512

### eICU

- unit_stays_total: 200859
- unit_stays_age_ge_65: 103278
- hospital_mortality_events_age_ge_65: 12022
- hospital_mortality_rate_age_ge_65: 0.1164
- icu_mortality_events_age_ge_65: 7016
- icu_mortality_rate_age_ge_65: 0.0679
- older_stays_with_admissionweight: 94495
- older_stays_with_admissionheight: 101265
- top_unit_types: [('Med-Surg ICU', 57460), ('MICU', 9021), ('CCU-CTICU', 8312), ('Neuro ICU', 6736), ('Cardiac ICU', 6414), ('SICU', 6058), ('CSICU', 5875), ('CTICU', 3402)]

## Nutrition-Relevant Table Hits

- MIMIC-IV: icu/d_items.csv.gz, hosp/d_labitems.csv.gz, hosp/d_icd_diagnoses.csv.gz, hosp/omr.csv.gz
- eICU: lab.csv.gz, intakeOutput.csv.gz, medication.csv.gz, treatment.csv.gz, diagnosis.csv.gz, nurseCharting.csv.gz

## Gate-0 Decision

GO for cohort construction and cross-database feature mapping.

## Immediate Next Experiment

Build first-24h and first-48h cohort tables with a frozen shared feature dictionary. Use hospital mortality as the initial robust endpoint, while auditing nutrition-specific proxies separately.
