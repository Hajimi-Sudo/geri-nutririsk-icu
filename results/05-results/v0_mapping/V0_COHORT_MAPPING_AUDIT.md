# V0 Cohort and Cross-Database Mapping Audit

Status date: 2026-08-26

## Cohort Summary

### MIMIC-IV Development Cohort Candidate

- n_first_older_icu: 33671
- age: {'n': 33671, 'mean': 76.717, 'p25': 70.0, 'median': 76.0, 'p75': 83.0, 'min': 65.0, 'max': 91.0}
- los_days: {'n': 33666, 'mean': 3.428, 'p25': 1.123, 'median': 1.98, 'p75': 3.77, 'min': 0.007, 'max': 99.638}
- hospital_mortality_events: 5279
- hospital_mortality_rate: 0.1568
- gender: Counter({'M': 17991, 'F': 15680})
- careunit_top: [('Cardiac Vascular Intensive Care Unit (CVICU)', 6761), ('Medical Intensive Care Unit (MICU)', 6041), ('Medical/Surgical Intensive Care Unit (MICU/SICU)', 5033), ('Coronary Care Unit (CCU)', 4586), ('Surgical Intensive Care Unit (SICU)', 4276), ('Trauma SICU (TSICU)', 3405), ('Neuro Intermediate', 2203), ('Neuro Surgical Intensive Care Unit (Neuro SICU)', 746), ('Neuro Stepdown', 504), ('Surgery/Vascular/Intermediate', 48)]
- race_top: [('WHITE', 22243), ('UNKNOWN', 3604), ('BLACK/AFRICAN AMERICAN', 1952), ('OTHER', 971), ('WHITE - OTHER EUROPEAN', 914), ('UNABLE TO OBTAIN', 836), ('WHITE - RUSSIAN', 475), ('ASIAN - CHINESE', 395), ('ASIAN', 361), ('HISPANIC/LATINO - PUERTO RICAN', 220), ('PATIENT DECLINED TO ANSWER', 197), ('BLACK/CAPE VERDEAN', 195)]
- insurance: Counter({'Medicare': 28677, 'Private': 3048, 'Medicaid': 1431, '': 269, 'Other': 245, 'No charge': 1})

### eICU External Validation Cohort Candidate

- n_first_older_unit_stays_by_uniquepid: 71861
- age: {'n': 71861, 'mean': 76.747, 'p25': 70, 'median': 76, 'p75': 83, 'min': 65, 'max': 90}
- height_cm: {'n': 70349, 'mean': 167.873, 'p25': 160.0, 'median': 167.6, 'p75': 177.0, 'min': 0.0, 'max': 605.0}
- weight_kg: {'n': 67601, 'mean': 79.745, 'p25': 64.0, 'median': 77.1, 'p75': 91.5, 'min': 0.0, 'max': 880.6}
- bmi: {'n': 66787, 'mean': 28.115, 'p25': 23.227, 'median': 26.949, 'p75': 31.55, 'min': 8.855, 'max': 265.274}
- hospital_mortality_events: 8824
- hospital_mortality_rate: 0.1228
- icu_mortality_events: 5720
- icu_mortality_rate: 0.0796
- gender: Counter({'Male': 37082, 'Female': 34731, 'Unknown': 28, '': 16, 'Other': 4})
- ethnicity_top: [('Caucasian', 59147), ('African American', 5044), ('Other/Unknown', 2828), ('Hispanic', 2497), ('Asian', 1217), ('', 880), ('Native American', 248)]
- unit_type_top: [('Med-Surg ICU', 39869), ('CCU-CTICU', 6223), ('MICU', 6032), ('Neuro ICU', 5008), ('Cardiac ICU', 4807), ('SICU', 4168), ('CSICU', 3219), ('CTICU', 2535)]

## Candidate Shared Feature Domains

| Domain | Feature | MIMIC-IV source | eICU source | Status |
|---|---|---|---|---|
| demographics | age | hosp/patients.anchor_age | patient.age | shared |
| demographics | sex/gender | hosp/patients.gender | patient.gender | shared |
| anthropometrics | height | hosp/omr result_name/item labels; requires extraction | patient.admissionheight | shared_with_mapping_work |
| anthropometrics | weight | icu d_items/input/chartevents or hosp/omr; requires extraction | patient.admissionweight | shared_with_mapping_work |
| nutrition/metabolic labs | albumin | hosp/d_labitems + labevents | apacheApsVar.albumin and lab.labname | shared |
| nutrition/metabolic labs | glucose | hosp/d_labitems + labevents | apacheApsVar.glucose and lab.labname | shared |
| renal/metabolic | creatinine | hosp/d_labitems + labevents | apacheApsVar.creatinine and lab.labname | shared |
| renal/metabolic | BUN | hosp/d_labitems + labevents | apacheApsVar.bun and lab.labname | shared |
| severity | ventilation | procedureevents/chartevents; requires extraction | apacheApsVar.vent | shared_with_mapping_work |
| severity | dialysis/CRRT | procedures/inputevents/diagnoses; requires extraction | apacheApsVar.dialysis | shared_with_mapping_work |
| outcome | hospital mortality | hosp/admissions.hospital_expire_flag | patient.hospitaldischargestatus or apachePatientResult.actualhospitalmortality | shared |
| nutrition support | enteral/parenteral nutrition signal | icu/inputevents and ingredientevents itemids; requires itemid extraction | intakeOutput/treatment/medication/nurseCharting text signals | candidate_shared_proxy |

## Nutrition-Specific Feasibility Signals

- MIMIC lab concepts found: albumin, bilirubin, bun, creatinine, glucose, hematocrit, sodium, wbc
- MIMIC ICU item concepts found: enteral_nutrition, height, parenteral_nutrition, weight
- eICU lab concepts found: albumin, bilirubin, bun, creatinine, glucose, prealbumin, sodium, wbc
- eICU text signal counts are stored in `v0_cohort_mapping_full.json`.

## V0 Decision

Proceed to cohort construction with hospital mortality as the first robust endpoint and a parallel nutrition-support proxy audit.

The target Frontiers in Nutrition framing should emphasize a transportable nutrition-risk framework rather than a generic ICU mortality model.
