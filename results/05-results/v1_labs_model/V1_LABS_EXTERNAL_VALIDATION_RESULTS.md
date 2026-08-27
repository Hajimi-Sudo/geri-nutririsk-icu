# V1 Laboratory-Enhanced External Validation Results

Status date: 2026-08-26

## Model

MIMIC-IV trained logistic regression with early nutrition/metabolic laboratory features and external validation in eICU.

Features: age, male, bmi, enteral_24h, parenteral_24h, albumin, glucose, creatinine, bun, sodium, bilirubin, hematocrit, wbc, albumin_missing, glucose_missing, creatinine_missing, bun_missing, sodium_missing, bilirubin_missing, hematocrit_missing, wbc_missing.

## Cohorts

- MIMIC-IV training cohort: n=33671, events=4659, event_rate=0.1384
- eICU external cohort: n=71861, events=8860, event_rate=0.1233

## Metrics

- MIMIC apparent AUROC: 0.7685; Brier: 0.1042
- eICU external AUROC: 0.6448; Brier: 0.136

## Feature Availability

### MIMIC-IV non-missing counts

- age: 33671
- male: 33671
- bmi: 17239
- enteral_24h: 33671
- parenteral_24h: 33671
- albumin: 8402
- glucose: 32659
- creatinine: 32647
- bun: 32635
- sodium: 32699
- bilirubin: 13354
- hematocrit: 32563
- wbc: 32455

### eICU non-missing counts

- age: 71861
- male: 71861
- bmi: 64641
- enteral_24h: 71861
- parenteral_24h: 71861
- albumin: 22000
- glucose: 52465
- creatinine: 47215
- bun: 47003
- sodium: 47418
- bilirubin: 19610
- hematocrit: 46868
- wbc: 45410

## Interpretation

V1 tests whether adding shared laboratory markers improves transportability beyond the V0 demographic/anthropometric/nutrition-support smoke test. Patient-level feature files remain on the server only.
