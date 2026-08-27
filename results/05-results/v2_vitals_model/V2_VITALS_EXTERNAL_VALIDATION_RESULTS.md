# V2 Vital-Enhanced External Validation Results

Status date: 2026-08-26

## Model

MIMIC-IV trained logistic regression with early nutrition/metabolic laboratory features plus first-24h vital signs; externally validated in eICU.

Feature families: demographics, BMI, early enteral/parenteral nutrition support signals, albumin/glucose/renal/metabolic labs, and vital signs.

## Cohorts

- MIMIC-IV training cohort: n=33671, events=4659, event_rate=0.1384
- eICU external cohort: n=71861, events=8860, event_rate=0.1233

## Metrics

- MIMIC apparent AUROC: 0.7909; Brier: 0.0974
- eICU external AUROC: 0.6718; Brier: 0.2129

## Interpretation

V2 evaluates whether a cross-database nutrition-risk framework gains transportable discrimination after adding shared physiological severity signals. Patient-level files remain on the server only.
