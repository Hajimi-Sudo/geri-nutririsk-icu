# V12 Review Repair Package

Status date: 2026-08-26

## Purpose

This package addresses external-review blockers for Frontiers in Nutrition: patient-level partitioning, clinical and nutrition proxy baselines, paired incremental value, calibration summaries, and reproducibility/log evidence.

## Split Audit

- MIMIC rows: 33671; unique subject IDs: 33671.
- MIMIC subject overlap between development and internal test: 0.
- eICU rows: 71861; unique uniquepid values: 71861; hospitals: 207.
- eICU uniquepid overlap between calibration and held-out evaluation: 0.

## Main LightGBM Feature-Set Results

| Model | Features | Held-out recalibrated AUROC | Brier | ECE |
|---|---:|---:|---:|---:|
| full_lightgbm | 39 | 0.794 | 0.088 | 0.021 |
| without_nutrition_support | 37 | 0.784 | 0.090 | 0.019 |
| without_nutrition_metabolic_labs | 23 | 0.771 | 0.090 | 0.017 |
| without_all_nutrition_construct | 20 | 0.778 | 0.089 | 0.018 |
| non_nutrition_clinical_baseline | 20 | 0.778 | 0.089 | 0.018 |
| severity_only | 8 | 0.706 | 0.099 | 0.040 |

## Interpretation Boundary

This repair package supports nutrition-risk-relevant mortality stratification, not malnutrition diagnosis, nutrition adequacy assessment, nutrition-treatment response, or prospective clinical utility.

Exact mNUTRIC, SOFA, GNRI, and PNI reconstruction is limited by cross-database variable availability. The reported clinical/nutrition scores are therefore marked as proxies unless exact source variables were available.
