# V4 Robustness Package

Status date: 2026-08-26

## Design

V4 reuses the server-only V3 feature tables to generate manuscript-grade robustness checks: MIMIC internal validation, eICU external validation, feature-family ablation, subgroup performance, calibration, and decision-curve data.

## Feature-Family Ablation

| Model | MIMIC internal test AUROC | eICU external AUROC | eICU external Brier |
|---|---:|---:|---:|
| V0_demographics_bmi_nutrition_support | 0.58384 | 0.57495 | 0.10841 |
| V1_plus_labs | 0.76455 | 0.6427 | 0.13744 |
| V2_plus_vitals | 0.78607 | 0.66627 | 0.22078 |
| V3_plus_severity | 0.82952 | 0.72739 | 0.15493 |
| nutrition_metabolic_core | 0.76455 | 0.6427 | 0.13744 |
| severity_only_baseline | 0.78184 | 0.72179 | 0.11853 |

## Current Main Model

- V3 full eICU external AUROC: 0.72739
- V3 full eICU external Brier: 0.15493
- V3 held-out eICU Platt AUROC: 0.72849
- V3 held-out eICU Platt Brier: 0.09893

## Outputs

- `v4_robustness_metrics.json`: full aggregate metrics, calibration bins, ROC points, decision-curve data, and subgroup results.

Patient-level data remain on the server only.
