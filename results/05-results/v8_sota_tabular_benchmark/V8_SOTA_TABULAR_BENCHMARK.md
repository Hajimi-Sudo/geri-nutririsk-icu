# V8 SOTA Tabular Benchmark

Status date: 2026-08-26

## Purpose

V8 benchmarks GERI-NutriRisk-ICU against strong contemporary tabular machine-learning comparators on the same harmonized MIMIC-IV development and eICU-CRD external-validation feature tables.

## Models Run

- GERI-NutriRisk-ICU_logistic
- XGBoost_hist
- LightGBM_gbdt
- CatBoost_ordered

## Key Metrics

| Model | MIMIC AUROC | eICU AUROC | eICU Brier | eICU Platt held-out Brier |
|---|---:|---:|---:|---:|
| GERI-NutriRisk-ICU_logistic | 0.830 | 0.727 | 0.155 | 0.099 |
| XGBoost_hist | 0.880 | 0.778 | 0.297 | 0.092 |
| LightGBM_gbdt | 0.882 | 0.793 | 0.291 | 0.088 |
| CatBoost_ordered | 0.881 | 0.741 | 0.377 | 0.099 |

## Interpretation

The strongest model should be selected by external validation, calibration, and interpretability rather than internal AUROC alone. If gradient-boosting models improve external discrimination without unacceptable calibration loss, they should be reported as SOTA comparators and either replace or complement the logistic framework. If they overfit or calibrate poorly, the logistic framework remains the primary clinical model.
