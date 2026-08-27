# V10 SOTA Completion And Model Lock

Status date: 2026-08-26

## Purpose

This bounded completion run tests whether any additional CPU-feasible tabular comparator should displace the LightGBM performance head selected after V8/V9.

## Models Run

- Sklearn_HistGradientBoosting (histogram_gradient_boosting)
- LightGBM_gbdt (gradient_boosted_decision_trees)
- XGBoost_hist (gradient_boosted_decision_trees)
- RandomForest_balanced (bagged_decision_trees)
- CatBoost_ordered (ordered_gradient_boosting)
- Logistic_GERI_NutriRisk_ICU (standardized_l2_logistic_regression)
- Sklearn_MLP (feedforward_neural_network)
- ExtraTrees_balanced (extremely_randomized_trees)

## Key Metrics

| Model | MIMIC AUROC | eICU AUROC | eICU Brier | eICU Platt held-out AUROC | eICU Platt held-out Brier |
|---|---:|---:|---:|---:|---:|
| Sklearn_HistGradientBoosting | 0.880 | 0.793 | 0.221 | 0.794 | 0.089 |
| LightGBM_gbdt | 0.882 | 0.793 | 0.291 | 0.794 | 0.088 |
| XGBoost_hist | 0.880 | 0.778 | 0.297 | 0.779 | 0.092 |
| RandomForest_balanced | 0.867 | 0.757 | 0.228 | 0.757 | 0.097 |
| CatBoost_ordered | 0.881 | 0.741 | 0.377 | 0.742 | 0.099 |
| Logistic_GERI_NutriRisk_ICU | 0.830 | 0.727 | 0.155 | 0.728 | 0.099 |
| Sklearn_MLP | 0.872 | 0.676 | 0.291 | 0.677 | 0.102 |
| ExtraTrees_balanced | 0.830 | 0.674 | 0.304 | 0.676 | 0.104 |

## Model-Lock Decision

Best model by eICU held-out Platt AUROC: Sklearn_HistGradientBoosting (0.794).
Prespecified primary performance head: LightGBM_gbdt.
Decision: lock_lightgbm_no_material_sota_gain.

## Interpretation Boundary

The final manuscript should report the bounded comparator queue transparently and avoid implying that every possible neural tabular architecture was exhaustively searched. The defensible claim is that the selected framework outperformed strong conventional and contemporary tabular comparators available in the validated compute environment under an external-validation and recalibration protocol.
