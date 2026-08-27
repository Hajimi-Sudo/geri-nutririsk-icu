# V7 Model Benchmark and Operating Profile

Status date: 2026-08-26

## Purpose

V7 adds reviewer-facing evidence beyond the V0--V6 package: model-class benchmarking, five repeated row-index split rotations, operating characteristics at prespecified risk thresholds, and bootstrap confidence intervals for eICU subgroup performance.

## Key Results

- Primary logistic framework external AUROC, offset 0: 0.727.
- Weighted one-hidden-layer MLP external AUROC, offset 0: 0.654.
- Five-split eICU original AUROC mean (SD): 0.728 (0.002).
- Five-split eICU Platt-held-out Brier mean (SD): 0.099 (0.000).

## Interpretation

The interpretable standardized logistic framework remains the primary model because it provides stable external discrimination, explicit calibration updating, and coefficient-level interpretation. The nonlinear MLP benchmark is reported as a sensitivity comparator rather than as the clinical framework.

## Outputs

- `v7_model_benchmark_metrics.json`
- `table_model_class_and_feature_benchmark.csv`
- `table_operating_profile_original.csv`
- `table_operating_profile_platt_heldout.csv`
- `table_subgroup_ci_original.csv`
