# V11 TabNet Neural SOTA Comparator

Status date: 2026-08-26

## Purpose

This run adds a neural tabular comparator to the sealed SOTA queue after LightGBM was selected as the performance head.

## Key Metrics

| Model | MIMIC AUROC | eICU AUROC | eICU Platt held-out AUROC | eICU Platt held-out Brier |
|---|---:|---:|---:|---:|
| LightGBM_gbdt | 0.882 | 0.793 | 0.794 | 0.088 |
| TabNet_neural | 0.857 | 0.636 | 0.636 | 0.108 |

## Model-Lock Decision

TabNet minus LightGBM eICU held-out Platt AUROC: -0.1574.
Decision: keep_lightgbm.
