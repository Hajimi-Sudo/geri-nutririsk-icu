# V3 Severity-Enhanced External Validation Results

Status date: 2026-08-26

## Model

V3 extends V2 by adding first-24h organ-support and neurologic severity features: urine output, ventilation, dialysis, and GCS.

## Cohorts

- MIMIC-IV training cohort: n=33671, events=4659, event_rate=0.1384
- eICU external cohort: n=71861, events=8860, event_rate=0.1233

## Metrics

- MIMIC apparent AUROC: 0.829; Brier: 0.0895
- eICU external AUROC: 0.7281; Brier: 0.1534
- V3 held-out eICU Platt-recalibrated AUROC: 0.7291; Brier: 0.0992

## Interpretation

V3 is the current candidate main model if severity features improve external discrimination without making the nutrition-risk framing disappear. Domain recalibration should be reported as a model-updating analysis.
