# V0 External Validation Results

Status date: 2026-08-26

## Model

Logistic regression trained in MIMIC-IV and externally validated in eICU. Features: age, male, bmi, enteral_24h, parenteral_24h.

## Cohorts

- MIMIC-IV training cohort: n=33671, events=4659, event_rate=0.1384
- eICU external cohort: n=103278, events=12022, event_rate=0.1164

## Metrics

- MIMIC apparent AUROC: 0.5888; Brier: 0.1176
- eICU external AUROC: 0.5759; Brier: 0.1036

## Interpretation

This is a deliberately minimal transportability smoke test. It demonstrates that a leakage-controlled cross-database pipeline can run, but manuscript-grade performance will require adding first-window laboratory and severity features.

Patient-level feature CSVs were kept on the server only and should not be copied into the manuscript workspace.
