# V2b External Recalibration Results

Status date: 2026-08-26

## Design

The MIMIC-IV trained V2 model was applied to eICU. A deterministic 20% eICU calibration split was used for recalibration, and the remaining 80% was held out for external test evaluation.

## Held-Out eICU Test Metrics

- Original V2: AUROC=0.673; Brier=0.2126
- Intercept-only recalibration: AUROC=0.673; Brier=0.1058
- Logistic recalibration: AUROC=0.673; Brier=0.103

## Calibration Parameters

- Intercept-only delta: -1.9890
- Logistic recalibration intercept: -1.8819
- Logistic recalibration slope: 0.5014

## Interpretation

This analysis tests whether a light external-domain calibration layer can repair cross-database probability shift while preserving discrimination. It should be framed as model updating, not as pure untouched external validation.
