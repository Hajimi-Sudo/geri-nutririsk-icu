# V15 V14 Explainability

Status date: 2026-08-26

Base held-out eICU AUROC: 0.872; Brier: 0.073.

## Leading Feature Groups By Permutation AUROC Drop
- Organ support and neurologic severity: AUROC drop 0.0627; Brier increase 0.0118.
- Vital signs: AUROC drop 0.0482; Brier increase 0.0110.
- Blood gas and perfusion: AUROC drop 0.0199; Brier increase 0.0052.
- Nutrition/metabolic labs: AUROC drop 0.0160; Brier increase 0.0027.
- Additional chemistry: AUROC drop 0.0096; Brier increase 0.0028.
- Demographics: AUROC drop 0.0075; Brier increase 0.0012.
- Missingness indicators: AUROC drop 0.0068; Brier increase 0.0017.
- Coagulation and hematology: AUROC drop 0.0055; Brier increase 0.0014.

## Leading Gain Features
- gcs_24h: gain fraction 0.119.
- calcium: gain fraction 0.058.
- bun: gain fraction 0.057.
- resp_rate: gain fraction 0.052.
- urine_24h_ml: gain fraction 0.046.
- lactate: gain fraction 0.043.
- heart_rate: gain fraction 0.040.
- icu_type_cardiac: gain fraction 0.033.
- ph: gain fraction 0.032.
- bmi: gain fraction 0.028.
- anion_gap: gain fraction 0.026.
- platelet: gain fraction 0.026.

## Boundary

Feature importance is explanatory for the fitted prediction model and does not support causal claims about nutrition treatment, malnutrition, or physiology.
