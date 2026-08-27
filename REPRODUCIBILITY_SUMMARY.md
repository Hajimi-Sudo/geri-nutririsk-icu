# Reproducibility Summary

This release contains the public, non-identifying materials needed to audit the manuscript results.

## Included

- Analysis scripts for cohort mapping, feature extraction, source-only validation, recalibration, model benchmarking, domain updating, APACHE comparison, nutrition-feature ablation, explainability, and hospital-cluster robustness.
- Aggregate result files in JSON, CSV, and Markdown format.
- Manuscript-ready figures and tables.
- English and Chinese LaTeX manuscript sources and compiled PDFs.

## Excluded

- Original MIMIC-IV and eICU-CRD database files.
- Derived patient-level feature tables.
- Row-level model predictions or identifiers.
- Server credentials, tokens, SSH information, or data-use agreement documents.

## Main Locked Results

- Final held-out eICU cohort: 57,488 ICU stays and 7,130 in-hospital mortality events.
- Locked domain-updated GERI-NutriRisk-ICU: AUROC 0.872, Brier score 0.073, expected calibration error 0.004.
- APACHE-IVa same-subset comparison: GERI-NutriRisk-ICU AUROC 0.871 versus APACHE-IVa AUROC 0.850.
- Paired AUROC difference versus APACHE-IVa: 0.0209, 95% CI 0.0158-0.0260.
- Nutrition/metabolic laboratory removal reduced domain-updated AUROC by 0.0059, 95% CI 0.0041-0.0077.

