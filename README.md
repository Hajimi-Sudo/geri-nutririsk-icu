# GERI-NutriRisk-ICU

Code and aggregate reproducibility materials for:

**GERI-NutriRisk-ICU: an APACHE-contextualized, hospital-robust, and nutrition-aware domain-updated framework for mortality risk stratification in older ICU patients**

This repository contains analysis scripts, manuscript figure/table export scripts, aggregate result tables, and non-identifying reproducibility records. It does **not** contain patient-level MIMIC-IV or eICU-CRD data, derived patient-level feature tables, or credentialed database files.

Public repository: https://github.com/Hajimi-Sudo/geri-nutririsk-icu

## Data Access

The patient-level source data are available to credentialed users through PhysioNet:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/ and https://doi.org/10.13026/kpb9-mt58
- eICU Collaborative Research Database v2.0: https://physionet.org/content/eicu-crd/2.0/ and https://doi.org/10.13026/C2WM1R

Users must complete the required training and data-use agreements before accessing either database.

## Repository Layout

- `src/`: analysis and export scripts used for the staged GERI-NutriRisk-ICU experiments.
- `results/`: aggregate metrics, model-comparison tables, robustness summaries, and reproducibility reports.
- `tables/`: manuscript-ready aggregate tables and figure source tables.
- `figures/`: manuscript figures and supplementary figures.
- `manuscript/`: LaTeX source for English and Chinese manuscript versions.

## Reproducibility Notes

The analysis scripts were run on credentialed local/server copies of MIMIC-IV v3.1 and eICU-CRD v2.0. Some scripts contain path constants matching the original analysis environment; adjust these paths to point to local credentialed copies of the datasets before rerunning.

The staged analysis sequence was:

1. `gate0_quick_audit.py` and `gate0_data_audit.py`
2. `build_v0_cohort_mapping.py`
3. `run_v0_external_validation.py` through `run_v4_robustness_package.py`
4. `run_v7_model_benchmark_and_operating_profile.py` through `run_v11_tabnet_neural_sota.py`
5. `run_v12_review_repair_package.py`
6. `run_v13_strong_clinical_feature_upgrade.py`
7. `run_v13b_domain_update_calibration.py`
8. `run_v14_domain_update_nutrition_apache_lock.py`
9. `run_v15_v14_explainability.py`
10. `run_v16_hospital_cluster_robustness.py`
11. `export_final_assets.py` and `restyle_nature_figures.py`

## Privacy Boundary

Do not commit or redistribute:

- original MIMIC-IV or eICU-CRD database files;
- patient-level extracted feature tables;
- row-level predictions or identifiers;
- credential files, server passwords, tokens, or data-use agreement materials.

Only aggregate tables and non-identifying summaries are included here.
