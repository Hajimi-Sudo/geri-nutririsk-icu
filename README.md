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
10. `run_v16_hospital_cluster_robustness.py` (Historical V16: cluster bootstrap on primary split)
11. `export_final_assets.py` and `restyle_nature_figures.py`
12. `run_leave_hospital_out_validation.py` (Revision EXP-001: strict 20/20/167 leave-hospital-out validation across 167 unseen hospitals)
13. `run_nutrition_construct_ablation.py` (Revision EXP-004: strict dissected nutrition-reserve vs. metabolic-labs ablation)
14. `run_revision_server_experiments.py` (Revision suite covering EXP-001 through EXP-007)

### Multicenter Validation Architecture

The framework incorporates two complementary multicenter validation analyses:

1. **Hospital Cluster-Bootstrap Robustness (`run_v16_hospital_cluster_robustness.py`)**:
   Evaluates intra-cohort robustness across participating hospitals within the primary patient-level test partition (500 resamples across 206 hospitals; Table 22/23).
2. **Strict Leave-Hospital-Out Validation (`run_leave_hospital_out_validation.py`)**:
   Partitions the 207 eICU hospitals strictly at the institutional level into 20 domain-update hospitals (n = 9,949), 20 calibration hospitals (n = 4,376), and 167 completely unseen held-out hospitals (n = 57,536; AUROC 0.871, Brier 0.074, ECE 0.009). Patients from the 167 held-out hospitals are completely excluded from training, updating, and calibration to prove true center-level generalizability.


### Nutrition-Construct Ablation Architecture

To resolve the distinction between baseline nutritional reserve and acute multiorgan metabolic derangement (Reviewer #2, R2-C5), the repository provides both the historical pre-revision baseline and the authoritative revision ablation:

1. **Historical Pre-Revision Development Baseline (`run_v14_domain_update_nutrition_apache_lock.py`)**:
   Historical August 2026 development lock script evaluating early coarse feature groupings (76 features without nutrition, 67 features without metabolic labs).
2. **Authoritative Revision Ablation (`run_nutrition_construct_ablation.py` / EXP-004)**:
   Strict, non-overlapping feature dissection executed on the held-out eICU cohort ($n = 57,488$ stays across 206 hospitals):
   - **Full Model**: 83 cross-database harmonized features (AUROC = 0.8722, Brier = 0.0732, ECE = 0.0037).
   - **Without Explicit Nutrition Reserve**: Removes `{"bmi", "albumin", "albumin_missing"}` (3 features -> **80 features remaining**; AUROC = 0.8712, $\Delta_{\\mathrm{AUC}} = -0.0010$, Brier = 0.0736, ECE = 0.0047).
   - **Without Nutrition Support**: Removes `{"enteral_24h", "parenteral_24h"}` (2 features -> **81 features remaining**; AUROC = 0.8716, Brier = 0.0735).
   - **Without Explicit Nutrition & Support**: Removes both sets (5 features -> **78 features remaining**; AUROC = 0.8711, $\Delta_{\\mathrm{AUC}} = -0.0011$).
   - **Without Acute Metabolic Laboratories**: Removes 7 acute labs (`glucose`, `creatinine`, `bun`, `sodium`, `bilirubin`, `hematocrit`, `wbc`) and their 7 missingness flags (14 features -> **69 features remaining**; strictly excludes albumin and BMI; AUROC = 0.8679, $\Delta_{\\mathrm{AUC}} = -0.0043$, Brier = 0.0741).

## Privacy Boundary

Do not commit or redistribute:

- original MIMIC-IV or eICU-CRD database files;
- patient-level extracted feature tables;
- row-level predictions or identifiers;
- credential files, server passwords, tokens, or data-use agreement materials.

Only aggregate tables and non-identifying summaries are included here.
