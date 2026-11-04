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
15. `run_apache_recalibration_and_baselines.py` (Revision EXP-002 & EXP-003: symmetric APACHE-IVa Platt recalibration, baseline suite, and paired DCA on n = 39,920 stays)
16. `run_hyperparameter_search_and_cv.py` (Revision 5-fold stratified cross-validation hyperparameter audit across 6 candidate configurations on pooled development partition N = 34,123)

### LightGBM Model Specification & Metric Standards

- **Locked Hyperparameters**: The primary LightGBM performance head is trained with:
  `n_estimators=1400`, `learning_rate=0.02`, `num_leaves=31`, `subsample=0.90`, `colsample_bytree=0.90`, `min_child_samples=80`, `reg_lambda=2.0`, `reg_alpha=0.0`.
  Outcome class imbalance (12.4% in-hospital mortality) is addressed prior to Platt recalibration via dynamic inverse event weighting:
  `scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)`.
- **Metric Computation**: All evaluation scripts strictly standardize on `sklearn.metrics.roc_auc_score` for AUROC calculation to enforce unbiased mid-rank handling for any tied prediction probabilities across tree models.

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
   - **Full Model**: 83 cross-database harmonized features (AUROC = 0.8722, Brier = 0.0733, ECE = 0.0041).
   - **Without Explicit Nutrition Reserve**: Removes `{"bmi", "albumin", "albumin_missing"}` (3 features dropped -> **80 features remaining**; AUROC = 0.8718, paired $\Delta_{\mathrm{AUC}} = +0.0003$, 95% CI: $-0.0010$ to $+0.0016$, $p = 0.621$, Brier = 0.0735, ECE = 0.0044).
   - **Without Early Nutrition Support**: Removes `{"enteral_24h", "parenteral_24h"}` (2 features dropped -> **81 features remaining**; AUROC = 0.8715, paired $\Delta_{\mathrm{AUC}} = +0.0006$, 95% CI: $-0.0002$ to $+0.0013$, $p = 0.0938$, Brier = 0.0734, ECE = 0.0041).
   - **Without Explicit Nutrition & Support**: Removes both sets (5 features dropped -> **78 features remaining**; AUROC = 0.8714, paired $\Delta_{\mathrm{AUC}} = +0.0008$, 95% CI: $-0.0005$ to $+0.0022$, $p = 0.219$, Brier = 0.0736, ECE = 0.0040).
   - **Without Acute Metabolic Laboratories**: Removes 7 acute labs (`glucose`, `creatinine`, `bun`, `sodium`, `bilirubin`, `hematocrit`, `wbc`) and their 7 missingness flags (14 features dropped -> **69 features remaining**; strictly excludes albumin and BMI; AUROC = 0.8679, paired $\Delta_{\mathrm{AUC}} = +0.0043$, 95% CI: $+0.0028$ to $+0.0060$, $p = 1.27 \times 10^{-7}$, Brier = 0.0740, $\Delta_{\mathrm{Brier}} = -0.0007$, ECE = 0.0042).
   - **Without Full Composite Nutrition Construct**: Removes all 19 nutritional and metabolic laboratory variables and flags combined (19 features dropped -> **64 features remaining**; AUROC = 0.8679, paired $\Delta_{\mathrm{AUC}} = +0.0042$, $p = 2.25 \times 10^{-5}$, Brier = 0.0742, ECE = 0.0049).
   - **Historical Pre-Revision Benchmarks (for audit transparency)**:
     - Historical without broad nutrition (76 features): AUROC = 0.8711, paired $\Delta_{\mathrm{AUC}} = +0.0011$, $p = 0.132$.
     - Historical without metabolic labs (67 features): AUROC = 0.8665, paired $\Delta_{\mathrm{AUC}} = +0.0057$, $p = 7.62 \times 10^{-11}$.

### APACHE-IVa Recalibration & Benchmark Architecture (EXP-002 & EXP-003)

To ensure an unassailably fair, symmetric comparison against established ICU clinical risk scoring systems (Reviewer #2, R2-C3 & R2-C4), the repository provides the standalone reproduction script `src/run_apache_recalibration_and_baselines.py`:

1. **Historical Pre-Revision Distinction**:
   In historical August 2026 pre-revision development scripts (`run_v14_domain_update_nutrition_apache_lock.py` and `run_v16_hospital_cluster_robustness.py`), APACHE predictions were loaded directly from the database field `predictedhospitalmortality` to represent existing raw hospital-grade prognostic risk.
2. **Symmetric Platt Recalibration Protocol**:
   In revision experiment EXP-002, APACHE-IVa is subjected to the exact same Platt scaling protocol as GERI-NutriRisk-ICU on the disjoint 10% eICU calibration partition ($n = 7,186$ stays; $5,000$ with valid APACHE scores):
   $$\text{logit}(P_{\text{recal}}) = -0.3537 + 1.0129 \cdot \text{logit}(P_{\text{APACHE-IVa}})$$
3. **Paired Head-to-Head Performance ($n = 39,920$ paired stays)**:
   - **Raw APACHE-IVa**: AUROC = 0.8504, Brier = 0.0820, ECE = 0.0289.
   - **Recalibrated APACHE-IVa**: AUROC = 0.8504, Brier = 0.0811, ECE = 0.0079.
   - **GERI-NutriRisk-ICU**: AUROC = 0.8713, Brier = 0.0741, ECE = 0.0092.
   - **Paired Advantage**: Statistically significant paired AUROC gain of $+0.0209$ (95% CI: $0.0158$--$0.0260$; $p < 0.001$) and lower Brier score.
4. **Sample Size Disambiguation for Figure 3**:
   - **Figure 3 Panel A (Reliability Diagram)**: Evaluates the full multicenter held-out eICU cohort ($n = 57,488$ stays across 206 hospitals; slope = 1.026, intercept = 0.077, ECE = 0.0037).
   - **Figure 3 Panel B (Decision-Curve Analysis)**: Evaluated strictly on the complete-case paired held-out subset with available APACHE-IVa predictions ($n = 39,920$ stays across 151 hospitals), demonstrating consistent net-benefit superiority across operational thresholds $1\%$ to $35\%$.
5. **Minimal Outcome Distortion in Unmeasured APACHE Stays**:
   The $17,568$ held-out stays without APACHE scores (30.6%) occurred in non-scoring ICUs and step-down units; their observed mortality was $12.38\%$ vs. $12.41\%$ in the scored subset, demonstrating minimal outcome-rate distortion while acknowledging that unmeasured clinical confounders in non-submitting centers cannot be completely excluded.


### Hyperparameter Optimization & 5-Fold Cross-Validation Audit (`run_hyperparameter_search_and_cv.py`)

To eliminate any ambiguity regarding hyperparameter optimization and to prove complete test-set isolation (Reviewer #3, R3-C6), the repository provides the standalone 5-fold cross-validation execution script `src/run_hyperparameter_search_and_cv.py` along with full candidate evaluation records:

1. **Protocolized Candidate Search Grid**:
   Explored 6 distinct architectural configurations (CONFIG-1 through CONFIG-6) across tree capacity (`num_leaves` from 15 to 127, `max_depth` from 5 to 8 and unconstrained), learning rates (0.015 to 0.05), boosting rounds (500 to 1,400), subsampling/colsample ratios (0.80 to 0.90), minimum child samples (50 to 100), and L1/L2 regularization ($\alpha \in [0.0, 0.2]$, $\lambda \in [1.0, 5.0]$).
2. **Stratified 5-Fold Partition Boundary**:
   Tuning was conducted strictly on the pooled development training partition ($N = 34,123$: MIMIC-IV development training fold $n = 26,937$ plus eICU local update subset $n = 7,186$).
   - The disjoint eICU calibration partition ($n = 7,186$) was completely sealed.
   - The primary held-out benchmark ($n = 57,488$, Protocol A) and the 167 held-out hospitals ($n = 57,536$, Protocol B) were strictly sequestered and untouched. Zero held-out test data was accessed or evaluated during hyperparameter search.
3. **Dynamic Class-Imbalance Weighting**:
   Across all candidate configurations, tree-split gradients dynamically penalized positive (mortality) instances via:
   $$\text{scale\_pos\_weight} = \frac{N_{\text{train}} - N_{\text{events}}}{N_{\text{events}}} \approx 7.06$$
   to counteract the ~12.4% mortality incidence prior to Platt recalibration.
4. **Locked Model Selection Rationale**:
   While ultra-deep heads with unconstrained depth (CONFIG-3 and CONFIG-6, 127 leaves) achieved slightly higher raw development CV AUROC (0.9585 vs. 0.9566), they presented higher risk of cross-center overfitting on unseen hospitals. CONFIG-4 (Locked) was selected and locked as the definitive performance head because it combined high discrimination (mean 5-fold CV AUROC $0.9566 \pm 0.0018$, Brier 0.0396) with conservative tree depth (`num_leaves = 31`, `max_depth = 6`), a robust leaf sample floor (`min_child_samples = 80`), and strong L2 regularization (`reg_lambda = 2.0`), maximizing generalizability across unseen hospital environments.
5. **Audit Artifacts**:
   - `results/table_hyperparameter_cv_audit.csv`: Detailed fold-by-fold and aggregate metrics for all 6 candidate configurations.
   - `results/hyperparameter_cv_summary.json`: Machine-readable summary with execution timestamps, sample sizes, and fold AUROC/Brier vectors.

## Privacy Boundary

Do not commit or redistribute:

- original MIMIC-IV or eICU-CRD database files;
- patient-level extracted feature tables;
- row-level predictions or identifiers;
- credential files, server passwords, tokens, or data-use agreement materials.

Only aggregate tables and non-identifying summaries are included here.
