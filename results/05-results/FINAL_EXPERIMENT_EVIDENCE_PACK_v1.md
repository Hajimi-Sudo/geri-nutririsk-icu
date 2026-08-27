# Final Experiment Evidence Pack

Status date: 2026-08-26

## Primary Framework

Name: GERI-NutriRisk-ICU.

Primary model: V3 plus severity, a standardized logistic framework combining demographics, BMI, early nutrition-support indicators, nutrition/metabolic laboratory markers, vital signs, organ-support features, urine output, and Glasgow Coma Scale.

## Completed Experimental Surfaces

- MIMIC-IV development and internal test cohort.
- eICU-CRD external validation cohort.
- Feature-family ablations from V0 through V3, nutrition/metabolic core, and severity-only baseline.
- Domain recalibration using a prespecified eICU calibration split and held-out eICU evaluation.
- eICU subgroup robustness by sex, age group, albumin availability, ventilation status, and enteral nutrition status.
- ROC, calibration, decision-curve, and standardized-coefficient visualizations.
- One thousand bootstrap resamples for primary AUROC, Brier score, and ECE.

## Primary Metrics

- MIMIC-IV internal test: AUROC 0.830 (0.815-0.843); Brier 0.090 (0.085-0.095); ECE 0.015 (0.011-0.023).
- eICU external full original model: AUROC 0.727 (0.722-0.733); Brier 0.155 (0.153-0.156); ECE 0.180 (0.177-0.182).
- eICU held-out after Platt recalibration: AUROC 0.728 (0.722-0.735); Brier 0.099 (0.097-0.101); ECE 0.009 (0.008-0.012).

## Key Interpretation

The strongest defensible claim is transportable, externally validated risk stratification with substantial calibration improvement after external-domain recalibration. The severity-only baseline is close to the full model externally, so the manuscript should not claim that nutrition variables alone explain performance.

## Nutrition Construct Increment

- Full V3 minus non-nutrition clinical baseline external AUROC difference: 0.00559 (95% CI 0.00198 to 0.00901).
- Full V3 minus model without nutrition/metabolic laboratories external AUROC difference: -0.00501 (95% CI -0.00840 to -0.00198).
- Interpretation: nutrition-related features support construct framing and care-context interpretation, but they should not be presented as the dominant source of external discrimination.

## Claim Boundary

Bootstrap is row/stay-level on server-only feature tables; no eICU labels used for MIMIC model fitting; eICU labels used only in prespecified Platt domain-recalibration split and held-out evaluation.

The study remains retrospective and cannot claim prospective clinical benefit, treatment effect, or deployment readiness.
## V7 Reviewer-Facing Extensions

- Model benchmark: the primary standardized logistic GERI-NutriRisk-ICU model achieved eICU external AUROC 0.727, whereas a weighted one-hidden-layer MLP achieved external AUROC 0.654 despite higher internal AUROC.
- Five split rotations: eICU original AUROC mean (SD) 0.728 (0.002); eICU held-out Platt Brier mean (SD) 0.099 (0.000).
- Added operating characteristics at prespecified thresholds and bootstrap subgroup confidence intervals for the eICU external cohort.
## V8 SOTA Tabular Benchmark

- SOTA comparator queue completed: XGBoost 3.4.1, LightGBM 4.7.0, and CatBoost 1.2.10.
- LightGBM was the strongest external-discrimination comparator: MIMIC internal AUROC 0.882; eICU external AUROC 0.793.
- Original LightGBM external calibration was poor (Brier 0.291; ECE 0.386), but prespecified eICU-domain Platt recalibration improved held-out Brier to 0.088 and ECE to 0.021.
- Logistic GERI-NutriRisk-ICU remains the most transparent coefficient-level anchor (eICU external AUROC 0.727; held-out Platt Brier 0.099); manuscript framing should present GERI-NutriRisk-ICU as a framework with LightGBM as the strongest SOTA performance comparator.
## V9 LightGBM Primary Performance Head

- Final upgraded framework interpretation: GERI-NutriRisk-ICU contains a harmonized feature schema, leakage-controlled multicohort protocol, transparent logistic anchor model, SOTA tabular performance head, external-domain recalibration, operating-profile layer, and explanation layer.
- LightGBM primary performance head: MIMIC internal AUROC 0.882 (0.870-0.893); eICU external AUROC 0.793 (0.788-0.798); eICU held-out Platt AUROC 0.794 (0.788-0.799).
- Recalibrated LightGBM eICU held-out calibration: Brier 0.088 (0.087-0.090); ECE 0.021 (0.019-0.023).
- Leading LightGBM contribution features: urine_24h_ml, gcs_24h, map, resp_rate, bun, bmi, heart_rate, wbc.
- The manuscript should not claim treatment benefit or standalone nutrition causality; it can claim externally validated, recalibrated, explainable nutrition-risk-relevant stratification.
## V10 SOTA Completion And Model Lock

- Bounded SOTA completion added HistGradientBoosting, balanced random forest, ExtraTrees, and MLP to the existing logistic, XGBoost, LightGBM, and CatBoost queue.
- HistGradientBoosting had the numerically highest eICU held-out Platt AUROC (0.794), but the advantage over LightGBM was only 0.00010, far below the prespecified 0.015 material replacement threshold.
- LightGBM retained better recalibrated Brier score than HistGradientBoosting (0.088 vs 0.089) and remains locked as the primary performance head.

## V11 TabNet Neural SOTA Comparator

- TabNet was run on GPU as a bounded neural tabular comparator with MIMIC-only fitting/early stopping and eICU external evaluation.
- TabNet did not improve transportability: eICU external AUROC 0.636; eICU held-out Platt AUROC 0.636; held-out Platt Brier 0.108.
- The final model decision is to keep LightGBM as the GERI-NutriRisk-ICU performance head and report logistic regression as the transparent clinical anchor.

## V12 Review Repair Package

- Patient-level split audit passed: MIMIC feature table contained 33,671 rows and 33,671 unique subject IDs, with zero subject overlap between development and internal-test partitions; eICU feature table contained 71,861 rows and 71,861 unique `uniquepid` values, with zero `uniquepid` overlap between recalibration and held-out evaluation partitions.
- eICU covered 207 hospitals, supporting multicenter external-validation language while still requiring careful wording because recalibration uses eICU labels.
- LightGBM feature-set sensitivity showed that the full 39-feature model had held-out recalibrated eICU AUROC 0.794, compared with 0.784 without nutrition-support indicators, 0.771 without nutrition/metabolic laboratories, 0.778 without all nutrition-construct variables, and 0.706 with severity-only features.
- Paired bootstrap differences supported a modest but measurable nutrition-construct contribution: full minus no nutrition-support AUROC difference 0.0099 (95% CI 0.0089 to 0.0109); full minus no nutrition/metabolic laboratories 0.0224 (0.0184 to 0.0262); full minus no all-nutrition construct 0.0156 (0.0119 to 0.0195).
- Clinical proxy baselines were added for review transparency. eICU APACHE-IV and APACHE-IVa predicted hospital mortality, available for 49,970 eICU stays, achieved AUROC 0.847 and 0.851, respectively. These results mean the manuscript should not claim superiority over APACHE-style ICU prognostic systems; instead, it should frame GERI-NutriRisk-ICU as an auditable, cross-database, nutrition-aware framework using harmonized features that remain reconstructable in both MIMIC-IV and eICU.
- SOFA-lite and mNUTRIC-lite proxies were also generated on held-out eICU. SOFA-lite proxy AUROC was 0.779 and mNUTRIC-lite proxy AUROC was 0.670. Both must be labeled as proxies, not exact published-score reconstructions.
- Reproducibility/log manifest now identifies three server-side run logs and the result/code files used for V0-V12.

Updated claim boundary: the study can claim externally validated and recalibrated nutrition-risk-relevant mortality stratification with modest incremental value from nutrition-related features over harmonized non-nutrition feature sets. It cannot claim malnutrition detection, nutrition-treatment benefit, clinical deployment readiness, or superiority to APACHE predicted mortality.

## V13 Strong Clinical-Feature Upgrade

- The harmonized feature schema was expanded from 39 to 83 cross-database
  features, adding additional chemistry, blood gas/perfusion variables,
  coagulation/hematology markers, ICU/admission context indicators, and derived
  PaO2/FiO2 ratio while preserving the rule that APACHE outputs were not used as
  model inputs.
- Source-only V13 performance improved to held-out recalibrated eICU AUROC
  0.810, Brier 0.084, and ECE 0.013, but source-only performance remained below
  APACHE-IV/IVa on the APACHE-available same subset.
- Domain-updated V13 heads exceeded APACHE discrimination but required a cleaner
  split because same-pool updating/calibration could be viewed as optimistic.

## V14 Domain-Update Nutrition and APACHE Lock

- V14 locked the primary model as the full 83-feature combined MIMIC plus eICU
  nested domain-update LightGBM head with Platt calibration.
- Nested split rule: eICU rows with index mod 10 equal to 0 were used for local
  model updating, rows with index mod 10 equal to 5 were used for Platt
  calibration, and all remaining eICU rows were held out for final evaluation.
- Held-out eICU performance: AUROC 0.872, Brier 0.073, ECE 0.004 across 57,488
  stays and 7,130 deaths.
- Same APACHE-available held-out subset:
  - APACHE-IV AUROC 0.846, Brier 0.085, ECE 0.046; V14 model AUROC 0.871,
    Brier 0.074, ECE 0.009. Paired AUROC difference 0.0252
    (95% CI 0.0201 to 0.0305).
  - APACHE-IVa AUROC 0.850, Brier 0.082, ECE 0.029; V14 model AUROC 0.871,
    Brier 0.074, ECE 0.009. Paired AUROC difference 0.0209
    (95% CI 0.0158 to 0.0260).
- Nutrition incremental-value check:
  - In source-only transport, nutrition-related features did not improve
    discrimination and should not be written as source-only performance drivers.
  - In the nested domain-updated model, full minus no all-nutrition construct
    showed AUROC difference 0.0049 (95% CI 0.0029 to 0.0071) and Brier
    difference -0.0011 (95% CI -0.0014 to -0.0007).
  - Full minus no nutrition/metabolic laboratories showed AUROC difference
    0.0059 (95% CI 0.0041 to 0.0077) and Brier difference -0.0012
    (95% CI -0.0015 to -0.0008).
  - Early nutrition-support indicators alone had no stable incremental
    discrimination.

## V15 V14 Explainability

- Server-run explainability was completed for the V14 locked domain-update model.
- Leading feature domains by permutation AUROC drop were organ support and
  neurologic severity (0.0627), vital signs (0.0482), blood gas/perfusion
  (0.0199), nutrition/metabolic laboratories (0.0160), and additional chemistry
  (0.0096).
- Leading gain features included GCS, calcium, BUN, respiratory rate, urine
  output, lactate, heart rate, cardiac ICU type, pH, BMI, anion gap, and
  platelet count.
- The explanation layer supports model interpretability and nutrition-aware
  clinical interpretation, not causal claims.

## V16 Hospital-Cluster Robustness

- Server-run hospital robustness analysis covered 206 eICU hospitals and 57,488
  held-out stays.
- Main held-out model performance remained stable after excluding the ten
  largest held-out hospitals: AUROC 0.870 and Brier 0.072.
- Restricting to hospitals with at least 100 held-out stays yielded AUROC 0.874;
  restricting to hospitals with at least 200 held-out stays yielded AUROC 0.875.
- Hospital-cluster bootstrap confirmed the same-subset APACHE advantage:
  - versus APACHE-IV: model-minus-APACHE AUROC mean 0.025, 95% cluster CI
    0.019 to 0.032; Brier difference mean -0.0107.
  - versus APACHE-IVa: model-minus-APACHE AUROC mean 0.021, 95% cluster CI
    0.015 to 0.028; Brier difference mean -0.0080.

## Final Model Decision After V16

The manuscript should now use the V14 locked domain-updated GERI-NutriRisk-ICU
head as the primary model, with V16 hospital-cluster robustness as a major
validation surface. The source-only MIMIC-to-eICU model should be retained as a
secondary transportability analysis. The APACHE comparison can be reported as a
same-subset contextual clinical benchmark, but the wording must specify that the
APACHE-superior result applies to the nested local domain-update head rather
than the source-only model.
