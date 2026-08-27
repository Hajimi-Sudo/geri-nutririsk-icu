# V16 Hospital-Cluster Robustness

Status date: 2026-08-26

Held-out eICU hospitals: 206; held-out stays: 57488.

## Main Robustness
- all_heldout_hospitals: AUROC 0.872, Brier 0.073, n 57488.
- excluding_top10_hospitals_by_n: AUROC 0.870, Brier 0.072, n 44777.
- hospitals_with_n_ge_100: AUROC 0.874, Brier 0.073, n 54833.
- hospitals_with_n_ge_200: AUROC 0.875, Brier 0.073, n 47517.

## APACHE Same-Subset Hospital-Cluster Bootstrap
- APACHE-IV: model-minus-APACHE AUROC mean 0.025, 95% cluster CI 0.019 to 0.032.
- APACHE-IVa: model-minus-APACHE AUROC mean 0.021, 95% cluster CI 0.015 to 0.028.

## Boundary

Hospital-cluster results assess robustness across the multicenter eICU validation dataset. They do not substitute for prospective implementation validation.
