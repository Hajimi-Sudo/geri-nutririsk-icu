# V13b Domain-Update Calibration

Status date: 2026-08-26

## Main held-out eICU
- source_only_platt: AUROC 0.807, Brier 0.085, ECE 0.014.
- eicu_update_same_pool_platt: AUROC 0.877, Brier 0.081, ECE 0.065.
- combined_update_same_pool_platt: AUROC 0.880, Brier 0.077, ECE 0.047.

## Nested held-out eICU
- eicu_update_nested_platt: AUROC 0.868, Brier 0.075, ECE 0.005.
- combined_update_nested_platt: AUROC 0.873, Brier 0.073, ECE 0.004.
- source_only_platt_on_nested_heldout: AUROC 0.807, Brier 0.085, ECE 0.014.

## Claim Boundary

Same-pool update can be reported only as a local domain-updated performance head. Nested update is the cleaner calibrated domain-update sensitivity because eICU update fitting and Platt calibration use disjoint calibration-pool halves. Neither is source-only transport superiority.
