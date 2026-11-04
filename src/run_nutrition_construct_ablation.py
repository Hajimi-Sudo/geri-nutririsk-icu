from sklearn.metrics import roc_auc_score
#!/usr/bin/env python3
"""
GERI-NutriRisk-ICU: Dissected Nutrition-Construct Ablation Experiment (EXP-004)
=============================================================================

This standalone script reproduces and audits the fine-grained feature ablation
evaluating the independent prognostic contribution of explicit nutritional reserve
(Albumin and BMI) versus acute metabolic laboratories on the multicenter held-out
eICU cohort (n = 57,488 stays across 206 hospitals).

Auditing Context & Clarification on Historical V14 vs Revision EXP-004:
----------------------------------------------------------------------
During the first round of review, Reviewer #2 (R2-C5) noted that acute metabolic
derangements (creatinine, BUN, sodium, etc.) were grouped alongside nutritional reserve,
and requested explicitly separating albumin and BMI from general severity laboratories.

In response, EXP-004 was introduced during revision to enforce a strict, non-overlapping
feature dissection:
  1. Full Model: 83 cross-database harmonized features.
  2. Without Explicit Nutrition Reserve: Removes {"bmi", "albumin", "albumin_missing"}
     -> Exactly 80 features remaining.
  3. Without Nutrition Support: Removes {"enteral_24h", "parenteral_24h"}
     -> Exactly 81 features remaining.
  4. Without Explicit Nutrition & Support: Removes both sets (5 features)
     -> Exactly 78 features remaining.
  5. Without Acute Metabolic Laboratories: Removes 7 acute labs and their missingness flags
     {"glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc", ...}
     (14 features total; strictly EXCLUDES albumin and BMI)
     -> Exactly 69 features remaining.

Additionally, for full audit transparency against the pre-revision historical script
(run_v14_domain_update_nutrition_apache_lock.py), this script also benchmarks the
historical coarse bundling (76 features and 67 features) side-by-side.

Output Files:
  - exp004_nutrition_ablation.json
  - table_exp004_nutrition_construct_ablation.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.stats import norm


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def compute_auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def compute_brier(y, p):
    return float(np.mean((p - y) ** 2))


def compute_ece(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    total = 0.0
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def compute_paired_delong_pvalue(y, p_full, p_abl):
    """
    Computes fast asymptotic paired comparison between full model and ablated model.
    """
    y = np.asarray(y, dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if m == 0 or n == 0:
        return 1.0

    # Structural components
    v10_f = np.mean(p_full[pos_idx, None] > p_full[None, neg_idx], axis=1) + 0.5 * np.mean(p_full[pos_idx, None] == p_full[None, neg_idx], axis=1)
    v01_f = np.mean(p_full[pos_idx, None] < p_full[None, neg_idx], axis=0) + 0.5 * np.mean(p_full[pos_idx, None] == p_full[None, neg_idx], axis=0)
    v10_a = np.mean(p_abl[pos_idx, None] > p_abl[None, neg_idx], axis=1) + 0.5 * np.mean(p_abl[pos_idx, None] == p_abl[None, neg_idx], axis=1)
    v01_a = np.mean(p_abl[pos_idx, None] < p_abl[None, neg_idx], axis=0) + 0.5 * np.mean(p_abl[pos_idx, None] == p_abl[None, neg_idx], axis=0)

    theta_f = np.mean(v10_f)
    theta_a = np.mean(v10_a)
    diff = theta_f - theta_a

    s10 = np.cov(v10_f, v10_a)
    s01 = np.cov(v01_f, v01_a)
    sigma2 = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n

    if sigma2 <= 0:
        return 1.0
    z = diff / np.sqrt(sigma2)
    p_val = 2 * (1.0 - norm.cdf(abs(z)))
    return float(p_val)


def fit_platt(score, y, lr=0.035, epochs=4500):
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w


def apply_platt(p, w):
    return sigmoid(w[0] + w[1] * logit(p))


def load_features(path: Path):
    ids, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        features = [c for c in reader.fieldnames if c not in {"id", "y"}]
        for row in reader:
            ids.append(row["id"])
            y.append(int(float(row["y"])))
            X.append([float(row[c]) if row.get(c, "") != "" else np.nan for c in features])
    return ids, np.asarray(X, dtype=float), np.asarray(y, dtype=int), features


def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X, med):
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def fit_lgbm(X, y, seed, n_estimators=1400, lr=0.02):
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_samples=80,
        reg_lambda=2.0,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run GERI-NutriRisk-ICU Dissected Nutrition Ablation (EXP-004)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing v13 CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--n-estimators", type=int, default=1400, help="Number of trees")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="LGBM learning rate")
    parser.add_argument("--seed", type=int, default=20261840, help="Random seed")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir:
        v13_dir = Path(args.data_dir)
    else:
        candidates = [
            Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v13_strong_clinical_feature_upgrade"),
            Path(__file__).resolve().parents[1] / "05-results" / "v13_strong_clinical_feature_upgrade",
            Path("/root/code/rh-workspace/medicine/Frontiers in Nutrition/mimic-eicu-geriatric-nutrition-risk/05-results/v13_strong_clinical_feature_upgrade"),
        ]
        v13_dir = next((c for c in candidates if (c / "eicu_v13_features_server_only.csv").exists()), None)
        if v13_dir is None:
            raise FileNotFoundError("Could not find v13 feature CSVs.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/revision_experiments")
        if not out_dir.parent.exists():
            out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GERI-NutriRisk-ICU: Dissected Nutrition-Construct Ablation (EXP-004)")
    print(f"Data directory:   {v13_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Trees: {args.n_estimators}, LR: {args.learning_rate}, Seed: {args.seed}")
    print("=" * 80)

    t_start = time.time()

    # Load data
    mimic_csv = v13_dir / "mimic_v13_features_server_only.csv"
    eicu_csv = v13_dir / "eicu_v13_features_server_only.csv"
    print("Loading MIMIC-IV development features...")
    mimic_ids, X_m_raw, y_m, _ = load_features(mimic_csv)
    print(f"Loaded MIMIC: {len(y_m)} records.")

    print("Loading eICU-CRD external evaluation features...")
    eicu_ids, X_e_raw, y_e, features = load_features(eicu_csv)
    print(f"Loaded eICU: {len(y_e)} records, Total harmonized features: {len(features)}")
    assert len(features) == 83, f"Expected 83 features, found {len(features)}"

    # Standard nested protocol partitions
    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_held = ~(e_update_fit | e_update_cal)

    # Enforce train-only median imputation: fit strictly on MIMIC-IV development training fold
    print("Fitting median imputation on MIMIC-IV train partition and transforming...")
    med = impute_fit(X_m_raw[m_train])
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    print(f"Cohort splits: MIMIC Train = {m_train.sum()}, eICU Update Fit = {e_update_fit.sum()}, eICU Cal = {e_update_cal.sum()}, eICU Held-Out = {e_held.sum()}")

    # Define Dissected Feature Sets (EXP-004)
    # 1. Explicit nutritional reserve: albumin, BMI, and albumin missingness indicator
    EXPLICIT_NUTRITION = {"bmi", "albumin", "albumin_missing"}

    # 2. Nutrition support indicators: enteral and parenteral nutrition in 24h
    NUTRITION_SUPPORT = {"enteral_24h", "parenteral_24h"}

    # 3. Acute metabolic / organ severity laboratories (strictly excluding albumin and BMI)
    METABOLIC_LABS = {
        "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
        "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
        "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    }

    # 4. Historical coarse sets from pre-revision run_v14 script (for side-by-side audit)
    HISTORICAL_V14_EXPLICIT = {
        "bmi", "enteral_24h", "parenteral_24h", "albumin", "glucose",
        "albumin_missing", "glucose_missing",
    }
    HISTORICAL_V14_METABOLIC = {
        "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
        "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
        "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    }

    configs = {
        "full_model": {
            "label": "Full GERI-NutriRisk-ICU Model",
            "features": features,
            "category": "Reference Locked Model",
            "description": "All 83 harmonized cross-database features",
        },
        "without_explicit_nutrition": {
            "label": "Ablation: Without Explicit Nutrition Reserve",
            "features": [f for f in features if f not in EXPLICIT_NUTRITION],
            "category": "Revision EXP-004 (Strict Dissection)",
            "description": "Removes Albumin, BMI, and Albumin missingness (3 variables -> 80 features)",
        },
        "without_nutrition_support": {
            "label": "Ablation: Without Nutrition Support",
            "features": [f for f in features if f not in NUTRITION_SUPPORT],
            "category": "Revision EXP-004 (Strict Dissection)",
            "description": "Removes Enteral and Parenteral nutrition in 24h (2 variables -> 81 features)",
        },
        "without_explicit_and_support": {
            "label": "Ablation: Without Explicit Nutrition & Support",
            "features": [f for f in features if f not in (EXPLICIT_NUTRITION | NUTRITION_SUPPORT)],
            "category": "Revision EXP-004 (Strict Dissection)",
            "description": "Removes both explicit nutrition and support indicators (5 variables -> 78 features)",
        },
        "without_metabolic_labs": {
            "label": "Ablation: Without Acute Metabolic Laboratories",
            "features": [f for f in features if f not in METABOLIC_LABS],
            "category": "Revision EXP-004 (Strict Dissection)",
            "description": "Removes 7 acute metabolic labs and 7 missingness flags (14 variables -> 69 features; excludes albumin/BMI)",
        },
        "without_v12_nutrition_construct": {
            "label": "Ablation: Without Full Composite Nutrition Construct",
            "features": [f for f in features if f not in (EXPLICIT_NUTRITION | NUTRITION_SUPPORT | METABOLIC_LABS)],
            "category": "Broad Nutrition Composite Benchmark",
            "description": "Removes all 19 nutritional and metabolic laboratory variables and flags combined (19 variables -> 64 features)",
        },
        "historical_v14_without_explicit": {
            "label": "Historical V14 Coarse: Without Broad Nutrition Set",
            "features": [f for f in features if f not in HISTORICAL_V14_EXPLICIT],
            "category": "Historical V14 Audit Benchmark",
            "description": "Historical pre-revision V14 grouping including glucose & support (7 variables -> 76 features)",
        },
        "historical_v14_without_metabolic": {
            "label": "Historical V14 Coarse: Without Broad Metabolic Labs",
            "features": [f for f in features if f not in HISTORICAL_V14_METABOLIC],
            "category": "Historical V14 Audit Benchmark",
            "description": "Historical pre-revision V14 grouping including albumin & glucose (16 variables -> 67 features)",
        },
    }

    results = {}
    csv_rows = []
    p_preds = {}

    print("\n--- Training and Evaluating Ablation Models ---")
    for key, cfg in configs.items():
        t0 = time.time()
        f_list = cfg["features"]
        n_feat = len(f_list)
        f_idx = [features.index(c) for c in f_list]

        # Fit model on MIMIC train + eICU domain update fit
        X_train_sub = np.vstack([X_m[m_train][:, f_idx], X_e[e_update_fit][:, f_idx]])
        y_train_sub = np.concatenate([y_m[m_train], y_e[e_update_fit]])

        model = fit_lgbm(
            X_train_sub,
            y_train_sub,
            seed=args.seed,
            n_estimators=args.n_estimators,
            lr=args.learning_rate,
        )

        # Predict raw probabilities on eICU
        p_raw = model.predict_proba(X_e[:, f_idx])[:, 1]

        # Disjoint Platt calibration on eICU update calibration partition
        w_cal = fit_platt(logit(p_raw[e_update_cal]), y_e[e_update_cal])
        p_cal = apply_platt(p_raw, w_cal)
        p_preds[key] = p_cal[e_held]

        # Evaluate on held-out stays
        y_eval = y_e[e_held]
        auc_val = compute_auc(y_eval, p_cal[e_held])
        brier_val = compute_brier(y_eval, p_cal[e_held])
        ece_val = compute_ece(y_eval, p_cal[e_held])
        elapsed = time.time() - t0

        results[key] = {
            "label": cfg["label"],
            "category": cfg["category"],
            "description": cfg["description"],
            "n_features": n_feat,
            "metrics": {
                "n": int(len(y_eval)),
                "events": int(y_eval.sum()),
                "event_rate": float(y_eval.mean()),
                "auc": float(auc_val),
                "brier": float(brier_val),
                "ece_q10": float(ece_val),
            },
            "runtime_seconds": float(elapsed),
        }
        print(f"[{key:32s}] Features: {n_feat:2d} | AUROC: {auc_val:.4f} | Brier: {brier_val:.4f} | ECE: {ece_val:.4f} | Time: {elapsed:.1f}s")

    # Compute delta AUROC and p-values relative to full model
    p_full = p_preds["full_model"]
    auc_full = results["full_model"]["metrics"]["auc"]
    brier_full = results["full_model"]["metrics"]["brier"]

    for key, cfg in configs.items():
        res = results[key]
        auc_cur = res["metrics"]["auc"]
        brier_cur = res["metrics"]["brier"]
        delta_auc = auc_cur - auc_full
        delta_brier = brier_cur - brier_full
        if key == "full_model":
            p_val = 1.0
        else:
            p_val = compute_paired_delong_pvalue(y_e[e_held], p_full, p_preds[key])
        
        res["delta_auc"] = float(delta_auc)
        res["delta_brier"] = float(delta_brier)
        res["p_value_vs_full"] = float(p_val)

        csv_rows.append({
            "model_key": key,
            "category": cfg["category"],
            "description": cfg["description"],
            "n_features": res["n_features"],
            "held_out_n": res["metrics"]["n"],
            "held_out_events": res["metrics"]["events"],
            "auroc": f"{auc_cur:.4f}",
            "delta_auroc": f"{delta_auc:+.4f}" if key != "full_model" else "Reference",
            "p_value_vs_full": f"{p_val:.4e}" if key != "full_model" else "N/A",
            "brier_score": f"{brier_cur:.4f}",
            "delta_brier": f"{delta_brier:+.4f}" if key != "full_model" else "Reference",
            "ece_q10": f"{res['metrics']['ece_q10']:.4f}",
        })

    # Save outputs
    json_path = out_dir / "exp004_nutrition_ablation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    csv_path = out_dir / "table_exp004_nutrition_construct_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV table to:   {csv_path}")

    total_time = time.time() - t_start
    print(f"\nAll ablation experiments completed successfully in {total_time:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
