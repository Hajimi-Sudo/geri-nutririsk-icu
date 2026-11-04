#!/usr/bin/env python3
"""V14 domain-update nutrition and APACHE lock.

This audit uses the V13 harmonized feature tables and evaluates whether the
clean nested eICU domain-update head remains above APACHE-IV/IVa while retaining
honest nutrition-related incremental-value checks.

APACHE predictions are used only as external comparators on the same eICU
held-out subset. No APACHE score, APS, or APACHE predicted mortality is used as
an input feature.

Harmonized Dissected Feature Sets (EXP-004 & V14 Lock):
-------------------------------------------------------
Enforces strict non-overlapping clinical dissection to isolate baseline
nutritional reserve from acute multiorgan failure:
  1. Explicit Nutritional Reserve:
     {"bmi", "albumin", "albumin_missing"}
     -> 3 features dropped, 80 features remaining.
  2. Early Nutrition Support:
     {"enteral_24h", "parenteral_24h"}
     -> 2 features dropped, 81 features remaining.
  3. Explicit Nutritional Reserve + Support Combined:
     {"bmi", "albumin", "albumin_missing", "enteral_24h", "parenteral_24h"}
     -> 5 features dropped, 78 features remaining.
  4. Acute Metabolic / Organ Severity Laboratories:
     {"glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc"}
     + 7 missingness indicators (strictly EXCLUDES albumin and BMI)
     -> 14 features dropped, 69 features remaining.
  5. Full Composite Nutrition Construct:
     All 19 nutrition and metabolic features combined
     -> 19 features dropped, 64 features remaining.
  6. Historical Pre-Revision Coarse Sets (for side-by-side audit):
     Historical coarse explicit (7 dropped -> 76 remaining)
     Historical coarse metabolic (16 dropped -> 67 remaining)
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


SERVER_ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
ROOT = SERVER_ROOT if SERVER_ROOT.exists() else LOCAL_ROOT / "05-results"
EICU = Path("/root/autodl-tmp/eicu/base")
V13 = ROOT / "v13_strong_clinical_feature_upgrade"
OUT = ROOT / "v14_domain_update_nutrition_apache_lock"

# 1. Explicit nutritional reserve (albumin, BMI, albumin missingness)
EXPLICIT_NUTRITION = {"bmi", "albumin", "albumin_missing"}

# 2. Nutrition support indicators (enteral and parenteral nutrition in 24h)
NUTRITION_SUPPORT = {"enteral_24h", "parenteral_24h"}

# 3. Explicit nutritional reserve + nutrition support combined
EXPLICIT_AND_SUPPORT = {"bmi", "albumin", "albumin_missing", "enteral_24h", "parenteral_24h"}

# 4. Acute metabolic and organ severity laboratories (strictly EXCLUDES albumin and BMI)
NUTRITION_METABOLIC_LABS = {
    "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
}

# 5. Composite broad nutrition construct (all 19 nutrition and metabolic features)
V12_NUTRITION_CONSTRUCT = {
    "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
}

# 6. Historical coarse sets from pre-revision exploration (for audit transparency)
HISTORICAL_V14_EXPLICIT = {
    "bmi", "enteral_24h", "parenteral_24h", "albumin", "glucose",
    "albumin_missing", "glucose_missing",
}
HISTORICAL_V14_METABOLIC = {
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
}


def load_table(path: Path, features: list[str] | None = None):
    rows, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        all_features = [c for c in reader.fieldnames or [] if c not in {"id", "y"}]
        features = features or all_features
        for row in reader:
            rows.append({"id": row["id"]})
            y.append(int(float(row["y"])))
            X.append([float(row[c]) if row.get(c, "") != "" else np.nan for c in features])
    return rows, np.asarray(X, dtype=float), np.asarray(y, dtype=int), features


def impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_q10": ece(y, p),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    total = 0.0
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_samples=80,
        reg_lambda=2.0,
        reg_alpha=0.0,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def fit_platt(score: np.ndarray, y: np.ndarray, lr: float = 0.035, epochs: int = 4500) -> np.ndarray:
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w


def apply_platt(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(w[0] + w[1] * logit(p))


def fnum(x: str):
    try:
        return float(x)
    except Exception:
        return None


def apache_predictions() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    path = EICU / "apachePatientResult.csv.gz"
    if not path.exists():
        return out
    with gzip.open(path, "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = fnum(row.get("predictedhospitalmortality", ""))
            if p is None or not (0 <= p <= 1):
                continue
            out.setdefault(row["patientunitstayid"], {})[row["apacheversion"]] = p
    return out


def compute_paired_delong_pvalue(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if m == 0 or n == 0:
        return 1.0

    v10_a = np.mean(p_a[pos_idx, None] > p_a[None, neg_idx], axis=1) + 0.5 * np.mean(p_a[pos_idx, None] == p_a[None, neg_idx], axis=1)
    v01_a = np.mean(p_a[pos_idx, None] < p_a[None, neg_idx], axis=0) + 0.5 * np.mean(p_a[pos_idx, None] == p_a[None, neg_idx], axis=0)
    v10_b = np.mean(p_b[pos_idx, None] > p_b[None, neg_idx], axis=1) + 0.5 * np.mean(p_b[pos_idx, None] == p_b[None, neg_idx], axis=1)
    v01_b = np.mean(p_b[pos_idx, None] < p_b[None, neg_idx], axis=0) + 0.5 * np.mean(p_b[pos_idx, None] == p_b[None, neg_idx], axis=0)

    theta_a = np.mean(v10_a)
    theta_b = np.mean(v10_b)
    diff = theta_a - theta_b

    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)
    sigma2 = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n

    if sigma2 <= 0:
        return 1.0
    z = diff / np.sqrt(sigma2)
    p_val = 2 * (1.0 - norm.cdf(abs(z)))
    return float(p_val)


def paired_bootstrap_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    fn = auc if metric == "auc" else lambda yy, pp: float(np.mean((pp - yy) ** 2))
    diff_point = float(fn(y, p_a) - fn(y, p_b))
    for _ in range(500):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if metric == "auc" and (yy.sum() == 0 or yy.sum() == len(yy)):
            continue
        vals.append(fn(yy, p_a[idx]) - fn(yy, p_b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p_val = None
    if metric == "auc":
        p_val = compute_paired_delong_pvalue(y, p_a, p_b)
    return {
        "metric": metric,
        "estimate": diff_point,
        "ci_95": [float(lo), float(hi)],
        "p_value": p_val,
        "n_resamples": int(len(vals)),
    }


def same_subset(rows: list[dict], y: np.ndarray, eval_mask: np.ndarray, pred_map: dict[str, np.ndarray]) -> dict:
    ap = apache_predictions()
    out = {}
    for version in ["IV", "IVa"]:
        idx, apred = [], []
        for i, row in enumerate(rows):
            if not eval_mask[i]:
                continue
            val = ap.get(row["id"], {}).get(version)
            if val is None:
                continue
            idx.append(i)
            apred.append(val)
        if not idx:
            continue
        idx_arr = np.asarray(idx, dtype=int)
        ap_arr = np.asarray(apred, dtype=float)
        block = {"APACHE": metric_block(y[idx_arr], ap_arr)}
        for name, pred in pred_map.items():
            block[name] = metric_block(y[idx_arr], pred[idx_arr])
            block[f"{name}_minus_APACHE_auc"] = block[name]["auc"] - block["APACHE"]["auc"]
            block[f"{name}_minus_APACHE_brier"] = block[name]["brier"] - block["APACHE"]["brier"]
            block[f"{name}_minus_APACHE_auc_bootstrap"] = paired_bootstrap_diff(
                y[idx_arr], pred[idx_arr], ap_arr, "auc", seed=20260840
            )
            block[f"{name}_minus_APACHE_brier_bootstrap"] = paired_bootstrap_diff(
                y[idx_arr], pred[idx_arr], ap_arr, "brier", seed=20260841
            )
        out[f"APACHE_{version}"] = block
    return out


def evaluate_feature_set(name: str, features: list[str], seed: int) -> dict:
    eicu_rows, X_e_raw, y_e, _ = load_table(V13 / "eicu_v13_features_server_only.csv", features)
    _, X_m_raw, y_m, _ = load_table(V13 / "mimic_v13_features_server_only.csv", features)

    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_nested_held = ~(e_update_fit | e_update_cal)

    # Enforce train-only median imputation: fit strictly on MIMIC-IV development training fold
    med = impute_fit(X_m_raw[m_train])
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    source = fit_lgbm(X_m[m_train], y_m[m_train], seed)
    p_source = source.predict_proba(X_e)[:, 1]
    w_source = fit_platt(logit(p_source[e_update_cal]), y_e[e_update_cal])
    p_source_platt = apply_platt(p_source, w_source)

    combined = fit_lgbm(
        np.vstack([X_m[m_train], X_e[e_update_fit]]),
        np.concatenate([y_m[m_train], y_e[e_update_fit]]),
        seed + 1000,
    )
    p_combined = combined.predict_proba(X_e)[:, 1]
    w_combined = fit_platt(logit(p_combined[e_update_cal]), y_e[e_update_cal])
    p_combined_platt = apply_platt(p_combined, w_combined)

    preds = {
        f"{name}_source_only_platt": p_source_platt,
        f"{name}_combined_nested_platt": p_combined_platt,
    }
    return {
        "name": name,
        "feature_count": len(features),
        "features": features,
        "nested_heldout": {
            "source_only_platt": metric_block(y_e[e_nested_held], p_source_platt[e_nested_held]),
            "combined_nested_platt": metric_block(y_e[e_nested_held], p_combined_platt[e_nested_held]),
        },
        "apache_same_subset_nested_heldout": same_subset(eicu_rows, y_e, e_nested_held, preds),
        "predictions_for_increment": {
            "y": y_e[e_nested_held].tolist(),
            "source_only_platt": p_source_platt[e_nested_held].tolist(),
            "combined_nested_platt": p_combined_platt[e_nested_held].tolist(),
        },
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, _, all_features = load_table(V13 / "eicu_v13_features_server_only.csv")
    
    feature_sets = {
        "full_all_features": {
            "features": all_features,
            "category": "Reference Locked Model",
            "description": "All 83 harmonized cross-database features",
            "dropped_features": [],
        },
        "without_explicit_nutrition": {
            "features": [f for f in all_features if f not in EXPLICIT_NUTRITION],
            "category": "Revision EXP-004 Dissected Ablation",
            "description": "Removes explicit nutritional reserve (albumin, BMI, albumin missingness; 3 features dropped -> 80 features remaining)",
            "dropped_features": sorted(list(EXPLICIT_NUTRITION)),
        },
        "without_nutrition_support": {
            "features": [f for f in all_features if f not in NUTRITION_SUPPORT],
            "category": "Revision EXP-004 Dissected Ablation",
            "description": "Removes early enteral and parenteral nutrition support in 24h (2 features dropped -> 81 features remaining)",
            "dropped_features": sorted(list(NUTRITION_SUPPORT)),
        },
        "without_explicit_and_support": {
            "features": [f for f in all_features if f not in EXPLICIT_AND_SUPPORT],
            "category": "Revision EXP-004 Dissected Ablation",
            "description": "Removes explicit nutritional reserve and nutrition support combined (5 features dropped -> 78 features remaining)",
            "dropped_features": sorted(list(EXPLICIT_AND_SUPPORT)),
        },
        "without_nutrition_metabolic_labs": {
            "features": [f for f in all_features if f not in NUTRITION_METABOLIC_LABS],
            "category": "Revision EXP-004 Dissected Ablation",
            "description": "Removes acute metabolic and organ severity laboratories and missingness flags (14 features dropped -> 69 features remaining; strictly excludes albumin and BMI)",
            "dropped_features": sorted(list(NUTRITION_METABOLIC_LABS)),
        },
        "without_v12_nutrition_construct": {
            "features": [f for f in all_features if f not in V12_NUTRITION_CONSTRUCT],
            "category": "Broad Nutrition Composite Benchmark",
            "description": "Removes all 19 nutritional and metabolic laboratory variables and flags combined (19 features dropped -> 64 features remaining)",
            "dropped_features": sorted(list(V12_NUTRITION_CONSTRUCT)),
        },
        "historical_v14_without_explicit": {
            "features": [f for f in all_features if f not in HISTORICAL_V14_EXPLICIT],
            "category": "Historical Pre-Revision Audit Benchmark",
            "description": "Historical coarse nutrition set including glucose & support (7 features dropped -> 76 features remaining)",
            "dropped_features": sorted(list(HISTORICAL_V14_EXPLICIT)),
        },
        "historical_v14_without_metabolic": {
            "features": [f for f in all_features if f not in HISTORICAL_V14_METABOLIC],
            "category": "Historical Pre-Revision Audit Benchmark",
            "description": "Historical coarse metabolic set including albumin & glucose (16 features dropped -> 67 features remaining)",
            "dropped_features": sorted(list(HISTORICAL_V14_METABOLIC)),
        },
    }

    result = {
        "protocol": "V14_DOMAIN_UPDATE_NUTRITION_APACHE_LOCK",
        "status_date": "2026-08-26",
        "revision_audit_date": "2026-11-04",
        "red_lines": [
            "APACHE predictions are comparators only and are never model inputs.",
            "Nested eICU update, Platt calibration, and held-out evaluation use disjoint eICU partitions.",
            "Nutrition incremental value is reported as observed, including negative or null increments.",
            "Explicit nutritional reserve (albumin, BMI) is strictly dissected from acute metabolic laboratories.",
        ],
        "feature_sets": {},
        "paired_incremental_value": {},
    }
    pred_cache = {}
    for i, (name, cfg) in enumerate(feature_sets.items()):
        f_list = cfg["features"]
        print(f"evaluating {name} ({len(f_list)} features)", flush=True)
        block = evaluate_feature_set(name, f_list, seed=20260840 + i)
        block["category"] = cfg["category"]
        block["description"] = cfg["description"]
        block["dropped_features"] = cfg["dropped_features"]
        block["n_dropped"] = len(cfg["dropped_features"])
        pred_cache[name] = block.pop("predictions_for_increment")
        result["feature_sets"][name] = block

    full = pred_cache["full_all_features"]
    y = np.asarray(full["y"], dtype=int)
    for other, cfg in feature_sets.items():
        if other == "full_all_features":
            continue
        result["paired_incremental_value"][f"full_minus_{other}"] = {
            "comparison": f"full_minus_{other}",
            "feature_set_ablated": other,
            "category": cfg["category"],
            "description": cfg["description"],
            "n_features_remaining": len(cfg["features"]),
            "n_features_dropped": len(cfg["dropped_features"]),
            "dropped_features": ", ".join(cfg["dropped_features"]),
            "source_only_auc_diff": paired_bootstrap_diff(
                y,
                np.asarray(full["source_only_platt"]),
                np.asarray(pred_cache[other]["source_only_platt"]),
                "auc",
                seed=20260850,
            ),
            "source_only_brier_diff": paired_bootstrap_diff(
                y,
                np.asarray(full["source_only_platt"]),
                np.asarray(pred_cache[other]["source_only_platt"]),
                "brier",
                seed=20260851,
            ),
            "combined_nested_auc_diff": paired_bootstrap_diff(
                y,
                np.asarray(full["combined_nested_platt"]),
                np.asarray(pred_cache[other]["combined_nested_platt"]),
                "auc",
                seed=20260852,
            ),
            "combined_nested_brier_diff": paired_bootstrap_diff(
                y,
                np.asarray(full["combined_nested_platt"]),
                np.asarray(pred_cache[other]["combined_nested_platt"]),
                "brier",
                seed=20260853,
            ),
        }

    (OUT / "v14_domain_update_nutrition_apache_lock_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    rows = []
    for name, block in result["feature_sets"].items():
        for model, metrics in block["nested_heldout"].items():
            rows.append({
                "feature_set": name,
                "category": block["category"],
                "n_features_remaining": block["feature_count"],
                "n_features_dropped": block["n_dropped"],
                "dropped_features": ", ".join(block["dropped_features"]),
                "model": model,
                **metrics,
            })
    with (OUT / "table_v14_nested_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    inc_rows = []
    for comparison, block in result["paired_incremental_value"].items():
        row = {
            "comparison": comparison,
            "feature_set_ablated": block["feature_set_ablated"],
            "category": block["category"],
            "n_features_remaining": block["n_features_remaining"],
            "n_features_dropped": block["n_features_dropped"],
            "dropped_features": block["dropped_features"],
            "source_only_auc_diff_estimate": block["source_only_auc_diff"]["estimate"],
            "source_only_auc_diff_ci_low": block["source_only_auc_diff"]["ci_95"][0],
            "source_only_auc_diff_ci_high": block["source_only_auc_diff"]["ci_95"][1],
            "source_only_brier_diff_estimate": block["source_only_brier_diff"]["estimate"],
            "source_only_brier_diff_ci_low": block["source_only_brier_diff"]["ci_95"][0],
            "source_only_brier_diff_ci_high": block["source_only_brier_diff"]["ci_95"][1],
            "combined_nested_auc_diff_estimate": block["combined_nested_auc_diff"]["estimate"],
            "combined_nested_auc_diff_ci_low": block["combined_nested_auc_diff"]["ci_95"][0],
            "combined_nested_auc_diff_ci_high": block["combined_nested_auc_diff"]["ci_95"][1],
            "combined_nested_auc_pvalue": block["combined_nested_auc_diff"]["p_value"],
            "combined_nested_brier_diff_estimate": block["combined_nested_brier_diff"]["estimate"],
            "combined_nested_brier_diff_ci_low": block["combined_nested_brier_diff"]["ci_95"][0],
            "combined_nested_brier_diff_ci_high": block["combined_nested_brier_diff"]["ci_95"][1],
        }
        inc_rows.append(row)
    with (OUT / "table_v14_paired_incremental_value.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(inc_rows[0]))
        writer.writeheader()
        writer.writerows(inc_rows)

    full_block = result["feature_sets"]["full_all_features"]["nested_heldout"]["combined_nested_platt"]
    md = [
        "# V14 Domain-Update Nutrition and APACHE Lock (Audited & Dissected)",
        "",
        "Status date: 2026-08-26 (Audited & Synchronized: 2026-11-04)",
        "",
        "## Primary Locked Result",
        "",
        f"- Full nested combined domain-update head: AUROC {full_block['auc']:.4f}, "
        f"Brier {full_block['brier']:.4f}, ECE {full_block['ece_q10']:.4f} on held-out eICU.",
        "",
        "## Nutrition Increment Boundary (EXP-004 Dissected Ablation)",
        "",
        "The paired incremental-value table reports observed full-minus-reduced differences with "
        "500-iteration bootstrap 95% confidence intervals and paired DeLong p-values. "
        "Strictly separates explicit nutritional reserve (albumin, BMI) from acute multiorgan metabolic derangements.",
        "",
        "## APACHE Boundary",
        "",
        "APACHE-IV/IVa predicted mortality is used only as a contextual comparator on the same "
        "held-out eICU subset and is not used as a model input.",
    ]
    (OUT / "V14_DOMAIN_UPDATE_NUTRITION_APACHE_LOCK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "full_nested_combined": full_block,
        "paired_incremental_value": result["paired_incremental_value"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
