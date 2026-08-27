#!/usr/bin/env python3
"""V14 domain-update nutrition and APACHE lock.

This audit uses the V13 harmonized feature tables and evaluates whether the
clean nested eICU domain-update head remains above APACHE-IV/IVa while retaining
honest nutrition-related incremental-value checks.

APACHE predictions are used only as external comparators on the same eICU
held-out subset. No APACHE score, APS, or APACHE predicted mortality is used as
an input feature.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


SERVER_ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
ROOT = SERVER_ROOT if SERVER_ROOT.exists() else LOCAL_ROOT / "05-results"
EICU = Path("/root/autodl-tmp/eicu/base")
V13 = ROOT / "v13_strong_clinical_feature_upgrade"
OUT = ROOT / "v14_domain_update_nutrition_apache_lock"

V12_NUTRITION_CONSTRUCT = {
    "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
}
EXPLICIT_NUTRITION = {
    "bmi", "enteral_24h", "parenteral_24h", "albumin", "glucose",
    "albumin_missing", "glucose_missing",
}
NUTRITION_SUPPORT = {"enteral_24h", "parenteral_24h"}
NUTRITION_METABOLIC_LABS = {
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
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    return total


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
        objective="binary",
        random_state=seed,
        n_jobs=8,
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


def paired_bootstrap_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    fn = auc if metric == "auc" else lambda yy, pp: float(np.mean((pp - yy) ** 2))
    for _ in range(500):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if metric == "auc" and (yy.sum() == 0 or yy.sum() == len(yy)):
            continue
        vals.append(fn(yy, p_a[idx]) - fn(yy, p_b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "metric": metric,
        "estimate": float(fn(y, p_a) - fn(y, p_b)),
        "ci_95": [float(lo), float(hi)],
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
        block = {"APACHE": metric_block(y[idx_arr], ap_arr), "coverage_n": int(len(idx_arr))}
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
    med = impute_fit(X_m_raw)
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_nested_held = ~(e_update_fit | e_update_cal)

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
        "full_all_features": all_features,
        "without_v12_nutrition_construct": [f for f in all_features if f not in V12_NUTRITION_CONSTRUCT],
        "without_explicit_nutrition": [f for f in all_features if f not in EXPLICIT_NUTRITION],
        "without_nutrition_support": [f for f in all_features if f not in NUTRITION_SUPPORT],
        "without_nutrition_metabolic_labs": [f for f in all_features if f not in NUTRITION_METABOLIC_LABS],
    }

    result = {
        "protocol": "V14_DOMAIN_UPDATE_NUTRITION_APACHE_LOCK",
        "status_date": "2026-08-26",
        "red_lines": [
            "APACHE predictions are comparators only and are never model inputs.",
            "Nested eICU update, Platt calibration, and held-out evaluation use disjoint eICU partitions.",
            "Nutrition incremental value is reported as observed, including negative or null increments.",
        ],
        "feature_sets": {},
        "paired_incremental_value": {},
    }
    pred_cache = {}
    for i, (name, features) in enumerate(feature_sets.items()):
        print(f"evaluating {name} ({len(features)} features)", flush=True)
        block = evaluate_feature_set(name, features, seed=20260840 + i)
        pred_cache[name] = block.pop("predictions_for_increment")
        result["feature_sets"][name] = block

    full = pred_cache["full_all_features"]
    y = np.asarray(full["y"], dtype=int)
    for other in [x for x in feature_sets if x != "full_all_features"]:
        result["paired_incremental_value"][f"full_minus_{other}"] = {
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
            rows.append({"feature_set": name, "model": model, **metrics})
    with (OUT / "table_v14_nested_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    inc_rows = []
    for comparison, block in result["paired_incremental_value"].items():
        row = {"comparison": comparison}
        for key, val in block.items():
            row[f"{key}_estimate"] = val["estimate"]
            row[f"{key}_ci_low"] = val["ci_95"][0]
            row[f"{key}_ci_high"] = val["ci_95"][1]
        inc_rows.append(row)
    with (OUT / "table_v14_paired_incremental_value.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(inc_rows[0]))
        writer.writeheader()
        writer.writerows(inc_rows)

    full_block = result["feature_sets"]["full_all_features"]["nested_heldout"]["combined_nested_platt"]
    md = [
        "# V14 Domain-Update Nutrition and APACHE Lock",
        "",
        "Status date: 2026-08-26",
        "",
        "## Primary Locked Result",
        "",
        f"- Full nested combined domain-update head: AUROC {full_block['auc']:.3f}, "
        f"Brier {full_block['brier']:.3f}, ECE {full_block['ece_q10']:.3f} on held-out eICU.",
        "",
        "## Nutrition Increment Boundary",
        "",
        "The paired incremental-value table reports observed full-minus-reduced differences. "
        "Positive AUROC differences support incremental discrimination; negative differences "
        "must be written as no observed incremental discrimination for that feature definition.",
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
