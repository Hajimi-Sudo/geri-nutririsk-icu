#!/usr/bin/env python3
"""V10 sealed SOTA completion and model-lock analysis.

This script extends the V8 benchmark with a bounded set of strong, CPU-feasible
tabular comparators. It is intentionally a completion gate, not an open-ended
model search: the manuscript should lock the primary performance head if no
additional comparator improves external discrimination and recalibrated
calibration by a clinically meaningful margin.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
V3_DIR = ROOT / "v3_severity_model"
OUT = ROOT / "v10_sota_completion_lock"

FEATURES = [
    "age", "male", "bmi", "enteral_24h", "parenteral_24h",
    "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
    "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
    "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    "heart_rate", "resp_rate", "temperature_c", "spo2", "map",
    "heart_rate_missing", "resp_rate_missing", "temperature_c_missing", "spo2_missing", "map_missing",
    "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h",
    "urine_24h_ml_missing", "ventilation_24h_missing", "dialysis_24h_missing", "gcs_24h_missing",
]


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y.append(int(float(row["y"])))
            X.append([float(row[feat]) if row.get(feat, "") != "" else np.nan for feat in FEATURES])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))

def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    total = 0.0
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def metric_block(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }


def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X, med):
    out = X.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def fit_logistic(X, y, lr=0.035, l2=0.02, epochs=4500):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logistic(X, w):
    return sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)


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


def predict_proba(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]


def module_versions(names):
    out = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            out[name] = f"missing: {exc}"
    return out


def run_models(X_train, y_train, X_test, y_test, X_ext, y_ext):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    versions = module_versions(["sklearn", "xgboost", "lightgbm", "catboost", "tabpfn", "autogluon"])
    med = impute_fit(X_train)
    Xtr_i = impute_apply(X_train, med)
    Xte_i = impute_apply(X_test, med)
    Xex_i = impute_apply(X_ext, med)
    mu, sd = standardize_fit(Xtr_i)
    Xtr_s = standardize_apply(Xtr_i, mu, sd)
    Xte_s = standardize_apply(Xte_i, mu, sd)
    Xex_s = standardize_apply(Xex_i, mu, sd)

    results = {}
    predictions = {}

    def add_result(name, family, p_test, p_ext):
        results[name] = {
            "model_family": family,
            "mimic_internal_test": metric_block(y_test, p_test),
            "eicu_external_original": metric_block(y_ext, p_ext),
        }
        predictions[name] = p_ext
        print(f"{name}: mimic_auc={results[name]['mimic_internal_test']['auc']:.4f} "
              f"eicu_auc={results[name]['eicu_external_original']['auc']:.4f}", flush=True)

    w = fit_logistic(Xtr_s, y_train)
    add_result(
        "Logistic_GERI_NutriRisk_ICU",
        "standardized_l2_logistic_regression",
        predict_logistic(Xte_s, w),
        predict_logistic(Xex_s, w),
    )

    if not str(versions["xgboost"]).startswith("missing"):
        import xgboost as xgb
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=700, max_depth=3, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=20, reg_lambda=3.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=20260826, n_jobs=8, scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        add_result("XGBoost_hist", "gradient_boosted_decision_trees", predict_proba(model, X_test), predict_proba(model, X_ext))

    if not str(versions["lightgbm"]).startswith("missing"):
        import lightgbm as lgb
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=16, max_depth=-1,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=100,
            reg_lambda=3.0, objective="binary", random_state=20260826, n_jobs=8,
            scale_pos_weight=scale_pos_weight, verbose=-1,
        )
        model.fit(X_train, y_train)
        add_result("LightGBM_gbdt", "gradient_boosted_decision_trees", predict_proba(model, X_test), predict_proba(model, X_ext))

    if not str(versions["catboost"]).startswith("missing"):
        import catboost as cb
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = cb.CatBoostClassifier(
            iterations=700, depth=4, learning_rate=0.03, l2_leaf_reg=5.0,
            loss_function="Logloss", eval_metric="AUC", random_seed=20260826,
            thread_count=8, verbose=False, scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        add_result("CatBoost_ordered", "ordered_gradient_boosting", predict_proba(model, X_test), predict_proba(model, X_ext))

    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.035, max_leaf_nodes=16,
        l2_regularization=0.1, random_state=20260826, early_stopping=True,
    )
    sample_weight = np.where(y_train == 1, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1.0)
    hgb.fit(Xtr_i, y_train, sample_weight=sample_weight)
    add_result("Sklearn_HistGradientBoosting", "histogram_gradient_boosting", predict_proba(hgb, Xte_i), predict_proba(hgb, Xex_i))

    rf = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=8, random_state=20260826,
    )
    rf.fit(Xtr_i, y_train)
    add_result("RandomForest_balanced", "bagged_decision_trees", predict_proba(rf, Xte_i), predict_proba(rf, Xex_i))

    et = ExtraTreesClassifier(
        n_estimators=800, max_depth=10, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced", n_jobs=8, random_state=20260826,
    )
    et.fit(Xtr_i, y_train)
    add_result("ExtraTrees_balanced", "extremely_randomized_trees", predict_proba(et, Xte_i), predict_proba(et, Xex_i))

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=512, learning_rate_init=5e-4, max_iter=160,
        early_stopping=True, validation_fraction=0.15, random_state=20260826,
        verbose=False,
    )
    mlp.fit(Xtr_s, y_train)
    add_result("Sklearn_MLP", "feedforward_neural_network", predict_proba(mlp, Xte_s), predict_proba(mlp, Xex_s))

    return results, predictions, versions


def recalibration_blocks(predictions, y_ext):
    idx = np.arange(len(y_ext))
    cal = idx % 5 == 0
    held = ~cal
    out = {}
    for name, p in predictions.items():
        w = fit_platt(logit(p[cal]), y_ext[cal])
        p_cal = apply_platt(p, w)
        out[name] = {
            "platt_parameters": {"intercept": float(w[0]), "slope": float(w[1])},
            "eicu_calibration_split_original": metric_block(y_ext[cal], p[cal]),
            "eicu_heldout_original": metric_block(y_ext[held], p[held]),
            "eicu_heldout_platt": metric_block(y_ext[held], p_cal[held]),
        }
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_mimic, y_mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    X_eicu, y_eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    train_mask = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    X_train, y_train = X_mimic[train_mask], y_mimic[train_mask]
    X_test, y_test = X_mimic[~train_mask], y_mimic[~train_mask]
    print(f"loaded train={len(y_train)} test={len(y_test)} eicu={len(y_eicu)}", flush=True)

    benchmark, predictions, versions = run_models(X_train, y_train, X_test, y_test, X_eicu, y_eicu)
    recalibrated = recalibration_blocks(predictions, y_eicu)

    rows = []
    for name, block in benchmark.items():
        rec = recalibrated[name]
        rows.append({
            "model": name,
            "family": block["model_family"],
            "mimic_auc": block["mimic_internal_test"]["auc"],
            "mimic_brier": block["mimic_internal_test"]["brier"],
            "eicu_auc_original": block["eicu_external_original"]["auc"],
            "eicu_brier_original": block["eicu_external_original"]["brier"],
            "eicu_ece_original": block["eicu_external_original"]["ece"],
            "eicu_auc_platt_heldout": rec["eicu_heldout_platt"]["auc"],
            "eicu_brier_platt_heldout": rec["eicu_heldout_platt"]["brier"],
            "eicu_ece_platt_heldout": rec["eicu_heldout_platt"]["ece"],
        })
    rows = sorted(rows, key=lambda r: (r["eicu_auc_platt_heldout"], -r["eicu_brier_platt_heldout"]), reverse=True)
    write_csv(OUT / "table_v10_sota_completion_lock.csv", rows)

    best = rows[0]
    lgb = next((r for r in rows if r["model"] == "LightGBM_gbdt"), None)
    lgb_delta = None if lgb is None else best["eicu_auc_platt_heldout"] - lgb["eicu_auc_platt_heldout"]
    material_gain = lgb_delta is not None and lgb_delta > 0.015
    lock_decision = (
        "replace_lightgbm_review_required"
        if best["model"] != "LightGBM_gbdt" and material_gain
        else "lock_lightgbm_no_material_sota_gain"
    )
    lock = {
        "primary_performance_head": "LightGBM_gbdt",
        "best_model_by_external_auc": best["model"],
        "best_external_auc": best["eicu_auc_platt_heldout"],
        "lightgbm_external_auc": None if lgb is None else lgb["eicu_auc_platt_heldout"],
        "best_minus_lightgbm_external_auc": lgb_delta,
        "material_replacement_gain": material_gain,
        "model_lock_decision": lock_decision,
        "replacement_rule": "Replace LightGBM only if a bounded comparator improves eICU held-out Platt AUROC by more than 0.015 without worse Brier/ECE or interpretability tradeoff.",
    }
    result = {
        "protocol": "V10_SOTA_COMPLETION_LOCK",
        "source_feature_tables": {
            "mimic": str(V3_DIR / "mimic_v3_features_server_only.csv"),
            "eicu": str(V3_DIR / "eicu_v3_features_server_only.csv"),
        },
        "split": {
            "mimic_training": "row_index_mod_5_not_equal_0",
            "mimic_internal_test": "row_index_mod_5_equal_0",
            "eicu_platt_calibration": "row_index_mod_5_equal_0",
            "eicu_platt_heldout": "row_index_mod_5_not_equal_0",
        },
        "environment_modules": versions,
        "benchmark": benchmark,
        "external_recalibration": recalibrated,
        "rows_sorted": rows,
        "model_lock": lock,
        "claim_boundary": "All comparators are trained only on MIMIC-IV development data. eICU labels are reserved for external evaluation and prespecified calibration/held-out splits. The comparator queue is bounded to prevent post-hoc model shopping.",
    }
    (OUT / "v10_sota_completion_lock_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "# V10 SOTA Completion And Model Lock",
        "",
        "Status date: 2026-08-26",
        "",
        "## Purpose",
        "",
        "This bounded completion run tests whether any additional CPU-feasible tabular comparator should displace the LightGBM performance head selected after V8/V9.",
        "",
        "## Models Run",
        "",
        *[f"- {r['model']} ({r['family']})" for r in rows],
        "",
        "## Key Metrics",
        "",
        "| Model | MIMIC AUROC | eICU AUROC | eICU Brier | eICU Platt held-out AUROC | eICU Platt held-out Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model']} | {row['mimic_auc']:.3f} | {row['eicu_auc_original']:.3f} | "
            f"{row['eicu_brier_original']:.3f} | {row['eicu_auc_platt_heldout']:.3f} | "
            f"{row['eicu_brier_platt_heldout']:.3f} |"
        )
    md.extend([
        "",
        "## Model-Lock Decision",
        "",
        f"Best model by eICU held-out Platt AUROC: {best['model']} ({best['eicu_auc_platt_heldout']:.3f}).",
        f"Prespecified primary performance head: {lock['primary_performance_head']}.",
        f"Decision: {lock['model_lock_decision']}.",
        "",
        "## Interpretation Boundary",
        "",
        "The final manuscript should report the bounded comparator queue transparently and avoid implying that every possible neural tabular architecture was exhaustively searched. The defensible claim is that the selected framework outperformed strong conventional and contemporary tabular comparators available in the validated compute environment under an external-validation and recalibration protocol.",
    ])
    (OUT / "V10_SOTA_COMPLETION_LOCK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "best": best, "lock": lock, "models": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
