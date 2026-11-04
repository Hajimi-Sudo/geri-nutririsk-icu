#!/usr/bin/env python3
"""V8 SOTA tabular benchmark for GERI-NutriRisk-ICU.

Runs strong contemporary tabular classifiers on the same server-only V3 feature
tables. Outputs are aggregate only; patient-level predictions remain server-side.
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
OUT = ROOT / "v8_sota_tabular_benchmark"

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


def load_table(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = []
    X = []
    y = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            y.append(int(float(row["y"])))
            vals = []
            for feat in FEATURES:
                val = row.get(feat, "")
                vals.append(float(val) if val != "" else np.nan)
            X.append(vals)
    return rows, np.array(X, dtype=float), np.array(y, dtype=int)


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
    med = np.where(np.isnan(med), 0.0, med)
    return med


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
    score = logit(p)
    return sigmoid(w[0] + w[1] * score)


def predict_proba(model, X):
    proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1]


def optional_modules():
    out = {}
    for name in ["xgboost", "lightgbm", "catboost"]:
        try:
            out[name] = importlib.import_module(name)
        except Exception as exc:
            out[name] = f"missing: {exc}"
    return out


def run_models(X_train, y_train, X_test, y_test, X_ext, y_ext):
    modules = optional_modules()
    results = {}
    predictions = {}
    models_run = []

    med = impute_fit(X_train)
    Xtr_i = impute_apply(X_train, med)
    Xte_i = impute_apply(X_test, med)
    Xex_i = impute_apply(X_ext, med)
    mu, sd = standardize_fit(Xtr_i)
    Xtr_s = standardize_apply(Xtr_i, mu, sd)
    Xte_s = standardize_apply(Xte_i, mu, sd)
    Xex_s = standardize_apply(Xex_i, mu, sd)

    w = fit_logistic(Xtr_s, y_train)
    p_test = predict_logistic(Xte_s, w)
    p_ext = predict_logistic(Xex_s, w)
    results["GERI-NutriRisk-ICU_logistic"] = {
        "model_family": "standardized_l2_logistic_regression",
        "mimic_internal_test": metric_block(y_test, p_test),
        "eicu_external_original": metric_block(y_ext, p_ext),
    }
    predictions["GERI-NutriRisk-ICU_logistic"] = p_ext
    models_run.append("GERI-NutriRisk-ICU_logistic")

    if not isinstance(modules["xgboost"], str):
        xgb = modules["xgboost"]
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=20,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=20260826,
            n_jobs=8,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        p_test = predict_proba(model, X_test)
        p_ext = predict_proba(model, X_ext)
        results["XGBoost_hist"] = {
            "model_family": "gradient_boosted_decision_trees",
            "mimic_internal_test": metric_block(y_test, p_test),
            "eicu_external_original": metric_block(y_ext, p_ext),
        }
        predictions["XGBoost_hist"] = p_ext
        models_run.append("XGBoost_hist")

    if not isinstance(modules["lightgbm"], str):
        lgb = modules["lightgbm"]
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=100,
            reg_lambda=3.0,
            objective="binary",
            random_state=20260826,
            n_jobs=8,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        p_test = predict_proba(model, X_test)
        p_ext = predict_proba(model, X_ext)
        results["LightGBM_gbdt"] = {
            "model_family": "gradient_boosted_decision_trees",
            "mimic_internal_test": metric_block(y_test, p_test),
            "eicu_external_original": metric_block(y_ext, p_ext),
        }
        predictions["LightGBM_gbdt"] = p_ext
        models_run.append("LightGBM_gbdt")

    if not isinstance(modules["catboost"], str):
        cb = modules["catboost"]
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        model = cb.CatBoostClassifier(
            iterations=700,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=20260826,
            thread_count=8,
            verbose=False,
            auto_class_weights=None,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        p_test = predict_proba(model, X_test)
        p_ext = predict_proba(model, X_ext)
        results["CatBoost_ordered"] = {
            "model_family": "ordered_gradient_boosting",
            "mimic_internal_test": metric_block(y_test, p_test),
            "eicu_external_original": metric_block(y_ext, p_ext),
        }
        predictions["CatBoost_ordered"] = p_ext
        models_run.append("CatBoost_ordered")

    return results, predictions, modules, models_run


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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, X_mimic, y_mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    _, X_eicu, y_eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    train_mask = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    X_train, y_train = X_mimic[train_mask], y_mimic[train_mask]
    X_test, y_test = X_mimic[~train_mask], y_mimic[~train_mask]
    print(f"loaded train={len(y_train)} test={len(y_test)} eicu={len(y_eicu)}", flush=True)

    benchmark, predictions, modules, models_run = run_models(X_train, y_train, X_test, y_test, X_eicu, y_eicu)
    recalibrated = recalibration_blocks(predictions, y_eicu)
    result = {
        "protocol": "V8_SOTA_TABULAR_BENCHMARK",
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
        "environment_modules": {k: (str(v) if isinstance(v, str) else getattr(v, "__version__", "installed")) for k, v in modules.items()},
        "benchmark": benchmark,
        "external_recalibration": recalibrated,
        "models_run": models_run,
        "claim_boundary": "SOTA comparators are trained only on MIMIC-IV development data. eICU labels are reserved for external evaluation and prespecified Platt-domain recalibration splits. Patient-level predictions are not exported.",
    }
    (OUT / "v8_sota_tabular_benchmark_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

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
    write_csv(OUT / "table_sota_tabular_benchmark.csv", rows)

    md = [
        "# V8 SOTA Tabular Benchmark",
        "",
        "Status date: 2026-08-26",
        "",
        "## Purpose",
        "",
        "V8 benchmarks GERI-NutriRisk-ICU against strong contemporary tabular machine-learning comparators on the same harmonized MIMIC-IV development and eICU-CRD external-validation feature tables.",
        "",
        "## Models Run",
        "",
        *[f"- {name}" for name in models_run],
        "",
        "## Key Metrics",
        "",
        "| Model | MIMIC AUROC | eICU AUROC | eICU Brier | eICU Platt held-out Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model']} | {row['mimic_auc']:.3f} | {row['eicu_auc_original']:.3f} | {row['eicu_brier_original']:.3f} | {row['eicu_brier_platt_heldout']:.3f} |"
        )
    md.extend([
        "",
        "## Interpretation",
        "",
        "The strongest model should be selected by external validation, calibration, and interpretability rather than internal AUROC alone. If gradient-boosting models improve external discrimination without unacceptable calibration loss, they should be reported as SOTA comparators and either replace or complement the logistic framework. If they overfit or calibrate poorly, the logistic framework remains the primary clinical model.",
    ])
    (OUT / "V8_SOTA_TABULAR_BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "models_run": models_run, "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
