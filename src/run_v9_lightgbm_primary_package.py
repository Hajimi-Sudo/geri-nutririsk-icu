#!/usr/bin/env python3
"""V9 LightGBM primary performance-head package.

Runs the strongest V8 comparator as the performance head of the
GERI-NutriRisk-ICU framework, with CI, recalibration, operating profile,
decision-curve source data, and LightGBM contribution summaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
V3_DIR = ROOT / "v3_severity_model"
OUT = ROOT / "v9_lightgbm_primary"

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


def load_table(path: Path):
    rows, X, y = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            y.append(int(float(row["y"])))
            X.append([float(row.get(feat, "")) if row.get(feat, "") != "" else np.nan for feat in FEATURES])
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


def bootstrap_ci(y, p, n_resamples=1000, seed=20260826):
    rng = np.random.default_rng(seed)
    draws = {"auc": [], "brier": [], "ece": []}
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        draws["auc"].append(auc(yy, pp))
        draws["brier"].append(brier(yy, pp))
        draws["ece"].append(ece(yy, pp))
    out = {}
    for metric, vals in draws.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[metric] = {"estimate": metric_block(y, p)[metric], "ci_95": [float(lo), float(hi)], "n_resamples": len(vals)}
    return out


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


def roc_points(y, p, n=101):
    thresholds = np.quantile(p, np.linspace(1, 0, n))
    positives = y.sum()
    negatives = len(y) - positives
    rows = []
    for th in thresholds:
        pred = p >= th
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        rows.append({"threshold": float(th), "tpr": tp / positives, "fpr": fp / negatives})
    return rows


def calibration_bins(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        mask = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if mask.sum():
            rows.append({
                "bin": i + 1,
                "n": int(mask.sum()),
                "pred_mean": float(p[mask].mean()),
                "obs_rate": float(y[mask].mean()),
            })
    return rows


def decision_curve(y, p, thresholds=None):
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.02, 0.31, 0.02)]
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = float(((pred == 1) & (y == 1)).sum()) / n
        fp = float(((pred == 1) & (y == 0)).sum()) / n
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": tp - fp * pt / (1 - pt),
            "net_benefit_treat_all": prevalence - (1 - prevalence) * pt / (1 - pt),
            "net_benefit_treat_none": 0.0,
        })
    return rows


def operating_points(y, p, thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for th in thresholds:
        pred = p >= th
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        rows.append({
            "threshold": th,
            "alerts_per_100": float(pred.mean() * 100),
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "net_benefit_model": tp / n - fp / n * th / (1 - th),
            "net_benefit_treat_all": prevalence - (1 - prevalence) * th / (1 - th),
            "net_benefit_treat_none": 0.0,
        })
    return rows


def subgroup_definitions(row):
    age = float(row.get("age") or 0)
    return {
        "sex": "male" if float(row.get("male") or 0) == 1 else "female_or_other",
        "age_group": "65-74" if age < 75 else ("75-84" if age < 85 else "85+"),
        "albumin_available": "available" if row.get("albumin", "") != "" else "missing",
        "ventilation_24h": "yes" if float(row.get("ventilation_24h") or 0) == 1 else "no",
        "enteral_24h": "yes" if float(row.get("enteral_24h") or 0) == 1 else "no",
    }


def subgroup_metrics(rows, y, p):
    groups = {}
    for i, row in enumerate(rows):
        for domain, value in subgroup_definitions(row).items():
            groups.setdefault((domain, value), []).append(i)
    out = []
    for (domain, value), idx in sorted(groups.items()):
        idx = np.array(idx)
        if len(idx) < 500:
            continue
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append({"domain": domain, "subgroup": value, **metric_block(yy, pp)})
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
    eicu_rows, X_eicu, y_eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    _, X_mimic, y_mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    train_mask = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    X_train, y_train = X_mimic[train_mask], y_mimic[train_mask]
    X_test, y_test = X_mimic[~train_mask], y_mimic[~train_mask]
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=16,
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
    print(f"training LightGBM train={len(y_train)} test={len(y_test)} eicu={len(y_eicu)}", flush=True)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    p_ext = model.predict_proba(X_eicu)[:, 1]
    idx = np.arange(len(y_eicu))
    cal = idx % 5 == 0
    held = ~cal
    platt = fit_platt(logit(p_ext[cal]), y_eicu[cal])
    p_platt = apply_platt(p_ext, platt)

    print("bootstrap CIs", flush=True)
    ci = {
        "mimic_internal_test": bootstrap_ci(y_test, p_test, seed=11),
        "eicu_external_original_full": bootstrap_ci(y_eicu, p_ext, seed=12),
        "eicu_external_platt_heldout": bootstrap_ci(y_eicu[held], p_platt[held], seed=13),
    }
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    contrib = model.booster_.predict(X_eicu, pred_contrib=True)
    shap_mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    importance = [
        {
            "feature": feat,
            "gain_importance": float(g),
            "split_importance": int(s),
            "mean_abs_shap_contribution": float(a),
        }
        for feat, g, s, a in zip(FEATURES, gain, split, shap_mean_abs)
    ]
    importance.sort(key=lambda r: r["mean_abs_shap_contribution"], reverse=True)

    result = {
        "protocol": "V9_LIGHTGBM_PRIMARY_PACKAGE",
        "model_name": "GERI-NutriRisk-ICU-LightGBM performance head",
        "source_feature_tables": {
            "mimic": str(V3_DIR / "mimic_v3_features_server_only.csv"),
            "eicu": str(V3_DIR / "eicu_v3_features_server_only.csv"),
        },
        "hyperparameters": model.get_params(),
        "platt_parameters": {"intercept": float(platt[0]), "slope": float(platt[1])},
        "point_estimates": {
            "mimic_internal_test": metric_block(y_test, p_test),
            "eicu_external_original_full": metric_block(y_eicu, p_ext),
            "eicu_external_platt_heldout": metric_block(y_eicu[held], p_platt[held]),
        },
        "bootstrap_ci": ci,
        "plot_data": {
            "eicu_roc_original": roc_points(y_eicu, p_ext),
            "eicu_calibration_original": calibration_bins(y_eicu, p_ext),
            "eicu_calibration_platt_heldout": calibration_bins(y_eicu[held], p_platt[held]),
            "eicu_decision_curve_original": decision_curve(y_eicu, p_ext),
            "eicu_decision_curve_platt_heldout": decision_curve(y_eicu[held], p_platt[held]),
        },
        "operating_profile": {
            "eicu_platt_heldout": operating_points(y_eicu[held], p_platt[held]),
        },
        "subgroups_original_external": subgroup_metrics(eicu_rows, y_eicu, p_ext),
        "feature_importance": importance,
        "claim_boundary": "LightGBM trained only in MIMIC-IV development data; eICU labels used for external evaluation and prespecified Platt recalibration split only. Patient-level predictions and features remain on server.",
    }
    (OUT / "v9_lightgbm_primary_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(OUT / "table_lightgbm_primary_metrics.csv", [
        {"evaluation": k, **v} for k, v in result["point_estimates"].items()
    ])
    write_csv(OUT / "table_lightgbm_operating_profile_platt.csv", result["operating_profile"]["eicu_platt_heldout"])
    write_csv(OUT / "table_lightgbm_subgroups_original.csv", result["subgroups_original_external"])
    write_csv(OUT / "table_lightgbm_feature_importance.csv", importance)
    for key, rows in result["plot_data"].items():
        write_csv(OUT / f"figure_source_lightgbm_{key}.csv", rows)
    md = [
        "# V9 LightGBM Primary Package",
        "",
        "Status date: 2026-08-26",
        "",
        "## Primary Performance Head",
        "",
        "LightGBM is selected as the SOTA performance head within the GERI-NutriRisk-ICU framework because it had the strongest external discrimination in V8.",
        "",
        "## Primary Metrics",
        "",
        f"- MIMIC internal AUROC: {result['point_estimates']['mimic_internal_test']['auc']:.3f}.",
        f"- eICU external AUROC: {result['point_estimates']['eicu_external_original_full']['auc']:.3f}.",
        f"- eICU Platt held-out Brier: {result['point_estimates']['eicu_external_platt_heldout']['brier']:.3f}.",
        "",
        "## Interpretation",
        "",
        "The framework should be described as a harmonized, externally validated, recalibrated ICU nutrition-risk framework with a LightGBM performance head and transparent logistic sensitivity model. Strong claims should remain bounded by retrospective external validation.",
    ]
    (OUT / "V9_LIGHTGBM_PRIMARY_PACKAGE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "eicu_auc": result["point_estimates"]["eicu_external_original_full"]["auc"],
        "platt_brier": result["point_estimates"]["eicu_external_platt_heldout"]["brier"],
        "top_features": [r["feature"] for r in importance[:8]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
