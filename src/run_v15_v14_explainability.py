#!/usr/bin/env python3
"""V15 explainability for the V14 locked domain-update model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SERVER_ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
ROOT = SERVER_ROOT if (SERVER_ROOT / "v13_strong_clinical_feature_upgrade").exists() else PROJECT / "05-results"
V13 = ROOT / "v13_strong_clinical_feature_upgrade"
OUT = ROOT / "v15_v14_explainability"


def load_table(path: Path):
    rows, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        features = [c for c in reader.fieldnames or [] if c not in {"id", "y"}]
        for row in reader:
            rows.append(row["id"])
            y.append(int(float(row["y"])))
            X.append([float(row[c]) if row.get(c, "") != "" else np.nan for c in features])
    return rows, np.asarray(X, dtype=float), np.asarray(y, dtype=int), features


def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X, med):
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


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


def fit_lgbm(X, y):
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
        random_state=20261840,
        n_jobs=8,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def feature_group(name: str) -> str:
    if name.endswith("_missing"):
        return "Missingness indicators"
    if name in {"age", "male"}:
        return "Demographics"
    if name in {"bmi", "enteral_24h", "parenteral_24h", "albumin", "glucose"}:
        return "Explicit nutrition"
    if name in {"creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc"}:
        return "Nutrition/metabolic labs"
    if name in {"heart_rate", "resp_rate", "temperature_c", "spo2", "map"}:
        return "Vital signs"
    if name in {"urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h"}:
        return "Organ support and neurologic severity"
    if name in {"ph", "pco2", "pao2", "fio2", "pf_ratio", "base_excess", "lactate"}:
        return "Blood gas and perfusion"
    if name in {"platelet", "inr", "pt", "ptt", "hemoglobin"}:
        return "Coagulation and hematology"
    if name in {"potassium", "chloride", "bicarbonate", "calcium", "magnesium", "anion_gap"}:
        return "Additional chemistry"
    if name.startswith("icu_type") or name.startswith("admit_"):
        return "ICU and admission context"
    return "Other"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    _, X_e_raw, y_e, features = load_table(V13 / "eicu_v13_features_server_only.csv")
    _, X_m_raw, y_m, _ = load_table(V13 / "mimic_v13_features_server_only.csv")
    med = impute_fit(X_m_raw)
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_nested_held = ~(e_update_fit | e_update_cal)

    model = fit_lgbm(
        np.vstack([X_m[m_train], X_e[e_update_fit]]),
        np.concatenate([y_m[m_train], y_e[e_update_fit]]),
    )
    p_all = model.predict_proba(X_e)[:, 1]
    w = fit_platt(logit(p_all[e_update_cal]), y_e[e_update_cal])
    p_all = apply_platt(p_all, w)
    Xh = X_e[e_nested_held]
    yh = y_e[e_nested_held]
    base_auc = auc(yh, p_all[e_nested_held])
    base_brier = brier(yh, p_all[e_nested_held])

    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    imp_rows = []
    for name, g, s in zip(features, gain, split):
        imp_rows.append({
            "feature": name,
            "group": feature_group(name),
            "gain": float(g),
            "split": int(s),
            "gain_fraction": float(g / gain.sum()) if gain.sum() else 0.0,
        })
    imp_rows.sort(key=lambda r: r["gain"], reverse=True)

    rng = np.random.default_rng(20260861)
    groups = {}
    for idx, name in enumerate(features):
        groups.setdefault(feature_group(name), []).append(idx)
    perm_rows = []
    for group, cols in groups.items():
        Xp = Xh.copy()
        for c in cols:
            Xp[:, c] = rng.permutation(Xp[:, c])
        pp = apply_platt(model.predict_proba(Xp)[:, 1], w)
        perm_rows.append({
            "group": group,
            "n_features": len(cols),
            "auc_after_permutation": auc(yh, pp),
            "auc_drop": base_auc - auc(yh, pp),
            "brier_after_permutation": brier(yh, pp),
            "brier_increase": brier(yh, pp) - base_brier,
        })
    perm_rows.sort(key=lambda r: r["auc_drop"], reverse=True)

    with (OUT / "table_v15_feature_importance_gain.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(imp_rows[0]))
        writer.writeheader()
        writer.writerows(imp_rows)
    with (OUT / "table_v15_group_permutation_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(perm_rows[0]))
        writer.writeheader()
        writer.writerows(perm_rows)

    top = list(reversed(imp_rows[:20]))
    plt.figure(figsize=(8, 6))
    plt.barh([r["feature"] for r in top], [r["gain_fraction"] for r in top], color="#2f6f8f")
    plt.xlabel("Fraction of total LightGBM gain")
    plt.tight_layout()
    plt.savefig(OUT / "fig_v15_top20_gain_importance.png", dpi=300)
    plt.close()

    groups_plot = list(reversed(perm_rows[:10]))
    plt.figure(figsize=(8, 4.8))
    plt.barh([r["group"] for r in groups_plot], [r["auc_drop"] for r in groups_plot], color="#8f5a2f")
    plt.xlabel("AUROC decrease after group permutation")
    plt.tight_layout()
    plt.savefig(OUT / "fig_v15_group_permutation_importance.png", dpi=300)
    plt.close()

    result = {
        "protocol": "V15_V14_EXPLAINABILITY",
        "status_date": "2026-08-26",
        "model": "V14 full-feature combined nested domain-update LightGBM with Platt calibration",
        "heldout_n": int(len(yh)),
        "base_auc": base_auc,
        "base_brier": base_brier,
        "platt_parameters": {"intercept": float(w[0]), "slope": float(w[1])},
        "top_gain_features": imp_rows[:20],
        "group_permutation_importance": perm_rows,
        "claim_boundary": (
            "Feature importance is explanatory for the fitted prediction model and does not "
            "support causal claims about nutrition treatment, malnutrition, or physiology."
        ),
    }
    (OUT / "v15_v14_explainability_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        "# V15 V14 Explainability",
        "",
        "Status date: 2026-08-26",
        "",
        f"Base held-out eICU AUROC: {base_auc:.3f}; Brier: {base_brier:.3f}.",
        "",
        "## Leading Feature Groups By Permutation AUROC Drop",
    ]
    for row in perm_rows[:8]:
        md.append(f"- {row['group']}: AUROC drop {row['auc_drop']:.4f}; Brier increase {row['brier_increase']:.4f}.")
    md += ["", "## Leading Gain Features"]
    for row in imp_rows[:12]:
        md.append(f"- {row['feature']}: gain fraction {row['gain_fraction']:.3f}.")
    md += ["", "## Boundary", "", result["claim_boundary"]]
    (OUT / "V15_V14_EXPLAINABILITY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "base_auc": base_auc,
        "base_brier": base_brier,
        "top_groups": perm_rows[:5],
        "top_features": imp_rows[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
