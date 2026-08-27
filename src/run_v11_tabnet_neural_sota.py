#!/usr/bin/env python3
"""V11 neural tabular SOTA comparator using TabNet.

TabNet is run as a bounded neural comparator against the locked LightGBM
performance head. It uses MIMIC-IV only for fitting and early stopping, and eICU
only for external evaluation plus the same prespecified Platt split used in V8
through V10.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
V3_DIR = ROOT / "v3_severity_model"
V10 = ROOT / "v10_sota_completion_lock"
OUT = ROOT / "v11_tabnet_neural_sota"

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
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


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
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
    return np.where(np.isnan(med), 0.0, med).astype(np.float32)


def impute_apply(X, med):
    out = X.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out.astype(np.float32)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


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


def run_tabnet(X_train, y_train, X_test, y_test, X_ext, y_ext):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    np.random.seed(20260826)
    torch.manual_seed(20260826)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260826)

    inner_val = np.array([(i % 5) == 0 for i in range(len(y_train))])
    inner_train = ~inner_val

    med = impute_fit(X_train[inner_train])
    Xtr_i = impute_apply(X_train, med)
    Xte_i = impute_apply(X_test, med)
    Xex_i = impute_apply(X_ext, med)
    mu, sd = standardize_fit(Xtr_i[inner_train])
    Xtr = standardize_apply(Xtr_i, mu, sd)
    Xte = standardize_apply(Xte_i, mu, sd)
    Xex = standardize_apply(Xex_i, mu, sd)

    model = TabNetClassifier(
        n_d=24,
        n_a=24,
        n_steps=4,
        gamma=1.4,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 1.5e-2, "weight_decay": 1e-5},
        scheduler_params={"step_size": 20, "gamma": 0.85},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=20260826,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    model.fit(
        Xtr[inner_train],
        y_train[inner_train],
        eval_set=[(Xtr[inner_val], y_train[inner_val])],
        eval_name=["mimic_inner_val"],
        eval_metric=["auc"],
        max_epochs=90,
        patience=14,
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
        weights=1,
    )
    p_test = model.predict_proba(Xte)[:, 1]
    p_ext = model.predict_proba(Xex)[:, 1]
    return p_test, p_ext, {
        "best_epoch": int(getattr(model, "best_epoch", -1)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def recalibrate(p_ext, y_ext):
    idx = np.arange(len(y_ext))
    cal = idx % 5 == 0
    held = ~cal
    w = fit_platt(logit(p_ext[cal]), y_ext[cal])
    p_cal = apply_platt(p_ext, w)
    return {
        "platt_parameters": {"intercept": float(w[0]), "slope": float(w[1])},
        "eicu_calibration_split_original": metric_block(y_ext[cal], p_ext[cal]),
        "eicu_heldout_original": metric_block(y_ext[held], p_ext[held]),
        "eicu_heldout_platt": metric_block(y_ext[held], p_cal[held]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_mimic, y_mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    X_eicu, y_eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    train_mask = np.array([(i % 5) != 0 for i in range(len(y_mimic))])
    X_train, y_train = X_mimic[train_mask], y_mimic[train_mask]
    X_test, y_test = X_mimic[~train_mask], y_mimic[~train_mask]
    print(f"loaded train={len(y_train)} test={len(y_test)} eicu={len(y_eicu)}", flush=True)

    p_test, p_ext, train_info = run_tabnet(X_train, y_train, X_test, y_test, X_eicu, y_eicu)
    rec = recalibrate(p_ext, y_eicu)
    tabnet_row = {
        "model": "TabNet_neural",
        "family": "attentive_deep_tabular_network",
        "mimic_auc": metric_block(y_test, p_test)["auc"],
        "mimic_brier": metric_block(y_test, p_test)["brier"],
        "eicu_auc_original": metric_block(y_eicu, p_ext)["auc"],
        "eicu_brier_original": metric_block(y_eicu, p_ext)["brier"],
        "eicu_ece_original": metric_block(y_eicu, p_ext)["ece"],
        "eicu_auc_platt_heldout": rec["eicu_heldout_platt"]["auc"],
        "eicu_brier_platt_heldout": rec["eicu_heldout_platt"]["brier"],
        "eicu_ece_platt_heldout": rec["eicu_heldout_platt"]["ece"],
    }

    v10_metrics = json.loads((V10 / "v10_sota_completion_lock_metrics.json").read_text(encoding="utf-8"))
    lgb = next(r for r in v10_metrics["rows_sorted"] if r["model"] == "LightGBM_gbdt")
    delta = tabnet_row["eicu_auc_platt_heldout"] - lgb["eicu_auc_platt_heldout"]
    lock_decision = "keep_lightgbm" if delta <= 0.015 else "review_tabnet_replacement"

    rows = [lgb, tabnet_row]
    with (OUT / "table_v11_tabnet_neural_sota.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(tabnet_row)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    result = {
        "protocol": "V11_TABNET_NEURAL_SOTA",
        "train_info": train_info,
        "lightgbm_reference": lgb,
        "tabnet": tabnet_row,
        "tabnet_minus_lightgbm_external_auc": delta,
        "model_lock_decision": lock_decision,
        "claim_boundary": "TabNet is included as a bounded neural tabular comparator. It is not used for post-hoc model shopping; LightGBM remains locked unless the neural comparator exceeds the prespecified material improvement threshold.",
    }
    (OUT / "v11_tabnet_neural_sota_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "# V11 TabNet Neural SOTA Comparator",
        "",
        "Status date: 2026-08-26",
        "",
        "## Purpose",
        "",
        "This run adds a neural tabular comparator to the sealed SOTA queue after LightGBM was selected as the performance head.",
        "",
        "## Key Metrics",
        "",
        "| Model | MIMIC AUROC | eICU AUROC | eICU Platt held-out AUROC | eICU Platt held-out Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model']} | {row['mimic_auc']:.3f} | {row['eicu_auc_original']:.3f} | "
            f"{row['eicu_auc_platt_heldout']:.3f} | {row['eicu_brier_platt_heldout']:.3f} |"
        )
    md.extend([
        "",
        "## Model-Lock Decision",
        "",
        f"TabNet minus LightGBM eICU held-out Platt AUROC: {delta:.4f}.",
        f"Decision: {lock_decision}.",
    ])
    (OUT / "V11_TABNET_NEURAL_SOTA.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
