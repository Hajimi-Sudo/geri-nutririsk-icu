#!/usr/bin/env python3
"""V7 benchmark, stability, subgroup-CI, and operating-profile package.

This script uses server-only V3 feature tables. It writes aggregate JSON/CSV
outputs only; patient-level rows and predictions remain on the secured server.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
V3_DIR = ROOT / "v3_severity_model"
OUT = ROOT / "v7_model_benchmark"

FEATURES = [
    "age",
    "male",
    "bmi",
    "enteral_24h",
    "parenteral_24h",
    "albumin",
    "glucose",
    "creatinine",
    "bun",
    "sodium",
    "bilirubin",
    "hematocrit",
    "wbc",
    "albumin_missing",
    "glucose_missing",
    "creatinine_missing",
    "bun_missing",
    "sodium_missing",
    "bilirubin_missing",
    "hematocrit_missing",
    "wbc_missing",
    "heart_rate",
    "resp_rate",
    "temperature_c",
    "spo2",
    "map",
    "heart_rate_missing",
    "resp_rate_missing",
    "temperature_c_missing",
    "spo2_missing",
    "map_missing",
    "urine_24h_ml",
    "ventilation_24h",
    "dialysis_24h",
    "gcs_24h",
    "urine_24h_ml_missing",
    "ventilation_24h_missing",
    "dialysis_24h_missing",
    "gcs_24h_missing",
]

FEATURE_SETS = {
    "intercept_only": [],
    "demographics_bmi": ["age", "male", "bmi"],
    "nutrition_support_only": ["age", "male", "bmi", "enteral_24h", "parenteral_24h"],
    "nutrition_metabolic_core": [
        "age", "male", "bmi", "enteral_24h", "parenteral_24h",
        "albumin", "glucose", "creatinine", "bun", "sodium", "bilirubin", "hematocrit", "wbc",
        "albumin_missing", "glucose_missing", "creatinine_missing", "bun_missing", "sodium_missing",
        "bilirubin_missing", "hematocrit_missing", "wbc_missing",
    ],
    "non_nutrition_clinical_baseline": [
        "age", "male", "heart_rate", "resp_rate", "spo2", "map",
        "urine_24h_ml", "ventilation_24h", "dialysis_24h", "gcs_24h",
        "heart_rate_missing", "resp_rate_missing", "spo2_missing", "map_missing",
        "urine_24h_ml_missing", "gcs_24h_missing",
    ],
    "full_geri_nutririsk_icu_logistic": FEATURES,
}


def load_table(path: Path) -> list[dict]:
    out = []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {"id": row["id"], "y": int(float(row["y"]))}
            for k, v in row.items():
                if k in {"id", "y"}:
                    continue
                rec[k] = float(v) if v != "" else None
            out.append(rec)
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def matrix(data, features, impute=None, scaler=None, add_intercept=True):
    if not features:
        X = np.ones((len(data), 1), dtype=float)
        y = np.array([r["y"] for r in data], dtype=float)
        return X, y, {}, {}
    if impute is None:
        impute = {}
        for feat in features:
            vals = [r.get(feat) for r in data if r.get(feat) is not None]
            impute[feat] = float(np.median(vals)) if vals else 0.0
    raw = np.array([[r.get(feat) if r.get(feat) is not None else impute[feat] for feat in features] for r in data], dtype=float)
    if scaler is None:
        scaler = {}
        for j, feat in enumerate(features):
            mu = float(raw[:, j].mean())
            sd = float(raw[:, j].std()) or 1.0
            scaler[feat] = (mu, sd)
    X = raw.copy()
    for j, feat in enumerate(features):
        mu, sd = scaler[feat]
        X[:, j] = (X[:, j] - mu) / sd
    if add_intercept:
        X = np.column_stack([np.ones(len(data)), X])
    y = np.array([r["y"] for r in data], dtype=float)
    return X, y, impute, scaler


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
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


def fit_logistic(X, y, lr=0.035, l2=0.02, epochs=4500):
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def fit_platt(score, y, lr=0.035, epochs=4500):
    X = np.column_stack([np.ones(len(score)), score])
    w = np.zeros(2, dtype=float)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        grad[1] += 0.001 * w[1]
        w -= lr * grad
    return w


def fit_mlp(X, y, seed=20260826, hidden=24, epochs=900, lr=0.0025, l2=0.0005):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 0.12, size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.12, size=hidden)
    b2 = 0.0
    beta1, beta2 = 0.9, 0.999
    params = [W1, b1, W2, np.array([b2])]
    m = [np.zeros_like(x) for x in params]
    v = [np.zeros_like(x) for x in params]
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    sw = np.where(y == 1, pos_weight, 1.0)
    sw = sw / sw.mean()
    for t in range(1, epochs + 1):
        H = np.tanh(X @ W1 + b1)
        z = H @ W2 + params[3][0]
        p = sigmoid(z)
        dz = (p - y) * sw / n
        gW2 = H.T @ dz + l2 * W2
        gb2 = np.array([dz.sum()])
        dH = np.outer(dz, W2) * (1 - H * H)
        gW1 = X.T @ dH + l2 * W1
        gb1 = dH.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] * grads[i])
            mh = m[i] / (1 - beta1 ** t)
            vh = v[i] / (1 - beta2 ** t)
            params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
        W1, b1, W2 = params[0], params[1], params[2]
    return params


def predict_mlp(X, params):
    W1, b1, W2, b2 = params
    return sigmoid(np.tanh(X @ W1 + b1) @ W2 + b2[0])


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def operating_points(y, p, thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    rows = []
    n = len(y)
    prevalence = float(y.mean())
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
    return {
        "sex": "male" if row.get("male") == 1 else "female_or_other",
        "age_group": "65-74" if row.get("age", 0) < 75 else ("75-84" if row.get("age", 0) < 85 else "85+"),
        "albumin_available": "available" if row.get("albumin") is not None else "missing",
        "ventilation_24h": "yes" if row.get("ventilation_24h") == 1 else "no",
        "enteral_24h": "yes" if row.get("enteral_24h") == 1 else "no",
    }


def subgroup_ci(eicu, y, p, n_resamples=400, seed=48):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for i, row in enumerate(eicu):
        for domain, value in subgroup_definitions(row).items():
            grouped[(domain, value)].append(i)
    out = []
    for (domain, value), idx in sorted(grouped.items()):
        if len(idx) < 500:
            continue
        idx = np.array(idx)
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        point = metric_block(yy, pp)
        draws = {"auc": [], "brier": [], "ece": []}
        for _ in range(n_resamples):
            sample = rng.integers(0, len(idx), len(idx))
            ys, ps = yy[sample], pp[sample]
            if ys.sum() == 0 or ys.sum() == len(ys):
                continue
            draws["auc"].append(auc(ys, ps))
            draws["brier"].append(brier(ys, ps))
            draws["ece"].append(ece(ys, ps))
        row = {"domain": domain, "subgroup": value, **point}
        for metric, vals in draws.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            row[f"{metric}_ci_95"] = [float(lo), float(hi)]
        out.append(row)
    return out


def run_split(offset, mimic, eicu, features):
    train_mask = np.array([(i % 5) != offset for i in range(len(mimic))], dtype=bool)
    train = [mimic[i] for i in np.where(train_mask)[0]]
    test = [mimic[i] for i in np.where(~train_mask)[0]]
    Xm_tr, ym_tr, impute, scaler = matrix(train, features)
    w = fit_logistic(Xm_tr, ym_tr)
    Xm_te, ym_te, _, _ = matrix(test, features, impute=impute, scaler=scaler)
    Xe, ye, _, _ = matrix(eicu, features, impute=impute, scaler=scaler)
    pm = sigmoid(Xm_te @ w)
    pe = sigmoid(Xe @ w)
    eidx = np.arange(len(ye))
    cal = eidx % 5 == offset
    held = ~cal
    platt = fit_platt(logit(pe[cal]), ye[cal])
    pe_platt = sigmoid(platt[0] + platt[1] * logit(pe))
    return {
        "offset": offset,
        "mimic_internal_test": metric_block(ym_te, pm),
        "eicu_external_original": metric_block(ye, pe),
        "eicu_heldout_platt": metric_block(ye[held], pe_platt[held]),
        "platt_parameters": {"intercept": float(platt[0]), "slope": float(platt[1])},
    }, pe, pe_platt, ye


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mimic = load_table(V3_DIR / "mimic_v3_features_server_only.csv")
    eicu = load_table(V3_DIR / "eicu_v3_features_server_only.csv")
    print(f"loaded mimic={len(mimic)} eicu={len(eicu)}", flush=True)

    split_runs = []
    original_predictions = {}
    platt_predictions = {}
    ye = None
    for offset in range(5):
        print(f"split stability offset={offset}", flush=True)
        res, pe, pe_platt, y_ext = run_split(offset, mimic, eicu, FEATURES)
        split_runs.append(res)
        original_predictions[offset] = pe
        platt_predictions[offset] = pe_platt
        ye = y_ext

    stability = {}
    for block in ["mimic_internal_test", "eicu_external_original", "eicu_heldout_platt"]:
        stability[block] = {
            metric: summarize([run[block][metric] for run in split_runs])
            for metric in ["auc", "brier", "ece"]
        }

    print("feature-set logistic benchmark on primary offset 0", flush=True)
    model_benchmark = {}
    train_mask = np.array([(i % 5) != 0 for i in range(len(mimic))], dtype=bool)
    train = [mimic[i] for i in np.where(train_mask)[0]]
    test = [mimic[i] for i in np.where(~train_mask)[0]]
    for name, feats in FEATURE_SETS.items():
        Xm_tr, ym_tr, impute, scaler = matrix(train, feats)
        w = fit_logistic(Xm_tr, ym_tr, l2=0.0 if not feats else 0.02)
        Xm_te, ym_te, _, _ = matrix(test, feats, impute=impute, scaler=scaler)
        Xe, ye, _, _ = matrix(eicu, feats, impute=impute, scaler=scaler)
        model_benchmark[name] = {
            "feature_count": len(feats),
            "mimic_internal_test": metric_block(ym_te, sigmoid(Xm_te @ w)),
            "eicu_external": metric_block(ye, sigmoid(Xe @ w)),
        }

    print("nonlinear MLP benchmark on full features, primary offset 0", flush=True)
    Xm_tr, ym_tr, impute, scaler = matrix(train, FEATURES, add_intercept=False)
    Xm_te, ym_te, _, _ = matrix(test, FEATURES, impute=impute, scaler=scaler, add_intercept=False)
    Xe, ye, _, _ = matrix(eicu, FEATURES, impute=impute, scaler=scaler, add_intercept=False)
    params = fit_mlp(Xm_tr, ym_tr)
    p_mlp_test = predict_mlp(Xm_te, params)
    p_mlp_eicu = predict_mlp(Xe, params)
    model_benchmark["full_features_weighted_mlp"] = {
        "feature_count": len(FEATURES),
        "hidden_units": 24,
        "mimic_internal_test": metric_block(ym_te, p_mlp_test),
        "eicu_external": metric_block(ye, p_mlp_eicu),
    }

    p_original = original_predictions[0]
    p_platt = platt_predictions[0]
    idx = np.arange(len(ye))
    held = idx % 5 != 0
    operating_original = operating_points(ye, p_original)
    operating_platt_heldout = operating_points(ye[held], p_platt[held])
    print("subgroup bootstrap CI for primary original external predictions", flush=True)
    subgroups = subgroup_ci(eicu, ye, p_original)

    result = {
        "protocol": "V7_MODEL_BENCHMARK_STABILITY_OPERATING_PROFILE",
        "source_feature_tables": {
            "mimic": str(V3_DIR / "mimic_v3_features_server_only.csv"),
            "eicu": str(V3_DIR / "eicu_v3_features_server_only.csv"),
        },
        "model_name": "GERI-NutriRisk-ICU",
        "benchmark": model_benchmark,
        "split_stability": {
            "splits": split_runs,
            "summary": stability,
        },
        "operating_profile": {
            "eicu_original_full": operating_original,
            "eicu_platt_heldout": operating_platt_heldout,
        },
        "subgroup_ci_original_external": subgroups,
        "claim_boundary": "Aggregate-only outputs; patient-level features and predictions remain server-only. MIMIC labels fit models; eICU labels are used only for external evaluation and prespecified Platt calibration splits.",
    }
    (OUT / "v7_model_benchmark_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    for name, block in model_benchmark.items():
        for split_name in ["mimic_internal_test", "eicu_external"]:
            m = block[split_name]
            rows.append({
                "model": name,
                "evaluation": split_name,
                "feature_count": block.get("feature_count", ""),
                "auc": m["auc"],
                "brier": m["brier"],
                "ece": m["ece"],
                "n": m["n"],
                "events": m["events"],
            })
    write_csv(OUT / "table_model_class_and_feature_benchmark.csv", rows)
    write_csv(OUT / "table_operating_profile_original.csv", operating_original)
    write_csv(OUT / "table_operating_profile_platt_heldout.csv", operating_platt_heldout)
    write_csv(OUT / "table_subgroup_ci_original.csv", subgroups)

    md = [
        "# V7 Model Benchmark and Operating Profile",
        "",
        "Status date: 2026-08-26",
        "",
        "## Purpose",
        "",
        "V7 adds reviewer-facing evidence beyond the V0--V6 package: model-class benchmarking, five repeated row-index split rotations, operating characteristics at prespecified risk thresholds, and bootstrap confidence intervals for eICU subgroup performance.",
        "",
        "## Key Results",
        "",
        f"- Primary logistic framework external AUROC, offset 0: {model_benchmark['full_geri_nutririsk_icu_logistic']['eicu_external']['auc']:.3f}.",
        f"- Weighted one-hidden-layer MLP external AUROC, offset 0: {model_benchmark['full_features_weighted_mlp']['eicu_external']['auc']:.3f}.",
        f"- Five-split eICU original AUROC mean (SD): {stability['eicu_external_original']['auc']['mean']:.3f} ({stability['eicu_external_original']['auc']['sd']:.3f}).",
        f"- Five-split eICU Platt-held-out Brier mean (SD): {stability['eicu_heldout_platt']['brier']['mean']:.3f} ({stability['eicu_heldout_platt']['brier']['sd']:.3f}).",
        "",
        "## Interpretation",
        "",
        "The interpretable standardized logistic framework remains the primary model because it provides stable external discrimination, explicit calibration updating, and coefficient-level interpretation. The nonlinear MLP benchmark is reported as a sensitivity comparator rather than as the clinical framework.",
        "",
        "## Outputs",
        "",
        "- `v7_model_benchmark_metrics.json`",
        "- `table_model_class_and_feature_benchmark.csv`",
        "- `table_operating_profile_original.csv`",
        "- `table_operating_profile_platt_heldout.csv`",
        "- `table_subgroup_ci_original.csv`",
    ]
    (OUT / "V7_MODEL_BENCHMARK_AND_OPERATING_PROFILE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "primary_external_auc": model_benchmark["full_geri_nutririsk_icu_logistic"]["eicu_external"]["auc"],
        "mlp_external_auc": model_benchmark["full_features_weighted_mlp"]["eicu_external"]["auc"],
        "split_stability_eicu_auc_mean": stability["eicu_external_original"]["auc"]["mean"],
        "subgroup_ci_rows": len(subgroups),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
