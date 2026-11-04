#!/usr/bin/env python3
"""V16 Hospital-Cluster Bootstrap Robustness Analysis (Primary-Split Evaluation).

========================================================================================
CRITICAL NOTICE ON METHODOLOGY & REPRODUCIBILITY SCOPE:
========================================================================================
This script represents the historical V16 development phase (pre-revision), which
evaluates hospital-cluster bootstrap stability and center-stratified performance
within the PRIMARY patient-level stratified split (10% eICU update, 10% eICU calibration,
80% eICU test partition partitioned via modulo indexing).

If you are looking for the STRICT CENTER-LEVEL LEAVE-HOSPITAL-OUT VALIDATION
(where 20 hospitals are allocated to update, 20 hospitals to calibration, and 167 hospitals
are strictly held out as unseen centers before any model fitting or imputation),
please refer to the dedicated standalone script:
    --> run_leave_hospital_out_validation.py (EXP-001)

Summary of the dual validation architecture:
  1. run_v16_hospital_cluster_robustness.py (this script):
     Primary patient-level split (i % 10), evaluating intra-cohort cluster bootstrap
     and center-stratified performance across participating sites.
  2. run_leave_hospital_out_validation.py (EXP-001):
     Strict leave-hospital-out partition (20 / 20 / 167 hospitals), evaluating
     zero-leakage generalizability to 167 completely independent, unseen hospitals.
========================================================================================
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
EICU = Path("/root/autodl-tmp/eicu/base")
V13 = ROOT / "v13_strong_clinical_feature_upgrade"
OUT = ROOT / "v16_hospital_cluster_robustness"


def load_table(path: Path):
    ids, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        features = [c for c in reader.fieldnames or [] if c not in {"id", "y"}]
        for row in reader:
            ids.append(row["id"])
            y.append(int(float(row["y"])))
            X.append([float(row[c]) if row.get(c, "") != "" else np.nan for c in features])
    return ids, np.asarray(X, dtype=float), np.asarray(y, dtype=int), features


def load_hospitals() -> dict[str, str]:
    out = {}
    with gzip.open(EICU / "patient.csv.gz", "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["patientunitstayid"]] = row.get("hospitalid", "")
    return out


def load_apache(version: str) -> dict[str, float]:
    out = {}
    with gzip.open(EICU / "apachePatientResult.csv.gz", "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("apacheversion") != version:
                continue
            try:
                p = float(row.get("predictedhospitalmortality", ""))
            except Exception:
                continue
            if 0 <= p <= 1:
                out[row["patientunitstayid"]] = p
    return out


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


def metrics(y, p):
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc(y, p),
        "brier": brier(y, p),
        "ece_q10": ece(y, p),
    }


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


def fit_model(X_m, y_m, X_e, y_e, m_train, e_update_fit):
    scale_pos_weight = float((len(y_m[m_train]) + len(y_e[e_update_fit]) - y_m[m_train].sum() - y_e[e_update_fit].sum()) / max(y_m[m_train].sum() + y_e[e_update_fit].sum(), 1))
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
    model.fit(
        np.vstack([X_m[m_train], X_e[e_update_fit]]),
        np.concatenate([y_m[m_train], y_e[e_update_fit]]),
    )
    return model


def cluster_bootstrap(ids, y, model_p, apache_p=None, n_resamples=500, seed=20260870):
    by_hospital = defaultdict(list)
    for i, h in enumerate(ids):
        by_hospital[h].append(i)
    hospitals = sorted(by_hospital)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_resamples):
        sampled = rng.choice(hospitals, size=len(hospitals), replace=True)
        idx = np.asarray([i for h in sampled for i in by_hospital[h]], dtype=int)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        row = {
            "model_auc": auc(yy, model_p[idx]),
            "model_brier": brier(yy, model_p[idx]),
        }
        if apache_p is not None:
            row["model_minus_apache_auc"] = auc(yy, model_p[idx]) - auc(yy, apache_p[idx])
            row["model_minus_apache_brier"] = brier(yy, model_p[idx]) - brier(yy, apache_p[idx])
        out.append(row)
    return summarize_bootstrap(out)


def summarize_bootstrap(rows):
    result = {}
    for key in rows[0]:
        vals = np.asarray([r[key] for r in rows], dtype=float)
        result[key] = {
            "mean": float(np.mean(vals)),
            "ci_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "n_resamples": len(rows),
        }
    return result


def hospital_rows(hospitals, y, p):
    rows = []
    for h in sorted(set(hospitals)):
        idx = np.asarray([i for i, x in enumerate(hospitals) if x == h], dtype=int)
        yy = y[idx]
        pp = p[idx]
        row = {"hospitalid": h, "n": int(len(idx)), "events": int(yy.sum()), "event_rate": float(yy.mean())}
        if len(idx) >= 50 and yy.sum() > 0 and yy.sum() < len(yy):
            row["auc"] = auc(yy, pp)
            row["brier"] = brier(yy, pp)
        else:
            row["auc"] = ""
            row["brier"] = brier(yy, pp)
        rows.append(row)
    return rows


def subset_metrics(label, mask, y, p):
    return {"label": label, **metrics(y[mask], p[mask])}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    eicu_ids, X_e_raw, y_e, _ = load_table(V13 / "eicu_v13_features_server_only.csv")
    _, X_m_raw, y_m, _ = load_table(V13 / "mimic_v13_features_server_only.csv")
    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_held = ~(e_update_fit | e_update_cal)

    # Train-only median imputation: fitted strictly on MIMIC development training fold
    med = impute_fit(X_m_raw[m_train])
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    model = fit_model(X_m, y_m, X_e, y_e, m_train, e_update_fit)
    p_e = model.predict_proba(X_e)[:, 1]
    w = fit_platt(logit(p_e[e_update_cal]), y_e[e_update_cal])
    p_e = apply_platt(p_e, w)

    hmap = load_hospitals()
    hospitals_all = np.asarray([hmap.get(x, "") for x in eicu_ids])
    held_ids = np.asarray(eicu_ids)[e_held]
    held_hosp = hospitals_all[e_held]
    yh = y_e[e_held]
    ph = p_e[e_held]

    hcounts = Counter(held_hosp)
    top10 = {h for h, _ in hcounts.most_common(10)}
    not_top10 = np.asarray([h not in top10 for h in held_hosp])
    hospital_metric_rows = hospital_rows(held_hosp.tolist(), yh, ph)

    sensitivity = [
        subset_metrics("all_heldout_hospitals", np.ones(len(yh), dtype=bool), yh, ph),
        subset_metrics("excluding_top10_hospitals_by_n", not_top10, yh, ph),
        subset_metrics("hospitals_with_n_ge_100", np.asarray([hcounts[h] >= 100 for h in held_hosp]), yh, ph),
        subset_metrics("hospitals_with_n_ge_200", np.asarray([hcounts[h] >= 200 for h in held_hosp]), yh, ph),
    ]

    apache_results = {}
    for version in ["IV", "IVa"]:
        amap = load_apache(version)
        ap = np.asarray([amap.get(x, np.nan) for x in held_ids])
        ok = np.isfinite(ap)
        apache_results[version] = {
            "coverage_n": int(ok.sum()),
            "model_same_subset": metrics(yh[ok], ph[ok]),
            "apache_same_subset": metrics(yh[ok], ap[ok]),
            "point_differences": {
                "auc": auc(yh[ok], ph[ok]) - auc(yh[ok], ap[ok]),
                "brier": brier(yh[ok], ph[ok]) - brier(yh[ok], ap[ok]),
            },
            "hospital_cluster_bootstrap": cluster_bootstrap(held_hosp[ok].tolist(), yh[ok], ph[ok], ap[ok]),
        }

    result = {
        "protocol": "V16_HOSPITAL_CLUSTER_ROBUSTNESS",
        "status_date": "2026-08-26",
        "heldout_hospitals": len(set(held_hosp)),
        "heldout_n": int(len(yh)),
        "heldout_events": int(yh.sum()),
        "heldout_model": metrics(yh, ph),
        "sensitivity": sensitivity,
        "hospital_cluster_bootstrap_model": cluster_bootstrap(held_hosp.tolist(), yh, ph),
        "apache_same_subset": apache_results,
        "top10_hospitals_by_heldout_n": hcounts.most_common(10),
        "claim_boundary": (
            "Hospital-cluster results assess robustness across the multicenter eICU validation "
            "dataset. They do not substitute for prospective implementation validation."
        ),
    }
    (OUT / "v16_hospital_cluster_robustness_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (OUT / "table_v16_hospital_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hospital_metric_rows[0]))
        writer.writeheader()
        writer.writerows(hospital_metric_rows)
    with (OUT / "table_v16_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)

    auc_vals = [float(r["auc"]) for r in hospital_metric_rows if r["auc"] != "" and r["n"] >= 100]
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(auc_vals, bins=24, color="#3b6f7f", edgecolor="white")
    plt.xlabel("Hospital-level AUROC (hospitals with n >= 100)")
    plt.ylabel("Number of hospitals")
    plt.tight_layout()
    plt.savefig(OUT / "fig_v16_hospital_auc_distribution.png", dpi=300)
    plt.close()

    rows_plot = sorted([r for r in hospital_metric_rows if r["auc"] != "" and r["n"] >= 200], key=lambda r: float(r["auc"]))
    plt.figure(figsize=(7.5, 7.0))
    plt.scatter([r["n"] for r in rows_plot], [float(r["auc"]) for r in rows_plot], s=20, alpha=0.75, color="#71508a")
    plt.xscale("log")
    plt.xlabel("Held-out stays per hospital (log scale)")
    plt.ylabel("Hospital-level AUROC")
    plt.tight_layout()
    plt.savefig(OUT / "fig_v16_hospital_size_auc_scatter.png", dpi=300)
    plt.close()

    md = [
        "# V16 Hospital-Cluster Robustness",
        "",
        "Status date: 2026-08-26",
        "",
        f"Held-out eICU hospitals: {result['heldout_hospitals']}; held-out stays: {result['heldout_n']}.",
        "",
        "## Main Robustness",
    ]
    for row in sensitivity:
        md.append(f"- {row['label']}: AUROC {row['auc']:.3f}, Brier {row['brier']:.3f}, n {row['n']}.")
    md += ["", "## APACHE Same-Subset Hospital-Cluster Bootstrap"]
    for version, block in apache_results.items():
        diff = block["hospital_cluster_bootstrap"]["model_minus_apache_auc"]
        md.append(
            f"- APACHE-{version}: model-minus-APACHE AUROC mean {diff['mean']:.3f}, "
            f"95% cluster CI {diff['ci_95'][0]:.3f} to {diff['ci_95'][1]:.3f}."
        )
    md += ["", "## Boundary", "", result["claim_boundary"]]
    (OUT / "V16_HOSPITAL_CLUSTER_ROBUSTNESS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "heldout_model": result["heldout_model"],
        "sensitivity": sensitivity,
        "apache_same_subset": apache_results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
