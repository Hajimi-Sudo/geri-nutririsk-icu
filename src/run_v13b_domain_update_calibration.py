#!/usr/bin/env python3
"""V13b domain-update calibration audit.

Uses the existing V13 server-only feature tables. No raw APACHE outputs are used
as model inputs. APACHE predicted mortality is used only as a same-subset
contextual comparator.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu")
EICU = Path("/root/autodl-tmp/eicu/base")
V13 = ROOT / "v13_strong_clinical_feature_upgrade"
OUT = ROOT / "v13b_domain_update_calibration"


def load_table(path: Path):
    rows, y, X = [], [], []
    with path.open("rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        features = [c for c in reader.fieldnames if c not in {"id", "y"}]
        for row in reader:
            rows.append({"id": row["id"]})
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


def fit_lgbm(X, y, seed=20260830, strong=True):
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=1400 if strong else 900,
        learning_rate=0.02 if strong else 0.03,
        num_leaves=31 if strong else 20,
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


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def apache_predictions():
    out = {}
    with gzip.open(EICU / "apachePatientResult.csv.gz", "rt", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = fnum(row.get("predictedhospitalmortality", ""))
            if p is None or not (0 <= p <= 1):
                continue
            out.setdefault(row["patientunitstayid"], {})[row["apacheversion"]] = p
    return out


def same_subset(rows, y, eval_mask, pred_map):
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
        idx = np.asarray(idx, dtype=int)
        apred = np.asarray(apred, dtype=float)
        if len(idx) == 0:
            continue
        block = {"APACHE": metric_block(y[idx], apred), "coverage_n": int(len(idx))}
        for name, p in pred_map.items():
            block[name] = metric_block(y[idx], p[idx])
            block[f"{name}_minus_APACHE_auc"] = block[name]["auc"] - block["APACHE"]["auc"]
            block[f"{name}_minus_APACHE_brier"] = block[name]["brier"] - block["APACHE"]["brier"]
        out[f"APACHE_{version}"] = block
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    eicu_rows, X_e_raw, y_e, features = load_table(V13 / "eicu_v13_features_server_only.csv")
    _, X_m_raw, y_m, _ = load_table(V13 / "mimic_v13_features_server_only.csv")
    m_train = np.array([(i % 5) != 0 for i in range(len(y_m))])
    e_cal_all = np.array([(i % 5) == 0 for i in range(len(y_e))])
    e_held = ~e_cal_all
    e_update_fit = np.array([(i % 10) == 0 for i in range(len(y_e))])
    e_update_cal = np.array([(i % 10) == 5 for i in range(len(y_e))])
    e_nested_held = ~(e_update_fit | e_update_cal)

    # Train-only median imputation: fitted strictly on MIMIC development training fold
    med = impute_fit(X_m_raw[m_train])
    X_m = impute_apply(X_m_raw, med)
    X_e = impute_apply(X_e_raw, med)

    # Source-only recalibrated reference.
    source = fit_lgbm(X_m[m_train], y_m[m_train], seed=20260831, strong=True)
    p_source = source.predict_proba(X_e)[:, 1]
    w_source = fit_platt(logit(p_source[e_cal_all]), y_e[e_cal_all])
    p_source_platt = apply_platt(p_source, w_source)

    # Same-calibration-pool update: highest performance but more optimistic
    # calibration because the model and Platt layer see the same eICU pool.
    eicu_update = fit_lgbm(X_e[e_cal_all], y_e[e_cal_all], seed=20260832, strong=True)
    p_eicu_update = eicu_update.predict_proba(X_e)[:, 1]
    w_eicu_update = fit_platt(logit(p_eicu_update[e_cal_all]), y_e[e_cal_all])
    p_eicu_update_platt = apply_platt(p_eicu_update, w_eicu_update)

    combined_update = fit_lgbm(
        np.vstack([X_m[m_train], X_e[e_cal_all]]),
        np.concatenate([y_m[m_train], y_e[e_cal_all]]),
        seed=20260833,
        strong=True,
    )
    p_combined = combined_update.predict_proba(X_e)[:, 1]
    w_combined = fit_platt(logit(p_combined[e_cal_all]), y_e[e_cal_all])
    p_combined_platt = apply_platt(p_combined, w_combined)

    # Nested update: cleaner because eICU model update and Platt calibration
    # use disjoint halves of the calibration pool.
    eicu_nested = fit_lgbm(X_e[e_update_fit], y_e[e_update_fit], seed=20260834, strong=True)
    p_eicu_nested = eicu_nested.predict_proba(X_e)[:, 1]
    w_eicu_nested = fit_platt(logit(p_eicu_nested[e_update_cal]), y_e[e_update_cal])
    p_eicu_nested_platt = apply_platt(p_eicu_nested, w_eicu_nested)

    combined_nested = fit_lgbm(
        np.vstack([X_m[m_train], X_e[e_update_fit]]),
        np.concatenate([y_m[m_train], y_e[e_update_fit]]),
        seed=20260835,
        strong=True,
    )
    p_combined_nested = combined_nested.predict_proba(X_e)[:, 1]
    w_combined_nested = fit_platt(logit(p_combined_nested[e_update_cal]), y_e[e_update_cal])
    p_combined_nested_platt = apply_platt(p_combined_nested, w_combined_nested)

    pred_all = {
        "source_only_platt": p_source_platt,
        "eicu_update_same_pool_platt": p_eicu_update_platt,
        "combined_update_same_pool_platt": p_combined_platt,
        "eicu_update_nested_platt": p_eicu_nested_platt,
        "combined_update_nested_platt": p_combined_nested_platt,
    }
    result = {
        "protocol": "V13B_DOMAIN_UPDATE_CALIBRATION",
        "status_date": "2026-08-26",
        "feature_count": len(features),
        "splits": {
            "eicu_calibration_all": int(e_cal_all.sum()),
            "eicu_update_fit_nested": int(e_update_fit.sum()),
            "eicu_update_cal_nested": int(e_update_cal.sum()),
            "eicu_heldout_main": int(e_held.sum()),
            "eicu_heldout_nested": int(e_nested_held.sum()),
        },
        "main_heldout": {
            "source_only_platt": metric_block(y_e[e_held], p_source_platt[e_held]),
            "eicu_update_same_pool_platt": metric_block(y_e[e_held], p_eicu_update_platt[e_held]),
            "combined_update_same_pool_platt": metric_block(y_e[e_held], p_combined_platt[e_held]),
        },
        "nested_heldout": {
            "eicu_update_nested_platt": metric_block(y_e[e_nested_held], p_eicu_nested_platt[e_nested_held]),
            "combined_update_nested_platt": metric_block(y_e[e_nested_held], p_combined_nested_platt[e_nested_held]),
            "source_only_platt_on_nested_heldout": metric_block(y_e[e_nested_held], p_source_platt[e_nested_held]),
        },
        "apache_same_subset_main_heldout": same_subset(eicu_rows, y_e, e_held, pred_all),
        "apache_same_subset_nested_heldout": same_subset(
            eicu_rows,
            y_e,
            e_nested_held,
            {
                "source_only_platt": p_source_platt,
                "eicu_update_nested_platt": p_eicu_nested_platt,
                "combined_update_nested_platt": p_combined_nested_platt,
            },
        ),
        "claim_boundary": (
            "Same-pool update can be reported only as a local domain-updated performance head. "
            "Nested update is the cleaner calibrated domain-update sensitivity because eICU update fitting "
            "and Platt calibration use disjoint calibration-pool halves. Neither is source-only transport superiority."
        ),
    }
    (OUT / "v13b_domain_update_calibration_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for group_name in ["main_heldout", "nested_heldout"]:
        for name, block in result[group_name].items():
            rows.append({"group": group_name, "model": name, **block})
    with (OUT / "table_v13b_main_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# V13b Domain-Update Calibration",
        "",
        "Status date: 2026-08-26",
        "",
        "## Main held-out eICU",
    ]
    for name, block in result["main_heldout"].items():
        md.append(f"- {name}: AUROC {block['auc']:.3f}, Brier {block['brier']:.3f}, ECE {block['ece_q10']:.3f}.")
    md += ["", "## Nested held-out eICU"]
    for name, block in result["nested_heldout"].items():
        md.append(f"- {name}: AUROC {block['auc']:.3f}, Brier {block['brier']:.3f}, ECE {block['ece_q10']:.3f}.")
    md += ["", "## Claim Boundary", "", result["claim_boundary"]]
    (OUT / "V13B_DOMAIN_UPDATE_CALIBRATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "main_heldout": result["main_heldout"],
        "nested_heldout": result["nested_heldout"],
        "apache_same_subset_main_heldout": result["apache_same_subset_main_heldout"],
        "apache_same_subset_nested_heldout": result["apache_same_subset_nested_heldout"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
