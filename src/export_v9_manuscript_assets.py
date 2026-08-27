#!/usr/bin/env python3
"""Export V9 LightGBM primary package manuscript assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05-results"
WRITING = ROOT / "06-writing"
FIG_DIR = WRITING / "figures"
TABLE_DIR = WRITING / "tables"
V9 = RESULTS / "v9_lightgbm_primary"


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def ci_text(ci: dict, metric: str) -> str:
    item = ci[metric]
    lo, hi = item["ci_95"]
    return f"{item['estimate']:.3f} ({lo:.3f}-{hi:.3f})"


def export_tables(metrics: dict) -> None:
    ci = metrics["bootstrap_ci"]
    points = metrics["point_estimates"]
    rows = []
    for key, label in [
        ("mimic_internal_test", "MIMIC-IV internal test"),
        ("eicu_external_original_full", "eICU external full, original"),
        ("eicu_external_platt_heldout", "eICU held-out, Platt recalibrated"),
    ]:
        rows.append(
            {
                "evaluation": label,
                "n": points[key]["n"],
                "events": points[key]["events"],
                "auroc_95ci": ci_text(ci[key], "auc"),
                "brier_95ci": ci_text(ci[key], "brier"),
                "ece_95ci": ci_text(ci[key], "ece"),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table12_lightgbm_primary_metrics_v9.csv", index=False)

    op = pd.DataFrame(metrics["operating_profile"]["eicu_platt_heldout"])
    op_out = op[["threshold", "alerts_per_100", "sensitivity", "specificity", "ppv", "npv", "net_benefit_model"]].copy()
    for col in ["sensitivity", "specificity", "ppv", "npv", "net_benefit_model"]:
        op_out[col] = op_out[col].map(lambda x: f"{x:.3f}")
    op_out["alerts_per_100"] = op_out["alerts_per_100"].map(lambda x: f"{x:.1f}")
    op_out.to_csv(TABLE_DIR / "table13_lightgbm_operating_profile_v9.csv", index=False)

    sub = pd.DataFrame(metrics["subgroups_original_external"])
    sub_out = sub[["domain", "subgroup", "n", "events", "auc", "brier", "ece"]].copy()
    for col in ["auc", "brier", "ece"]:
        sub_out[col] = sub_out[col].map(lambda x: f"{x:.3f}")
    sub_out.to_csv(TABLE_DIR / "table14_lightgbm_subgroups_v9.csv", index=False)

    imp = pd.DataFrame(metrics["feature_importance"]).head(20)
    imp.to_csv(TABLE_DIR / "table15_lightgbm_feature_importance_v9.csv", index=False)


def plot_lightgbm(metrics: dict) -> None:
    roc = pd.DataFrame(metrics["plot_data"]["eicu_roc_original"])
    cal_raw = pd.DataFrame(metrics["plot_data"]["eicu_calibration_original"])
    cal_platt = pd.DataFrame(metrics["plot_data"]["eicu_calibration_platt_heldout"])
    dca_raw = pd.DataFrame(metrics["plot_data"]["eicu_decision_curve_original"])
    dca_platt = pd.DataFrame(metrics["plot_data"]["eicu_decision_curve_platt_heldout"])
    imp = pd.DataFrame(metrics["feature_importance"]).head(14).sort_values("mean_abs_shap_contribution")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    ax = axes[0, 0]
    ax.plot(roc["fpr"], roc["tpr"], color="#366b8b", lw=2, label="LightGBM eICU AUROC 0.793")
    ax.plot([0, 1], [0, 1], color="#777777", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("External discrimination")

    ax = axes[0, 1]
    ax.plot(cal_raw["pred_mean"], cal_raw["obs_rate"], marker="o", label="Original external", color="#9b5f69")
    ax.plot(cal_platt["pred_mean"], cal_platt["obs_rate"], marker="s", label="Platt held-out", color="#5a7f5a")
    ax.plot([0, 0.55], [0, 0.55], color="#777777", lw=1, ls="--")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left")
    ax.set_title("External calibration")

    ax = axes[1, 0]
    ax.plot(dca_raw["threshold"], dca_raw["net_benefit_model"], label="Original external", color="#366b8b")
    ax.plot(dca_platt["threshold"], dca_platt["net_benefit_model"], label="Platt held-out", color="#c76f51")
    ax.plot(dca_raw["threshold"], dca_raw["net_benefit_treat_all"], label="Treat all", color="#777777", ls="--")
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right")
    ax.set_title("Decision curve")

    ax = axes[1, 1]
    ax.barh(imp["feature"], imp["mean_abs_shap_contribution"], color="#6a7f3f")
    ax.set_xlabel("Mean absolute contribution")
    ax.set_title("LightGBM feature contribution")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_lightgbm_primary_package.png")
    plt.close(fig)


def update_evidence_pack(metrics: dict) -> None:
    path = RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md"
    text = path.read_text(encoding="utf-8")
    if "## V9 LightGBM Primary Performance Head" in text:
        text = text.split("## V9 LightGBM Primary Performance Head")[0].rstrip() + "\n"
    ci = metrics["bootstrap_ci"]
    pts = metrics["point_estimates"]
    addition = f"""

## V9 LightGBM Primary Performance Head

- Final upgraded framework interpretation: GERI-NutriRisk-ICU contains a harmonized feature schema, leakage-controlled multicohort protocol, transparent logistic anchor model, SOTA tabular performance head, external-domain recalibration, operating-profile layer, and explanation layer.
- LightGBM primary performance head: MIMIC internal AUROC {ci_text(ci['mimic_internal_test'], 'auc')}; eICU external AUROC {ci_text(ci['eicu_external_original_full'], 'auc')}; eICU held-out Platt AUROC {ci_text(ci['eicu_external_platt_heldout'], 'auc')}.
- Recalibrated LightGBM eICU held-out calibration: Brier {ci_text(ci['eicu_external_platt_heldout'], 'brier')}; ECE {ci_text(ci['eicu_external_platt_heldout'], 'ece')}.
- Leading LightGBM contribution features: {', '.join([x['feature'] for x in metrics['feature_importance'][:8]])}.
- The manuscript should not claim treatment benefit or standalone nutrition causality; it can claim externally validated, recalibrated, explainable nutrition-risk-relevant stratification.
"""
    path.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((V9 / "v9_lightgbm_primary_metrics.json").read_text(encoding="utf-8"))
    setup_plot()
    export_tables(metrics)
    plot_lightgbm(metrics)
    update_evidence_pack(metrics)
    print(json.dumps({"figure": str(FIG_DIR / "fig9_lightgbm_primary_package.png"), "tables": 4}, indent=2))


if __name__ == "__main__":
    main()
