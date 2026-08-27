#!/usr/bin/env python3
"""Export V10/V11 model-lock manuscript assets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05-results"
WRITING = ROOT / "06-writing"
FIG_DIR = WRITING / "figures"
TABLE_DIR = WRITING / "tables"
V10 = RESULTS / "v10_sota_completion_lock"
V11 = RESULTS / "v11_tabnet_neural_sota"

DISPLAY = {
    "Logistic_GERI_NutriRisk_ICU": "Logistic GERI-NutriRisk-ICU",
    "XGBoost_hist": "XGBoost",
    "LightGBM_gbdt": "LightGBM",
    "CatBoost_ordered": "CatBoost",
    "Sklearn_HistGradientBoosting": "HistGradientBoosting",
    "RandomForest_balanced": "Random forest",
    "ExtraTrees_balanced": "ExtraTrees",
    "Sklearn_MLP": "MLP",
    "TabNet_neural": "TabNet",
}


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


def export_tables(v10: dict, v11: dict) -> None:
    rows = pd.DataFrame(v10["rows_sorted"]).copy()
    rows["model"] = rows["model"].map(lambda x: DISPLAY.get(x, x))
    for col in [
        "mimic_auc",
        "mimic_brier",
        "eicu_auc_original",
        "eicu_brier_original",
        "eicu_ece_original",
        "eicu_auc_platt_heldout",
        "eicu_brier_platt_heldout",
        "eicu_ece_platt_heldout",
    ]:
        rows[col] = rows[col].map(lambda x: f"{float(x):.3f}")
    rows.to_csv(TABLE_DIR / "table16_sota_completion_lock_v10.csv", index=False)

    tabnet = pd.DataFrame([v11["lightgbm_reference"], v11["tabnet"]]).copy()
    tabnet["model"] = tabnet["model"].map(lambda x: DISPLAY.get(x, x))
    for col in [
        "mimic_auc",
        "mimic_brier",
        "eicu_auc_original",
        "eicu_brier_original",
        "eicu_ece_original",
        "eicu_auc_platt_heldout",
        "eicu_brier_platt_heldout",
        "eicu_ece_platt_heldout",
    ]:
        tabnet[col] = tabnet[col].map(lambda x: f"{float(x):.3f}")
    tabnet.to_csv(TABLE_DIR / "table17_tabnet_neural_sota_v11.csv", index=False)


def plot_model_lock(v10: dict, v11: dict) -> None:
    rows = pd.DataFrame(v10["rows_sorted"]).copy()
    rows = pd.concat([rows, pd.DataFrame([v11["tabnet"]])], ignore_index=True)
    rows["display"] = rows["model"].map(lambda x: DISPLAY.get(x, x))
    rows = rows.sort_values("eicu_auc_platt_heldout")

    colors = ["#6f7f8f"] * len(rows)
    for i, model in enumerate(rows["model"]):
        if model == "LightGBM_gbdt":
            colors[i] = "#2f6f73"
        elif model == "TabNet_neural":
            colors[i] = "#9b5f69"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), gridspec_kw={"width_ratios": [1.1, 0.9]})
    axes[0].barh(rows["display"], rows["eicu_auc_platt_heldout"], color=colors)
    axes[0].set_xlabel("eICU held-out Platt AUROC")
    axes[0].set_xlim(0.60, 0.82)
    axes[0].set_title("Bounded SOTA completion")
    axes[0].axvline(v10["model_lock"]["lightgbm_external_auc"], color="#222222", lw=1, ls="--")

    brier_rows = rows.sort_values("eicu_brier_platt_heldout", ascending=False)
    brier_colors = ["#6f7f8f"] * len(brier_rows)
    for i, model in enumerate(brier_rows["model"]):
        if model == "LightGBM_gbdt":
            brier_colors[i] = "#2f6f73"
        elif model == "TabNet_neural":
            brier_colors[i] = "#9b5f69"
    axes[1].barh(brier_rows["display"], brier_rows["eicu_brier_platt_heldout"], color=brier_colors)
    axes[1].set_xlabel("eICU held-out Platt Brier")
    axes[1].set_xlim(0.08, 0.112)
    axes[1].set_title("Recalibrated error")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_sota_completion_model_lock.png")
    plt.close(fig)


def update_evidence_pack(v10: dict, v11: dict) -> None:
    path = RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md"
    text = path.read_text(encoding="utf-8")
    for marker in ["## V10 SOTA Completion And Model Lock", "## V11 TabNet Neural SOTA Comparator"]:
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
    lgb = next(r for r in v10["rows_sorted"] if r["model"] == "LightGBM_gbdt")
    hgb = next(r for r in v10["rows_sorted"] if r["model"] == "Sklearn_HistGradientBoosting")
    addition = f"""

## V10 SOTA Completion And Model Lock

- Bounded SOTA completion added HistGradientBoosting, balanced random forest, ExtraTrees, and MLP to the existing logistic, XGBoost, LightGBM, and CatBoost queue.
- HistGradientBoosting had the numerically highest eICU held-out Platt AUROC ({hgb['eicu_auc_platt_heldout']:.3f}), but the advantage over LightGBM was only {v10['model_lock']['best_minus_lightgbm_external_auc']:.5f}, far below the prespecified 0.015 material replacement threshold.
- LightGBM retained better recalibrated Brier score than HistGradientBoosting ({lgb['eicu_brier_platt_heldout']:.3f} vs {hgb['eicu_brier_platt_heldout']:.3f}) and remains locked as the primary performance head.

## V11 TabNet Neural SOTA Comparator

- TabNet was run on GPU as a bounded neural tabular comparator with MIMIC-only fitting/early stopping and eICU external evaluation.
- TabNet did not improve transportability: eICU external AUROC {v11['tabnet']['eicu_auc_original']:.3f}; eICU held-out Platt AUROC {v11['tabnet']['eicu_auc_platt_heldout']:.3f}; held-out Platt Brier {v11['tabnet']['eicu_brier_platt_heldout']:.3f}.
- The final model decision is to keep LightGBM as the GERI-NutriRisk-ICU performance head and report logistic regression as the transparent clinical anchor.
"""
    path.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    v10 = json.loads((V10 / "v10_sota_completion_lock_metrics.json").read_text(encoding="utf-8"))
    v11 = json.loads((V11 / "v11_tabnet_neural_sota_metrics.json").read_text(encoding="utf-8"))
    setup_plot()
    export_tables(v10, v11)
    plot_model_lock(v10, v11)
    update_evidence_pack(v10, v11)
    print(
        json.dumps(
            {
                "tables": [
                    str(TABLE_DIR / "table16_sota_completion_lock_v10.csv"),
                    str(TABLE_DIR / "table17_tabnet_neural_sota_v11.csv"),
                ],
                "figure": str(FIG_DIR / "fig10_sota_completion_model_lock.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
