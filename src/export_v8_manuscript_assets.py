#!/usr/bin/env python3
"""Export V8 SOTA benchmark manuscript assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05-results"
WRITING = ROOT / "06-writing"
FIG_DIR = WRITING / "figures"
TABLE_DIR = WRITING / "tables"
V8 = RESULTS / "v8_sota_tabular_benchmark"


DISPLAY = {
    "GERI-NutriRisk-ICU_logistic": "Logistic GERI-NutriRisk-ICU",
    "XGBoost_hist": "XGBoost",
    "LightGBM_gbdt": "LightGBM",
    "CatBoost_ordered": "CatBoost",
}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def export_table(metrics: dict) -> None:
    rows = []
    for key, block in metrics["benchmark"].items():
        recal = metrics["external_recalibration"][key]["eicu_heldout_platt"]
        rows.append(
            {
                "model": DISPLAY.get(key, key),
                "family": block["model_family"],
                "mimic_auc": f"{block['mimic_internal_test']['auc']:.3f}",
                "mimic_brier": f"{block['mimic_internal_test']['brier']:.3f}",
                "eicu_auc_original": f"{block['eicu_external_original']['auc']:.3f}",
                "eicu_brier_original": f"{block['eicu_external_original']['brier']:.3f}",
                "eicu_ece_original": f"{block['eicu_external_original']['ece']:.3f}",
                "eicu_auc_platt_heldout": f"{recal['auc']:.3f}",
                "eicu_brier_platt_heldout": f"{recal['brier']:.3f}",
                "eicu_ece_platt_heldout": f"{recal['ece']:.3f}",
            }
        )
    write_csv(
        TABLE_DIR / "table11_sota_tabular_benchmark_v8.csv",
        rows,
        [
            "model",
            "family",
            "mimic_auc",
            "mimic_brier",
            "eicu_auc_original",
            "eicu_brier_original",
            "eicu_ece_original",
            "eicu_auc_platt_heldout",
            "eicu_brier_platt_heldout",
            "eicu_ece_platt_heldout",
        ],
    )


def plot_sota(metrics: dict) -> None:
    models = ["GERI-NutriRisk-ICU_logistic", "XGBoost_hist", "LightGBM_gbdt", "CatBoost_ordered"]
    labels = [DISPLAY[m] for m in models]
    mimic_auc = [metrics["benchmark"][m]["mimic_internal_test"]["auc"] for m in models]
    eicu_auc = [metrics["benchmark"][m]["eicu_external_original"]["auc"] for m in models]
    eicu_brier_raw = [metrics["benchmark"][m]["eicu_external_original"]["brier"] for m in models]
    eicu_brier_platt = [metrics["external_recalibration"][m]["eicu_heldout_platt"]["brier"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    x = range(len(models))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], mimic_auc, width=width, color="#366b8b", label="MIMIC internal")
    axes[0].bar([i + width / 2 for i in x], eicu_auc, width=width, color="#c76f51", label="eICU external")
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.68, 0.91)
    axes[0].set_title("Discrimination")
    axes[0].legend(loc="lower left")

    axes[1].bar([i - width / 2 for i in x], eicu_brier_raw, width=width, color="#9b5f69", label="External original")
    axes[1].bar([i + width / 2 for i in x], eicu_brier_platt, width=width, color="#5a7f5a", label="Held-out Platt")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].set_ylabel("Brier score")
    axes[1].set_title("External calibration")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_sota_tabular_benchmark.png")
    plt.close(fig)


def update_evidence_pack(metrics: dict) -> None:
    path = RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md"
    text = path.read_text(encoding="utf-8")
    if "## V8 SOTA Tabular Benchmark" in text:
        text = text.split("## V8 SOTA Tabular Benchmark")[0].rstrip() + "\n"
    lgb = metrics["benchmark"]["LightGBM_gbdt"]
    lgb_recal = metrics["external_recalibration"]["LightGBM_gbdt"]["eicu_heldout_platt"]
    logistic = metrics["benchmark"]["GERI-NutriRisk-ICU_logistic"]
    addition = f"""

## V8 SOTA Tabular Benchmark

- SOTA comparator queue completed: XGBoost {metrics['environment_modules'].get('xgboost')}, LightGBM {metrics['environment_modules'].get('lightgbm')}, and CatBoost {metrics['environment_modules'].get('catboost')}.
- LightGBM was the strongest external-discrimination comparator: MIMIC internal AUROC {lgb['mimic_internal_test']['auc']:.3f}; eICU external AUROC {lgb['eicu_external_original']['auc']:.3f}.
- Original LightGBM external calibration was poor (Brier {lgb['eicu_external_original']['brier']:.3f}; ECE {lgb['eicu_external_original']['ece']:.3f}), but prespecified eICU-domain Platt recalibration improved held-out Brier to {lgb_recal['brier']:.3f} and ECE to {lgb_recal['ece']:.3f}.
- Logistic GERI-NutriRisk-ICU remains the most transparent coefficient-level anchor (eICU external AUROC {logistic['eicu_external_original']['auc']:.3f}; held-out Platt Brier {metrics['external_recalibration']['GERI-NutriRisk-ICU_logistic']['eicu_heldout_platt']['brier']:.3f}); manuscript framing should present GERI-NutriRisk-ICU as a framework with LightGBM as the strongest SOTA performance comparator.
"""
    path.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((V8 / "v8_sota_tabular_benchmark_metrics.json").read_text(encoding="utf-8"))
    setup_plot()
    export_table(metrics)
    plot_sota(metrics)
    update_evidence_pack(metrics)
    print(json.dumps({"figure": str(FIG_DIR / "fig8_sota_tabular_benchmark.png"), "tables": 1}, indent=2))


if __name__ == "__main__":
    main()
