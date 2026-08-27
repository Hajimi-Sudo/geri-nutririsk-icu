#!/usr/bin/env python3
"""Export V7 aggregate benchmark assets for manuscript drafting."""

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
V7 = RESULTS / "v7_model_benchmark"


DISPLAY = {
    "intercept_only": "Intercept only",
    "demographics_bmi": "Demographics + BMI",
    "nutrition_support_only": "Nutrition support",
    "nutrition_metabolic_core": "Nutrition/metabolic core",
    "non_nutrition_clinical_baseline": "Non-nutrition clinical baseline",
    "full_geri_nutririsk_icu_logistic": "GERI-NutriRisk-ICU",
    "full_features_weighted_mlp": "Weighted MLP",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: str | float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


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


def export_formatted_tables(metrics: dict) -> None:
    benchmark = read_csv(V7 / "table_model_class_and_feature_benchmark.csv")
    rows_by_model = {}
    for row in benchmark:
        rows_by_model.setdefault(row["model"], {})[row["evaluation"]] = row
    out = []
    for model in [
        "intercept_only",
        "demographics_bmi",
        "nutrition_support_only",
        "nutrition_metabolic_core",
        "non_nutrition_clinical_baseline",
        "full_geri_nutririsk_icu_logistic",
        "full_features_weighted_mlp",
    ]:
        internal = rows_by_model[model]["mimic_internal_test"]
        external = rows_by_model[model]["eicu_external"]
        out.append(
            {
                "model": DISPLAY[model],
                "feature_count": internal["feature_count"],
                "mimic_auc": fmt(internal["auc"]),
                "mimic_brier": fmt(internal["brier"]),
                "eicu_auc": fmt(external["auc"]),
                "eicu_brier": fmt(external["brier"]),
                "eicu_ece": fmt(external["ece"]),
            }
        )
    write_csv(
        TABLE_DIR / "table7_model_benchmark_v7.csv",
        out,
        ["model", "feature_count", "mimic_auc", "mimic_brier", "eicu_auc", "eicu_brier", "eicu_ece"],
    )

    stability = metrics["split_stability"]["summary"]
    stability_rows = []
    for block, label in [
        ("mimic_internal_test", "MIMIC internal test"),
        ("eicu_external_original", "eICU external original"),
        ("eicu_heldout_platt", "eICU held-out Platt recalibrated"),
    ]:
        stability_rows.append(
            {
                "evaluation": label,
                "auc_mean_sd": f"{stability[block]['auc']['mean']:.3f} ({stability[block]['auc']['sd']:.3f})",
                "brier_mean_sd": f"{stability[block]['brier']['mean']:.3f} ({stability[block]['brier']['sd']:.3f})",
                "ece_mean_sd": f"{stability[block]['ece']['mean']:.3f} ({stability[block]['ece']['sd']:.3f})",
                "auc_range": f"{stability[block]['auc']['min']:.3f}-{stability[block]['auc']['max']:.3f}",
            }
        )
    write_csv(TABLE_DIR / "table8_split_stability_v7.csv", stability_rows, list(stability_rows[0]))

    operating = read_csv(V7 / "table_operating_profile_platt_heldout.csv")
    op_rows = []
    for row in operating:
        op_rows.append(
            {
                "threshold": fmt(row["threshold"], 2),
                "alerts_per_100": fmt(row["alerts_per_100"], 1),
                "sensitivity": fmt(row["sensitivity"]),
                "specificity": fmt(row["specificity"]),
                "ppv": fmt(row["ppv"]),
                "npv": fmt(row["npv"]),
                "net_benefit": fmt(row["net_benefit_model"], 4),
            }
        )
    write_csv(
        TABLE_DIR / "table9_operating_profile_platt_v7.csv",
        op_rows,
        ["threshold", "alerts_per_100", "sensitivity", "specificity", "ppv", "npv", "net_benefit"],
    )

    subgroup = read_csv(V7 / "table_subgroup_ci_original.csv")
    subgroup_rows = []
    for row in subgroup:
        auc_ci = json.loads(row["auc_ci_95"].replace("'", '"'))
        brier_ci = json.loads(row["brier_ci_95"].replace("'", '"'))
        subgroup_rows.append(
            {
                "domain": row["domain"],
                "subgroup": row["subgroup"],
                "n": row["n"],
                "events": row["events"],
                "auc_95ci": f"{float(row['auc']):.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
                "brier_95ci": f"{float(row['brier']):.3f} ({brier_ci[0]:.3f}-{brier_ci[1]:.3f})",
            }
        )
    write_csv(
        TABLE_DIR / "table10_subgroup_ci_v7.csv",
        subgroup_rows,
        ["domain", "subgroup", "n", "events", "auc_95ci", "brier_95ci"],
    )


def plot_benchmark(metrics: dict) -> None:
    benchmark = metrics["benchmark"]
    models = [
        "demographics_bmi",
        "nutrition_support_only",
        "nutrition_metabolic_core",
        "non_nutrition_clinical_baseline",
        "full_geri_nutririsk_icu_logistic",
        "full_features_weighted_mlp",
    ]
    labels = [DISPLAY[m] for m in models]
    internal = [benchmark[m]["mimic_internal_test"]["auc"] for m in models]
    external = [benchmark[m]["eicu_external"]["auc"] for m in models]
    y = list(range(len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.barh([i + 0.18 for i in y], internal, height=0.34, color="#366b8b", label="MIMIC internal")
    ax.barh([i - 0.18 for i in y], external, height=0.34, color="#c76f51", label="eICU external")
    ax.set_yticks(y, labels)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.50, 0.90)
    ax.legend(loc="lower right")
    ax.set_title("Model and feature benchmark")

    stability = metrics["split_stability"]["summary"]
    labels2 = ["MIMIC internal", "eICU original", "eICU recalibrated"]
    vals = [
        stability["mimic_internal_test"]["auc"]["mean"],
        stability["eicu_external_original"]["auc"]["mean"],
        stability["eicu_heldout_platt"]["auc"]["mean"],
    ]
    errs = [
        stability["mimic_internal_test"]["auc"]["sd"],
        stability["eicu_external_original"]["auc"]["sd"],
        stability["eicu_heldout_platt"]["auc"]["sd"],
    ]
    ax2 = axes[1]
    ax2.bar(labels2, vals, yerr=errs, color=["#5a7f5a", "#7867a5", "#bf8a3d"], capsize=4)
    ax2.set_ylim(0.68, 0.86)
    ax2.set_ylabel("AUROC mean across five splits")
    ax2.set_title("Split stability")
    ax2.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_benchmark_stability.png")
    plt.close(fig)


def update_evidence_pack(metrics: dict) -> None:
    path = RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md"
    text = path.read_text(encoding="utf-8")
    if "## V7 Reviewer-Facing Extensions" in text:
        text = text.split("## V7 Reviewer-Facing Extensions")[0].rstrip() + "\n"
    stability = metrics["split_stability"]["summary"]
    addition = f"""

## V7 Reviewer-Facing Extensions

- Model benchmark: the primary standardized logistic GERI-NutriRisk-ICU model achieved eICU external AUROC {metrics['benchmark']['full_geri_nutririsk_icu_logistic']['eicu_external']['auc']:.3f}, whereas a weighted one-hidden-layer MLP achieved external AUROC {metrics['benchmark']['full_features_weighted_mlp']['eicu_external']['auc']:.3f} despite higher internal AUROC.
- Five split rotations: eICU original AUROC mean (SD) {stability['eicu_external_original']['auc']['mean']:.3f} ({stability['eicu_external_original']['auc']['sd']:.3f}); eICU held-out Platt Brier mean (SD) {stability['eicu_heldout_platt']['brier']['mean']:.3f} ({stability['eicu_heldout_platt']['brier']['sd']:.3f}).
- Added operating characteristics at prespecified thresholds and bootstrap subgroup confidence intervals for the eICU external cohort.
"""
    path.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((V7 / "v7_model_benchmark_metrics.json").read_text(encoding="utf-8"))
    setup_plot()
    export_formatted_tables(metrics)
    plot_benchmark(metrics)
    update_evidence_pack(metrics)
    print(json.dumps({"figure": str(FIG_DIR / "fig7_benchmark_stability.png"), "tables": 4}, indent=2))


if __name__ == "__main__":
    main()
