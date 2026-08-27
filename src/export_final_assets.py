#!/usr/bin/env python3
"""Export final manuscript tables, PNG figures, and evidence pack."""

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


def load_json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def ci_text(block: dict, metric: str, digits: int = 3) -> str:
    item = block[metric]
    lo, hi = item["ci_95"]
    return f"{item['estimate']:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def export_tables(v3: dict, v4: dict, v5: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cohorts = []
    for name, summary in [
        ("MIMIC-IV development/internal validation", v3["mimic_summary"]),
        ("eICU-CRD external validation", v3["eicu_summary"]),
    ]:
        cohorts.append(
            {
                "cohort": name,
                "n": summary["n"],
                "events": summary["events"],
                "event_rate": f"{summary['event_rate']:.4f}",
                "ventilation_24h_positive": summary["severity_positive"]["ventilation_24h"],
                "dialysis_24h_positive": summary["severity_positive"]["dialysis_24h"],
                "urine_24h_ml_median": summary["urine_summary"]["median"],
                "gcs_24h_median": summary["gcs_summary"]["median"],
            }
        )
    save_csv(
        TABLE_DIR / "table1_cohort_summary.csv",
        cohorts,
        [
            "cohort",
            "n",
            "events",
            "event_rate",
            "ventilation_24h_positive",
            "dialysis_24h_positive",
            "urine_24h_ml_median",
            "gcs_24h_median",
        ],
    )

    ablation_rows = []
    for model, result in v4["feature_set_results"].items():
        ablation_rows.append(
            {
                "model": model,
                "mimic_internal_auc": result["mimic_internal_test"]["auc"],
                "mimic_internal_brier": result["mimic_internal_test"]["brier"],
                "eicu_external_auc": result["eicu_external"]["auc"],
                "eicu_external_brier": result["eicu_external"]["brier"],
            }
        )
    save_csv(
        TABLE_DIR / "table2_feature_family_ablation.csv",
        ablation_rows,
        ["model", "mimic_internal_auc", "mimic_internal_brier", "eicu_external_auc", "eicu_external_brier"],
    )

    subgroup_rows = []
    for domain, groups in v4["v3_eicu_subgroups_original"].items():
        for group, result in groups.items():
            subgroup_rows.append(
                {
                    "subgroup_domain": domain,
                    "subgroup": group,
                    "n": result["n"],
                    "events": result["events"],
                    "event_rate": result["event_rate"],
                    "auc": result["auc"],
                    "brier": result["brier"],
                }
            )
    save_csv(
        TABLE_DIR / "table3_eicu_subgroup_performance.csv",
        subgroup_rows,
        ["subgroup_domain", "subgroup", "n", "events", "event_rate", "auc", "brier"],
    )

    ci = v5["bootstrap_ci"]
    perf_rows = [
        {
            "evaluation": "MIMIC-IV internal test",
            "n": v5["point_estimates"]["mimic_internal_test"]["n"],
            "events": v5["point_estimates"]["mimic_internal_test"]["events"],
            "auroc_95ci": ci_text(ci["mimic_internal_test"], "auc"),
            "brier_95ci": ci_text(ci["mimic_internal_test"], "brier"),
            "ece_95ci": ci_text(ci["mimic_internal_test"], "ece"),
        },
        {
            "evaluation": "eICU external full, original",
            "n": v5["point_estimates"]["eicu_external_original_full"]["n"],
            "events": v5["point_estimates"]["eicu_external_original_full"]["events"],
            "auroc_95ci": ci_text(ci["eicu_external_original_full"], "auc"),
            "brier_95ci": ci_text(ci["eicu_external_original_full"], "brier"),
            "ece_95ci": ci_text(ci["eicu_external_original_full"], "ece"),
        },
        {
            "evaluation": "eICU held-out, Platt recalibrated",
            "n": v5["point_estimates"]["eicu_external_platt_heldout"]["n"],
            "events": v5["point_estimates"]["eicu_external_platt_heldout"]["events"],
            "auroc_95ci": ci_text(ci["eicu_external_platt_heldout"], "auc"),
            "brier_95ci": ci_text(ci["eicu_external_platt_heldout"], "brier"),
            "ece_95ci": ci_text(ci["eicu_external_platt_heldout"], "ece"),
        },
    ]
    save_csv(TABLE_DIR / "table4_primary_metrics_with_ci.csv", perf_rows, list(perf_rows[0]))

    coef_rows = [
        {"feature": k, "standardized_coefficient": v}
        for k, v in sorted(v3["coefficients"].items(), key=lambda item: abs(item[1]), reverse=True)
        if k != "intercept"
    ]
    save_csv(TABLE_DIR / "table5_standardized_coefficients.csv", coef_rows, ["feature", "standardized_coefficient"])

    v6_path = RESULTS / "v6_nutrition_increment" / "v6_nutrition_increment_metrics.json"
    if v6_path.exists():
        v6 = json.loads(v6_path.read_text(encoding="utf-8"))
        increment_rows = []
        for name, result in v6["feature_sets"].items():
            ext = result["eicu_external"]
            increment_rows.append(
                {
                    "model": name,
                    "eicu_auc": ext["auc"],
                    "eicu_brier": ext["brier"],
                    "eicu_ece": ext["ece"],
                }
            )
        for item in v6["paired_external_deltas"]:
            row = {
                "model": item["comparison"],
                "eicu_auc": f"{item['point']['auc']:.5f} ({item['ci_95']['auc'][0]:.5f}-{item['ci_95']['auc'][1]:.5f})",
                "eicu_brier": f"{item['point']['brier']:.5f} ({item['ci_95']['brier'][0]:.5f}-{item['ci_95']['brier'][1]:.5f})",
                "eicu_ece": f"{item['point']['ece']:.5f} ({item['ci_95']['ece'][0]:.5f}-{item['ci_95']['ece'][1]:.5f})",
            }
            increment_rows.append(row)
        save_csv(
            TABLE_DIR / "table6_nutrition_construct_increment.csv",
            increment_rows,
            ["model", "eicu_auc", "eicu_brier", "eicu_ece"],
        )


def plot_workflow(v3: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.axis("off")
    boxes = [
        ("MIMIC-IV\\nolder ICU stays\\nn=33,671", 0.08, 0.58),
        ("Feature mapping\\nnutrition, labs, vitals, severity", 0.33, 0.58),
        ("GERI-NutriRisk-ICU\\nV3 + severity", 0.58, 0.58),
        ("eICU-CRD external\\nn=71,861", 0.83, 0.58),
        ("MIMIC internal test\\nAUROC 0.830", 0.33, 0.22),
        ("eICU Platt held-out\\nAUROC 0.728\\nBrier 0.099", 0.66, 0.22),
    ]
    for text, x, y in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f2", ec="#5a6f73", lw=1.0),
            transform=ax.transAxes,
        )
    arrows = [((0.19, 0.58), (0.27, 0.58)), ((0.44, 0.58), (0.52, 0.58)), ((0.69, 0.58), (0.77, 0.58)), ((0.58, 0.48), (0.66, 0.31)), ((0.40, 0.48), (0.34, 0.31))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2, color="#2d3f42"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_workflow.png")
    plt.close(fig)


def plot_ablation(v4: dict) -> None:
    rows = []
    for model, result in v4["feature_set_results"].items():
        rows.append(
            {
                "model": model.replace("_", " "),
                "MIMIC internal": result["mimic_internal_test"]["auc"],
                "eICU external": result["eicu_external"]["auc"],
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = range(len(df))
    ax.barh([i + 0.18 for i in y], df["MIMIC internal"], height=0.34, label="MIMIC internal", color="#336b87")
    ax.barh([i - 0.18 for i in y], df["eICU external"], height=0.34, label="eICU external", color="#c96f53")
    ax.set_yticks(list(y), df["model"])
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.5, 0.88)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_feature_ablation.png")
    plt.close(fig)


def plot_roc(v4: dict) -> None:
    pts = pd.DataFrame(v4["plot_data"]["v3_eicu_roc"])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(pts["fpr"], pts["tpr"], color="#336b87", lw=2, label="eICU external AUROC 0.727")
    ax.plot([0, 1], [0, 1], color="#777777", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_eicu_roc.png")
    plt.close(fig)


def plot_calibration(v4: dict) -> None:
    original = pd.DataFrame(v4["plot_data"]["v3_eicu_calibration_original"])
    platt = pd.DataFrame(v4["plot_data"]["v3_eicu_calibration_platt_heldout"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(original["pred_mean"], original["obs_rate"], marker="o", label="Original external")
    ax.plot(platt["pred_mean"], platt["obs_rate"], marker="s", label="Platt held-out")
    ax.plot([0, 0.45], [0, 0.45], color="#777777", lw=1, ls="--")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.45)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_calibration.png")
    plt.close(fig)


def plot_decision_curve(v4: dict) -> None:
    original = pd.DataFrame(v4["plot_data"]["v3_eicu_decision_curve_original"])
    platt = pd.DataFrame(v4["plot_data"]["v3_eicu_decision_curve_platt_heldout"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(original["threshold"], original["net_benefit_model"], label="Original external", color="#336b87")
    ax.plot(platt["threshold"], platt["net_benefit_model"], label="Platt held-out", color="#c96f53")
    ax.plot(original["threshold"], original["net_benefit_treat_all"], label="Treat all", color="#777777", ls="--")
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_decision_curve.png")
    plt.close(fig)


def plot_coefficients(v3: dict) -> None:
    coef = pd.DataFrame(
        [
            {"feature": k, "coef": v}
            for k, v in v3["coefficients"].items()
            if k != "intercept" and not k.endswith("_missing")
        ]
    )
    coef["abs"] = coef["coef"].abs()
    coef = coef.sort_values("abs", ascending=False).head(14).sort_values("coef")
    colors = ["#336b87" if x < 0 else "#c96f53" for x in coef["coef"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.barh(coef["feature"], coef["coef"], color=colors)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_xlabel("Standardized coefficient")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_standardized_coefficients.png")
    plt.close(fig)


def export_figure_source_tables(v4: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(v4["plot_data"]["v3_eicu_roc"]).to_csv(TABLE_DIR / "figure_source_roc_points.csv", index=False)
    pd.DataFrame(v4["plot_data"]["v3_eicu_calibration_original"]).to_csv(TABLE_DIR / "figure_source_calibration_original.csv", index=False)
    pd.DataFrame(v4["plot_data"]["v3_eicu_calibration_platt_heldout"]).to_csv(TABLE_DIR / "figure_source_calibration_platt.csv", index=False)
    pd.DataFrame(v4["plot_data"]["v3_eicu_decision_curve_original"]).to_csv(TABLE_DIR / "figure_source_decision_curve_original.csv", index=False)
    pd.DataFrame(v4["plot_data"]["v3_eicu_decision_curve_platt_heldout"]).to_csv(TABLE_DIR / "figure_source_decision_curve_platt.csv", index=False)


def write_evidence_pack(v3: dict, v4: dict, v5: dict) -> None:
    ci = v5["bootstrap_ci"]
    v6_path = RESULTS / "v6_nutrition_increment" / "v6_nutrition_increment_metrics.json"
    v6_lines = []
    if v6_path.exists():
        v6 = json.loads(v6_path.read_text(encoding="utf-8"))
        deltas = {x["comparison"]: x for x in v6["paired_external_deltas"]}
        baseline = deltas["full_v3 minus non_nutrition_clinical_baseline"]
        no_labs = deltas["full_v3 minus without_nutrition_metabolic_labs"]
        v6_lines = [
            "",
            "## Nutrition Construct Increment",
            "",
            f"- Full V3 minus non-nutrition clinical baseline external AUROC difference: {baseline['point']['auc']:.5f} (95% CI {baseline['ci_95']['auc'][0]:.5f} to {baseline['ci_95']['auc'][1]:.5f}).",
            f"- Full V3 minus model without nutrition/metabolic laboratories external AUROC difference: {no_labs['point']['auc']:.5f} (95% CI {no_labs['ci_95']['auc'][0]:.5f} to {no_labs['ci_95']['auc'][1]:.5f}).",
            "- Interpretation: nutrition-related features support construct framing and care-context interpretation, but they should not be presented as the dominant source of external discrimination.",
        ]
    lines = [
        "# Final Experiment Evidence Pack",
        "",
        "Status date: 2026-08-26",
        "",
        "## Primary Framework",
        "",
        "Name: GERI-NutriRisk-ICU.",
        "",
        "Primary model: V3 plus severity, a standardized logistic framework combining demographics, BMI, early nutrition-support indicators, nutrition/metabolic laboratory markers, vital signs, organ-support features, urine output, and Glasgow Coma Scale.",
        "",
        "## Completed Experimental Surfaces",
        "",
        "- MIMIC-IV development and internal test cohort.",
        "- eICU-CRD external validation cohort.",
        "- Feature-family ablations from V0 through V3, nutrition/metabolic core, and severity-only baseline.",
        "- Domain recalibration using a prespecified eICU calibration split and held-out eICU evaluation.",
        "- eICU subgroup robustness by sex, age group, albumin availability, ventilation status, and enteral nutrition status.",
        "- ROC, calibration, decision-curve, and standardized-coefficient visualizations.",
        "- One thousand bootstrap resamples for primary AUROC, Brier score, and ECE.",
        "",
        "## Primary Metrics",
        "",
        f"- MIMIC-IV internal test: AUROC {ci_text(ci['mimic_internal_test'], 'auc')}; Brier {ci_text(ci['mimic_internal_test'], 'brier')}; ECE {ci_text(ci['mimic_internal_test'], 'ece')}.",
        f"- eICU external full original model: AUROC {ci_text(ci['eicu_external_original_full'], 'auc')}; Brier {ci_text(ci['eicu_external_original_full'], 'brier')}; ECE {ci_text(ci['eicu_external_original_full'], 'ece')}.",
        f"- eICU held-out after Platt recalibration: AUROC {ci_text(ci['eicu_external_platt_heldout'], 'auc')}; Brier {ci_text(ci['eicu_external_platt_heldout'], 'brier')}; ECE {ci_text(ci['eicu_external_platt_heldout'], 'ece')}.",
        "",
        "## Key Interpretation",
        "",
        "The strongest defensible claim is transportable, externally validated risk stratification with substantial calibration improvement after external-domain recalibration. The severity-only baseline is close to the full model externally, so the manuscript should not claim that nutrition variables alone explain performance.",
        *v6_lines,
        "",
        "## Claim Boundary",
        "",
        v5["claim_boundary"],
        "",
        "The study remains retrospective and cannot claim prospective clinical benefit, treatment effect, or deployment readiness.",
    ]
    (RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot()
    v3 = load_json("v3_severity_model/v3_severity_external_validation_metrics.json")
    v4 = load_json("v4_robustness/v4_robustness_metrics.json")
    v5 = load_json("v5_final_ci/v5_final_ci_metrics.json")
    export_tables(v3, v4, v5)
    export_figure_source_tables(v4)
    plot_workflow(v3)
    plot_ablation(v4)
    plot_roc(v4)
    plot_calibration(v4)
    plot_decision_curve(v4)
    plot_coefficients(v3)
    write_evidence_pack(v3, v4, v5)
    print(json.dumps({"figures": len(list(FIG_DIR.glob("fig*.png"))), "tables": len(list(TABLE_DIR.glob("*.csv"))), "evidence_pack": str(RESULTS / "FINAL_EXPERIMENT_EVIDENCE_PACK_v1.md")}, indent=2))


if __name__ == "__main__":
    main()
