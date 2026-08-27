#!/usr/bin/env python3
"""Restyle locked manuscript figures with varied Nature-like chart archetypes.

This script does not recompute any model result. It reads locked aggregate
tables/JSON outputs and redraws the manuscript JPG assets with a more diverse,
publication-oriented visual language.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05-results"
WRITING = ROOT / "06-writing"
FIG_DIR = WRITING / "figures"
TABLE_DIR = WRITING / "tables"

COL = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "blue": "#3B6EA8",
    "cyan": "#2F8C95",
    "teal": "#1B9E77",
    "orange": "#D95F02",
    "vermillion": "#C44E52",
    "purple": "#7B61A3",
    "green": "#4C956C",
    "gold": "#C9A227",
    "gray": "#9CA3AF",
    "pale_blue": "#EAF2F8",
    "pale_teal": "#E8F4F2",
    "pale_gold": "#FBF3D5",
    "pale_gray": "#F5F6F7",
}

METHOD_COLORS = {
    "Logistic regression": COL["blue"],
    "XGBoost": COL["orange"],
    "GERI-NutriRisk-ICU": COL["teal"],
    "CatBoost": COL["purple"],
    "HistGradientBoosting": COL["gold"],
    "Random forest": COL["gray"],
    "ExtraTrees": "#A46A6A",
    "MLP": "#5E6C84",
    "TabNet": COL["vermillion"],
}

GROUP_COLORS = {
    "Organ support and neurologic severity": COL["vermillion"],
    "Vital signs": COL["orange"],
    "Blood gas and perfusion": COL["purple"],
    "Nutrition/metabolic labs": COL["teal"],
    "Explicit nutrition": COL["green"],
    "Additional chemistry": COL["gold"],
    "Demographics": COL["blue"],
    "Missingness indicators": COL["gray"],
    "Coagulation and hematology": "#8B5E3C",
    "ICU and admission context": "#64748B",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COL["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: mpl.figure.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)


def clear_existing_figures() -> None:
    for pattern in ("fig*.jpg", "fig*.png", "main_fig*.jpg", "main_fig*.png", "supp_fig*.jpg", "supp_fig*.png"):
        for path in FIG_DIR.glob(pattern):
            path.unlink()


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def read_json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def nice_label(text: str) -> str:
    mapping = {
        "V0_demographics_bmi_nutrition_support": "V0 demo/BMI/nutrition support",
        "V1_plus_labs": "V1 + labs",
        "V2_plus_vitals": "V2 + vitals",
        "V3_plus_severity": "V3 + severity",
        "nutrition_metabolic_core": "Nutrition/metabolic core",
        "severity_only_baseline": "Severity-only baseline",
        "full_minus_without_v12_nutrition_construct": "Full vs no nutrition construct",
        "full_minus_without_explicit_nutrition": "Full vs no explicit nutrition",
        "full_minus_without_nutrition_support": "Full vs no nutrition support",
        "full_minus_without_nutrition_metabolic_labs": "Full vs no nutrition/metabolic labs",
        "all_heldout_hospitals": "All held-out hospitals",
        "excluding_top10_hospitals_by_n": "Excluding 10 largest hospitals",
        "hospitals_with_n_ge_100": "Hospitals with n >= 100",
        "hospitals_with_n_ge_200": "Hospitals with n >= 200",
    }
    return mapping.get(text, text.replace("_", " "))


def parse_estimate_ci(text: str) -> tuple[float, float, float]:
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(text))]
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    if nums:
        return nums[0], np.nan, np.nan
    return np.nan, np.nan, np.nan


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COL["ink"],
    )


def add_arrow(ax, xy1, xy2, color=COL["ink"], rad=0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.1,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_box(ax, xy, wh, text, fc, ec, size=8.0) -> None:
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=size,
        color=COL["ink"],
        linespacing=1.2,
    )


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.axis("off")
    ax.text(0.02, 0.95, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text(0.07, 0.95, "Study design and locked evaluation path", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=COL["ink"])

    draw_box(ax, (0.05, 0.66), (0.20, 0.17), "MIMIC-IV\nolder ICU stays\nn=33,671", COL["pale_blue"], COL["blue"])
    draw_box(ax, (0.32, 0.66), (0.20, 0.17), "Harmonized\n83-feature schema\nfirst 24 h", COL["pale_teal"], COL["teal"])
    draw_box(ax, (0.59, 0.66), (0.20, 0.17), "Locked head\nLightGBM + logistic\nanchor", "#F2F0FA", COL["purple"])
    draw_box(ax, (0.74, 0.38), (0.20, 0.17), "eICU held-out\nn=57,488\n7,130 events", COL["pale_gold"], COL["gold"])

    add_arrow(ax, (0.25, 0.745), (0.32, 0.745), COL["blue"])
    add_arrow(ax, (0.52, 0.745), (0.59, 0.745), COL["teal"])
    add_arrow(ax, (0.70, 0.66), (0.78, 0.55), COL["purple"], rad=-0.15)

    ax.plot([0.10, 0.90], [0.30, 0.30], color=COL["grid"], lw=2, transform=ax.transAxes, clip_on=False)
    milestones = [
        (0.16, "Source\ntraining"),
        (0.36, "Local domain\nupdate"),
        (0.56, "Disjoint Platt\ncalibration"),
        (0.76, "APACHE same-subset\ncomparison"),
        (0.91, "Hospital-cluster\nrobustness"),
    ]
    for x, label in milestones:
        ax.scatter([x], [0.30], s=75, color=COL["teal"], edgecolor="white", lw=1, transform=ax.transAxes, zorder=3)
        ax.text(x, 0.19, label, transform=ax.transAxes, ha="center", va="top", fontsize=7.3, linespacing=1.12)

    ax.text(
        0.05,
        0.07,
        "Claim boundary: retrospective mortality-risk stratification and calibration; not malnutrition diagnosis or treatment-effect evidence.",
        transform=ax.transAxes,
        fontsize=7.8,
        color=COL["muted"],
    )
    savefig(fig, "main_fig1_workflow.jpg")


def plot_ablation() -> None:
    df = read_csv("table2_feature_family_ablation.csv").copy()
    df["label"] = df["model"].map(nice_label)
    df = df.sort_values("mimic_internal_auc")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, row in df.iterrows():
        yi = list(df.index).index(i)
        ax.plot([row["eicu_external_auc"], row["mimic_internal_auc"]], [yi, yi], color="#CBD5E1", lw=2.2, zorder=1)
    ax.scatter(df["eicu_external_auc"], y, s=42, color=COL["orange"], label="eICU external", zorder=3)
    ax.scatter(df["mimic_internal_auc"], y, s=42, color=COL["blue"], label="MIMIC internal", zorder=3)
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.text(max(row["eicu_external_auc"], row["mimic_internal_auc"]) + 0.006, yi, f"{row['eicu_external_auc']:.3f}", va="center", fontsize=7.2, color=COL["muted"])
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.54, 0.86)
    ax.set_title("Feature-family trajectory: source-domain gain and transport gap")
    ax.legend(loc="lower right", ncol=2)
    savefig(fig, "main_fig2_feature_ablation.jpg")


def plot_roc() -> None:
    pts = read_csv("figure_source_roc_points.csv")
    fig, ax = plt.subplots(figsize=(4.7, 4.3))
    ax.fill_between(pts["fpr"], pts["tpr"], pts["fpr"], color=COL["pale_blue"], alpha=0.8, linewidth=0)
    ax.plot(pts["fpr"], pts["tpr"], color=COL["blue"], lw=2.0)
    ax.plot([0, 1], [0, 1], color=COL["gray"], lw=0.9, ls=(0, (3, 3)))
    ax.text(0.58, 0.15, "AUROC 0.727", fontsize=8.5, color=COL["blue"], transform=ax.transAxes)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, "supp_figS1_eicu_roc.jpg")


def plot_calibration() -> None:
    raw = read_csv("figure_source_calibration_original.csv")
    platt = read_csv("figure_source_calibration_platt.csv")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 0.46], [0, 0.46], color=COL["gray"], lw=0.9, ls=(0, (3, 3)))
    ax.plot(raw["pred_mean"], raw["obs_rate"], "-o", color=COL["vermillion"], ms=4.0, lw=1.7, label="Original")
    ax.plot(platt["pred_mean"], platt["obs_rate"], "-s", color=COL["teal"], ms=4.0, lw=1.7, label="Platt held-out")
    ax.scatter(platt["pred_mean"], np.zeros(len(platt)) - 0.008, s=np.sqrt(platt["n"]) * 2.2, color=COL["teal"], alpha=0.15, clip_on=False)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 0.46)
    ax.set_ylim(-0.012, 0.46)
    ax.set_title("External-domain calibration")
    ax.legend(loc="upper left")
    savefig(fig, "main_fig3_calibration.jpg")


def plot_decision_curve() -> None:
    raw = read_csv("figure_source_decision_curve_original.csv")
    platt = read_csv("figure_source_decision_curve_platt.csv")
    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    ax.axhline(0, color=COL["ink"], lw=0.8)
    ax.plot(raw["threshold"], raw["net_benefit_model"], color=COL["blue"], lw=1.8, label="Original")
    ax.plot(platt["threshold"], platt["net_benefit_model"], color=COL["teal"], lw=2.0, label="Platt held-out")
    ax.plot(raw["threshold"], raw["net_benefit_treat_all"], color=COL["gray"], lw=1.2, ls=(0, (3, 3)), label="Treat all")
    ax.fill_between(platt["threshold"], 0, platt["net_benefit_model"], where=platt["net_benefit_model"] > 0, color=COL["pale_teal"], alpha=0.7)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve behavior after recalibration")
    ax.legend(loc="upper right")
    savefig(fig, "supp_figS2_decision_curve.jpg")


def plot_coefficients() -> None:
    df = read_csv("table5_standardized_coefficients.csv")
    df = df[~df["feature"].str.endswith("_missing")].copy()
    df["abs"] = df["standardized_coefficient"].abs()
    df = df.sort_values("abs", ascending=False).head(16).sort_values("standardized_coefficient")
    y = np.arange(len(df))
    colors = [COL["blue"] if x < 0 else COL["vermillion"] for x in df["standardized_coefficient"]]
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.axvline(0, color=COL["ink"], lw=0.8)
    for yi, x, c in zip(y, df["standardized_coefficient"], colors):
        ax.plot([0, x], [yi, yi], color=c, lw=1.6, alpha=0.75)
    ax.scatter(df["standardized_coefficient"], y, s=34, color=colors, zorder=3)
    ax.set_yticks(y, df["feature"])
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Transparent logistic-anchor coefficients")
    savefig(fig, "supp_figS3_standardized_coefficients.jpg")


def plot_split_stability() -> None:
    df = read_csv("table8_split_stability_v7").copy() if False else read_csv("table8_split_stability_v7.csv").copy()
    metrics = []
    for _, row in df.iterrows():
        for col, metric in [("auc_mean_sd", "AUROC"), ("brier_mean_sd", "Brier"), ("ece_mean_sd", "ECE")]:
            est, _, _ = parse_estimate_ci(row[col])
            metrics.append({"evaluation": row["evaluation"], "metric": metric, "estimate": est})
    mat = pd.DataFrame(metrics).pivot(index="evaluation", columns="metric", values="estimate")
    mat = mat[["AUROC", "Brier", "ECE"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.42})
    im = axes[0].imshow(mat.values, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(np.arange(mat.shape[1]), mat.columns)
    axes[0].set_yticks(np.arange(mat.shape[0]), [x.replace(" recalibrated", "\nrecalibrated") for x in mat.index])
    axes[0].set_title("Five-split stability")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text_color = "white" if mat.values[i, j] >= 0.45 else COL["ink"]
            axes[0].text(j, i, f"{mat.values[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=text_color)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=7.0)

    bm = read_csv("table7_model_benchmark_v7.csv")
    axes[1].plot(bm["feature_count"], bm["mimic_auc"], "-o", color=COL["blue"], label="MIMIC")
    axes[1].plot(bm["feature_count"], bm["eicu_auc"], "-o", color=COL["orange"], label="eICU")
    axes[1].set_xlabel("Feature count")
    axes[1].set_ylabel("AUROC", labelpad=9)
    axes[1].set_title("Feature-count benchmark")
    axes[1].legend(loc="lower right")
    savefig(fig, "supp_figS4_benchmark_stability.jpg")


def plot_sota_initial() -> None:
    df = read_csv("table11_sota_tabular_benchmark_v8.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for _, row in df.iterrows():
        color = METHOD_COLORS.get(row["model"], COL["gray"])
        ax.plot([row["mimic_auc"], row["eicu_auc_original"]], [row["model"], row["model"]], color="#D1D5DB", lw=1.8, zorder=1)
        ax.scatter(row["mimic_auc"], row["model"], s=48, marker="o", color=color, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(row["eicu_auc_original"], row["model"], s=48, marker="D", color=color, edgecolor="white", lw=0.8, zorder=3)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.70, 0.90)
    ax.set_title("Initial tabular benchmark: internal-to-external transport")
    ax.scatter([], [], marker="o", color=COL["ink"], label="MIMIC internal")
    ax.scatter([], [], marker="D", color=COL["ink"], label="eICU external")
    ax.legend(loc="lower right")
    savefig(fig, "supp_figS5_sota_tabular_benchmark.jpg")


def plot_lightgbm_package() -> None:
    v9 = read_json("v9_lightgbm_primary/v9_lightgbm_primary_metrics.json")
    roc = pd.DataFrame(v9["plot_data"]["eicu_roc_original"])
    cal_raw = pd.DataFrame(v9["plot_data"]["eicu_calibration_original"])
    cal_platt = pd.DataFrame(v9["plot_data"]["eicu_calibration_platt_heldout"])
    dca_platt = pd.DataFrame(v9["plot_data"]["eicu_decision_curve_platt_heldout"])
    imp = pd.DataFrame(v9["feature_importance"]).head(12).sort_values("mean_abs_shap_contribution")

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.4))
    ax = axes[0, 0]
    panel_label(ax, "A")
    ax.fill_between(roc["fpr"], roc["tpr"], roc["fpr"], color=COL["pale_blue"])
    ax.plot(roc["fpr"], roc["tpr"], color=COL["blue"], lw=1.8)
    ax.plot([0, 1], [0, 1], color=COL["gray"], lw=0.8, ls=(0, (3, 3)))
    ax.set_title("External ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.plot([0, 0.55], [0, 0.55], color=COL["gray"], lw=0.8, ls=(0, (3, 3)))
    ax.plot(cal_raw["pred_mean"], cal_raw["obs_rate"], "o-", color=COL["vermillion"], ms=3.8, label="Original")
    ax.plot(cal_platt["pred_mean"], cal_platt["obs_rate"], "s-", color=COL["teal"], ms=3.8, label="Platt")
    ax.set_title("Calibration")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.axhline(0, color=COL["ink"], lw=0.7)
    ax.plot(dca_platt["threshold"], dca_platt["net_benefit_model"], color=COL["teal"], lw=2.0)
    ax.fill_between(dca_platt["threshold"], 0, dca_platt["net_benefit_model"], color=COL["pale_teal"])
    ax.set_title("Decision curve")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")

    ax = axes[1, 1]
    panel_label(ax, "D")
    y = np.arange(len(imp))
    ax.hlines(y, 0, imp["mean_abs_shap_contribution"], color=COL["grid"], lw=2.0)
    ax.scatter(imp["mean_abs_shap_contribution"], y, s=36, color=COL["purple"])
    ax.set_yticks(y, imp["feature"])
    ax.set_xlabel("Mean absolute contribution")
    ax.set_title("Feature contribution")
    savefig(fig, "supp_figS6_lightgbm_primary_package.jpg")


def plot_model_lock() -> None:
    df = read_csv("table16_sota_completion_lock_v10.csv")
    tab = read_csv("table17_tabnet_neural_sota_v11.csv")
    df = pd.concat([df, tab[tab["model"] == "TabNet"]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    label_offsets = {
        "GERI-NutriRisk-ICU": (-58, -18),
        "HistGradientBoosting": (10, 19),
        "XGBoost": (12, -8),
        "Random forest": (12, 10),
        "CatBoost": (12, -10),
        "Logistic regression": (12, -16),
        "MLP": (12, 7),
        "ExtraTrees": (12, -9),
        "TabNet": (-34, 10),
    }
    for _, row in df.iterrows():
        color = METHOD_COLORS.get(row["model"], COL["gray"])
        size = 230 if row["model"] == "GERI-NutriRisk-ICU" else 90
        marker = "*" if row["model"] == "GERI-NutriRisk-ICU" else "o"
        ax.scatter(row["eicu_brier_platt_heldout"], row["eicu_auc_platt_heldout"], s=size, color=color, marker=marker, edgecolor="white", lw=0.9, zorder=3)
        dx, dy = label_offsets.get(row["model"], (10, 0))
        ax.annotate(
            row["model"],
            xy=(row["eicu_brier_platt_heldout"], row["eicu_auc_platt_heldout"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.86),
            arrowprops=dict(arrowstyle="-", lw=0.45, color="#CBD5E1", shrinkA=0, shrinkB=3) if row["model"] in {"GERI-NutriRisk-ICU", "HistGradientBoosting"} else None,
            zorder=4,
        )
    ax.axhline(0.794, color=COL["grid"], lw=0.9, zorder=0)
    ax.axvline(0.088, color=COL["grid"], lw=0.9, zorder=0)
    ax.annotate(
        "Locked performance head",
        xy=(0.088, 0.794),
        xytext=(0.099, 0.812),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COL["ink"]),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=COL["grid"], alpha=0.95),
    )
    ax.set_xlabel("Held-out eICU Brier score after Platt calibration (lower is better)")
    ax.set_ylabel("Held-out eICU AUROC (higher is better)")
    ax.set_xlim(0.085, 0.118)
    ax.set_ylim(0.615, 0.826)
    ax.set_title("Bounded comparator queue: discrimination-calibration tradeoff")
    savefig(fig, "main_fig4_sota_completion_model_lock.jpg")


def plot_gain_importance() -> None:
    df = read_csv("table20_v15_feature_importance_gain.csv").head(20).copy()
    df = df.sort_values("gain_fraction")
    y = np.arange(len(df))
    colors = [GROUP_COLORS.get(g, COL["gray"]) for g in df["group"]]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.hlines(y, 0, df["gain_fraction"], color="#D1D5DB", lw=1.5)
    ax.scatter(df["gain_fraction"], y, s=36, color=colors, edgecolor="white", lw=0.6, zorder=3)
    ax.set_yticks(y, df["feature"])
    ax.set_xlabel("Fraction of total LightGBM gain")
    ax.set_title("Top gain-ranked features")
    handles = []
    for group in df["group"].drop_duplicates():
        handles.append(plt.Line2D([], [], marker="o", lw=0, color=GROUP_COLORS.get(group, COL["gray"]), label=group, markersize=5))
    ax.legend(handles=handles[:6], loc="lower right", fontsize=6.8)
    savefig(fig, "supp_figS7_v14_top20_gain_importance.jpg")


def plot_group_importance() -> None:
    df = read_csv("table21_v15_group_permutation_importance.csv").copy()
    df["display"] = df["group"].str.replace(" and ", " & ", regex=False).map(lambda x: textwrap.fill(x, width=26))
    df = df.sort_values("auc_drop")
    y = np.arange(len(df))
    colors = [GROUP_COLORS.get(g, COL["gray"]) for g in df["group"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.hlines(y, 0, df["auc_drop"], color="#D1D5DB", lw=1.8, zorder=1)
    sizes = 70 + df["n_features"] * 16
    ax.scatter(df["auc_drop"], y, s=sizes, color=colors, alpha=0.90, edgecolor="white", lw=0.8, zorder=3)
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.annotate(
            f"+{row['brier_increase']:.4f} Brier",
            xy=(row["auc_drop"], yi),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            color=COL["muted"],
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.82),
        )
    ax.set_yticks(y, df["display"])
    ax.set_xlabel("AUROC decrease after group permutation")
    ax.set_xlim(0, max(df["auc_drop"]) + 0.026)
    ax.set_title("Feature-domain permutation importance")
    ax.text(
        0.99,
        0.03,
        "Point size encodes feature count; text gives Brier increase.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=COL["muted"],
    )
    fig.subplots_adjust(left=0.34, right=0.98)
    savefig(fig, "main_fig5_group_permutation_importance.jpg")


def plot_hospital_distribution() -> None:
    h = read_csv("table23_v16_hospital_metrics.csv")
    h = h[(h["n"] >= 100) & h["auc"].notna()].copy()
    s = read_csv("table22_v16_hospital_sensitivity.csv")
    rng = np.random.default_rng(20260826)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    parts = ax.violinplot([h["auc"].to_numpy()], positions=[0], widths=0.62, showmeans=False, showextrema=False, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(COL["pale_blue"])
        pc.set_edgecolor(COL["blue"])
        pc.set_alpha(0.9)
    x = rng.normal(0, 0.055, len(h))
    ax.scatter(x, h["auc"], s=np.clip(np.sqrt(h["n"]) * 1.8, 8, 42), color=COL["blue"], alpha=0.45, edgecolor="white", lw=0.25)
    median = h["auc"].median()
    ax.plot([-0.22, 0.22], [median, median], color=COL["ink"], lw=1.3)
    all_auc = float(s.loc[s["label"] == "all_heldout_hospitals", "auc"].iloc[0])
    ax.axhline(all_auc, color=COL["teal"], lw=1.4, ls=(0, (4, 2)), zorder=1)
    ax.text(
        0.98,
        0.96,
        f"Overall AUROC {all_auc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        color=COL["teal"],
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=COL["teal"], alpha=1.0),
        zorder=5,
    )
    ax.set_xlim(-0.42, 0.68)
    ax.set_xticks([0], ["eICU hospitals\nn >= 100"])
    ax.set_ylabel("Hospital-level AUROC")
    ax.set_title("Hospital-level performance distribution")
    savefig(fig, "main_fig6_hospital_auc_distribution.jpg")


def plot_hospital_scatter() -> None:
    h = read_csv("table23_v16_hospital_metrics.csv")
    h = h[(h["n"] >= 200) & h["auc"].notna()].copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    sc = ax.scatter(h["n"], h["auc"], s=np.clip(h["events"] * 0.7, 12, 95), c=h["event_rate"], cmap="viridis", alpha=0.78, edgecolor="white", lw=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("Held-out stays per hospital (log scale)")
    ax.set_ylabel("Hospital-level AUROC")
    ax.set_title("Hospital size, event rate, and AUROC")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Event rate")
    savefig(fig, "supp_figS8_hospital_size_auc_scatter.jpg")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clear_existing_figures()
    setup()
    plot_workflow()
    plot_ablation()
    plot_roc()
    plot_calibration()
    plot_decision_curve()
    plot_coefficients()
    plot_split_stability()
    plot_sota_initial()
    plot_lightgbm_package()
    plot_model_lock()
    plot_gain_importance()
    plot_group_importance()
    plot_hospital_distribution()
    plot_hospital_scatter()
    print(json.dumps({"jpg_figures": len(list(FIG_DIR.glob("*.jpg"))), "figure_dir": str(FIG_DIR)}, indent=2))


if __name__ == "__main__":
    main()
