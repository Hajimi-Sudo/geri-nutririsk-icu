#!/usr/bin/env python3
"""Export manuscript-ready aggregate CSV tables from V4 robustness metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


INP = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v4_robustness/v4_robustness_metrics.json")
OUT = Path("/root/autodl-tmp/frontiers_nutrition_mimic_eicu/v4_robustness/tables")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(INP.read_text())

    model_rows = []
    for name, res in d["feature_set_results"].items():
        model_rows.append({
            "model": name,
            "mimic_internal_test_auc": res.get("mimic_internal_test", {}).get("auc", ""),
            "mimic_internal_test_brier": res.get("mimic_internal_test", {}).get("brier", ""),
            "eicu_external_auc": res["eicu_external"]["auc"],
            "eicu_external_brier": res["eicu_external"]["brier"],
            "eicu_external_n": res["eicu_external"]["n"],
            "eicu_external_events": res["eicu_external"]["events"],
        })
    write_csv(OUT / "table_model_ablation.csv", model_rows)

    subgroup_rows = []
    for family, groups in d["v3_eicu_subgroups_original"].items():
        for group, metrics in groups.items():
            subgroup_rows.append({
                "subgroup_family": family,
                "subgroup": group,
                "n": metrics["n"],
                "events": metrics["events"],
                "event_rate": metrics["event_rate"],
                "auc": metrics["auc"],
                "brier": metrics["brier"],
            })
    write_csv(OUT / "table_eicu_subgroups_original.csv", subgroup_rows)

    calib_rows = []
    for label, bins in [
        ("original_full_eicu", d["plot_data"]["v3_eicu_calibration_original"]),
        ("platt_heldout_eicu", d["plot_data"]["v3_eicu_calibration_platt_heldout"]),
    ]:
        for row in bins:
            calib_rows.append({"curve": label, **row})
    write_csv(OUT / "figure_calibration_bins.csv", calib_rows)

    roc_rows = [{"curve": "v3_original_full_eicu", **row} for row in d["plot_data"]["v3_eicu_roc"]]
    write_csv(OUT / "figure_roc_points.csv", roc_rows)

    dca_rows = []
    for label, rows_ in [
        ("original_full_eicu", d["plot_data"]["v3_eicu_decision_curve_original"]),
        ("platt_heldout_eicu", d["plot_data"]["v3_eicu_decision_curve_platt_heldout"]),
    ]:
        for row in rows_:
            dca_rows.append({"curve": label, **row})
    write_csv(OUT / "figure_decision_curve.csv", dca_rows)

    summary = {
        "tables_dir": str(OUT),
        "files": sorted(p.name for p in OUT.glob("*.csv")),
        "main_message": "V3 improves external discrimination over V0-V2; domain recalibration repairs Brier score while preserving AUROC.",
    }
    (OUT / "table_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
