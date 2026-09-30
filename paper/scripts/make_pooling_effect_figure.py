"""Create publication-ready Fig. B for adjusted pooling effects on CNN performance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_pooling_accuracy_effect import build_design as build_accuracy_design
from analyze_pooling_accuracy_effect import partial_residuals as accuracy_partial_residuals
from analyze_pooling_energy_effect import add_design_columns as build_energy_design
from analyze_pooling_energy_effect import partial_residuals as energy_partial_residuals


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "measurements/ml/model_dataset.csv"
ENERGY_SUMMARY = PROJECT_ROOT / "paper/results/pooling_effects/energy/pooling_energy_summary.json"
ACCURACY_SUMMARY = PROJECT_ROOT / "paper/results/pooling_effects/accuracy/pooling_accuracy_summary.json"
OUTPUT_PREFIX = PROJECT_ROOT / "paper/figures/final/Fig_B_pooling_effect"
VERTICAL_OUTPUT_PREFIX = PROJECT_ROOT / "paper/figures/final/Fig_B_pooling_effect_vertical"


def load_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)["primary_pool_count"]


def plot_partial_effect(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    ylabel: str,
    annotation: str,
    panel_label: str,
) -> None:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    order = np.argsort(x)
    axis.scatter(x, y, s=15, alpha=0.52, color=color, edgecolors="none", rasterized=True)
    axis.plot(x[order], intercept + slope * x[order], color="#111827", linewidth=1.8)
    axis.axhline(0, color="#9ca3af", linewidth=0.7, zorder=0)
    axis.axvline(0, color="#9ca3af", linewidth=0.7, zorder=0)
    axis.set(
        xlabel="Pooling count residual after adjustment",
        ylabel=ylabel,
    )
    axis.text(0.01, 1.03, panel_label, transform=axis.transAxes, fontsize=11, fontweight="bold")
    axis.text(
        0.98, 0.97, annotation, transform=axis.transAxes, ha="right", va="top",
        fontsize=7.8, linespacing=1.25,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    frame = pd.read_csv(DATASET, encoding="utf-8-sig")

    energy_frame = frame.dropna(
        subset=[
            "target_energy_j", "feature_pool_count", "feature_depth", "feature_parameter_count",
            "feature_channel_mean", "feature_channel_std", "feature_channel_last_first_ratio",
            "feature_pattern", "feature_growth_pattern",
        ]
    ).copy()
    energy_frame = energy_frame[energy_frame["target_energy_j"] > 0]
    energy_x, energy_names, energy_y = build_energy_design(energy_frame, "feature_pool_count")
    energy_partial_x, energy_partial_y = energy_partial_residuals(
        energy_x, energy_y, energy_names.index("feature_pool_count")
    )

    accuracy_frame = frame.dropna(
        subset=[
            "target_accuracy_percent", "feature_pool_count", "feature_depth", "feature_parameter_count",
            "feature_channel_mean", "feature_channel_std", "feature_channel_last_first_ratio",
            "feature_pattern", "feature_growth_pattern",
        ]
    ).copy()
    accuracy_x, accuracy_names, accuracy_y = build_accuracy_design(accuracy_frame, "feature_pool_count")
    accuracy_partial_x, accuracy_partial_y = accuracy_partial_residuals(
        accuracy_x, accuracy_y, accuracy_names.index("feature_pool_count")
    )

    energy = load_summary(ENERGY_SUMMARY)
    accuracy = load_summary(ACCURACY_SUMMARY)
    energy_annotation = (
        f"Energy: {energy['percent_change_per_unit']:.1f}% per pool\n"
        f"95% CI [{energy['ci95_lower_percent']:.1f}, {energy['ci95_upper_percent']:.1f}]%; HC3 p < 0.001\n"
        f"n = {int(energy['n_observations'])}"
    )
    accuracy_annotation = (
        f"Accuracy: +{accuracy['coefficient_percentage_points_per_unit']:.2f} pp per pool\n"
        f"95% CI [{accuracy['ci95_lower_percentage_points']:.2f}, {accuracy['ci95_upper_percentage_points']:.2f}] pp; HC3 p < 0.001\n"
        f"n = {int(accuracy['n_observations'])}"
    )

    # 6.85 inches (17.4 cm) fits inside the template's 18 cm two-column text width.
    figure, axes = plt.subplots(1, 2, figsize=(6.85, 2.92), constrained_layout=True)
    plot_partial_effect(
        axes[0], energy_partial_x, energy_partial_y, "#0072B2",
        "Adjusted log inference energy residual (ln J)", energy_annotation, "(a)",
    )
    plot_partial_effect(
        axes[1], accuracy_partial_x, accuracy_partial_y, "#D55E00",
        "Adjusted CIFAR-10 accuracy residual (pp)", accuracy_annotation, "(b)",
    )
    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    vertical, axes = plt.subplots(2, 1, figsize=(6.85, 5.35), constrained_layout=True)
    plot_partial_effect(
        axes[0], energy_partial_x, energy_partial_y, "#0072B2",
        "Adjusted log inference energy residual (ln J)", energy_annotation, "(a)",
    )
    plot_partial_effect(
        axes[1], accuracy_partial_x, accuracy_partial_y, "#D55E00",
        "Adjusted CIFAR-10 accuracy residual (pp)", accuracy_annotation, "(b)",
    )
    vertical.savefig(VERTICAL_OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.14)
    vertical.savefig(VERTICAL_OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.14)
    plt.close(vertical)
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.pdf')}")
    print(f"Wrote {VERTICAL_OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"Wrote {VERTICAL_OUTPUT_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
