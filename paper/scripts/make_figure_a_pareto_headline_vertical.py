"""Create the vertical-layout version of Figure A for narrow paper columns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from make_figure_a_pareto_headline import PROJECT_ROOT, load_headline, load_pareto_data


OUTPUT_PREFIX = PROJECT_ROOT / "paper/figures/final/Fig_A_pareto_headline_vertical"


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
    candidates, pareto_mask, selected = load_pareto_data()
    headline = load_headline()
    reference = headline.loc["reference"]
    pareto = headline.loc["pareto"]

    figure = plt.figure(figsize=(6.85, 6.45))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.22, 0.78], hspace=0.48)
    landscape = figure.add_subplot(grid[0, 0])
    outcome = figure.add_subplot(grid[1, 0])

    landscape.scatter(
        candidates["energy_mj"], candidates["accuracy_percent"],
        s=3.2, color="#94a3b8", alpha=0.16, linewidths=0, rasterized=True, label="All candidates (n = 50,000)",
    )
    frontier = candidates.loc[pareto_mask].sort_values("energy_mj")
    landscape.plot(frontier["energy_mj"], frontier["accuracy_percent"], color="#172033", linewidth=1.8, label="Pareto frontier")
    landscape.scatter(frontier["energy_mj"], frontier["accuracy_percent"], s=9, color="#172033", zorder=3)
    landscape.scatter(
        [selected["energy_mj"]], [selected["accuracy_percent"]],
        s=38, color="#D55E00", linewidths=0, zorder=5, label="Headline model (C002793)",
    )
    landscape.annotate(
        "C002793", xy=(selected["energy_mj"], selected["accuracy_percent"]), xytext=(6, 3),
        textcoords="offset points", fontsize=7.1, color="#A94400", va="bottom",
    )
    landscape.axhline(85, color="#c3473c", linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9)
    landscape.set_xscale("log")
    landscape.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    landscape.xaxis.set_minor_formatter(NullFormatter())
    landscape.set(
        xlabel="Predicted inference energy (mJ, log scale)",
        ylabel="Predicted CIFAR-10 accuracy (%)",
        ylim=(candidates["accuracy_percent"].quantile(0.002) - 0.4, candidates["accuracy_percent"].max() + 1.20),
    )
    landscape.text(landscape.get_xlim()[1] * 0.98, 85.11, "Accuracy = 85%", ha="right", va="bottom", fontsize=7.2, color="#a9362f")
    landscape.spines[["top", "right"]].set_visible(False)
    landscape.tick_params(labelsize=8)
    landscape.legend(
        loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.92,
        fontsize=7.0, handlelength=1.8, labelspacing=0.35,
    )
    landscape.text(-0.09, 1.06, "(a)", transform=landscape.transAxes, fontsize=11, fontweight="bold")

    reference_percent = np.array([100.0, 100.0])
    pareto_percent = np.array(
        [
            float(pareto["mean_energy_mj"]) / float(reference["mean_energy_mj"]) * 100,
            float(pareto["mean_latency_ms"]) / float(reference["mean_latency_ms"]) * 100,
        ]
    )
    y_positions = np.array([0.15, -0.85])
    outcome.barh(y_positions + 0.17, reference_percent, height=0.28, color="#64748B", label="Reference 0551")
    outcome.barh(y_positions - 0.17, pareto_percent, height=0.28, color="#2563EB", label="Pareto 0530")
    outcome.set(
        xlim=(0, 118), ylim=(-1.28, 1.22), yticks=y_positions,
        yticklabels=["Inf.\nEnergy", "Inf.\nLatency"], xlabel="Measured value (% of reference 0551)",
    )
    outcome.set_xticks([0, 25, 50, 75, 100])
    outcome.axvline(100, color="#64748B", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7)
    outcome.grid(axis="x", color="#e4e9f0", linewidth=0.7)
    outcome.set_axisbelow(True)
    outcome.spines[["top", "right", "left"]].set_visible(False)
    outcome.tick_params(labelsize=8)
    # Keep the series key clear of the 100% reference line and the lower bar.
    # The anchor corresponds to x=75% in the 0--118% plotting range.
    outcome.legend(
        loc="upper left", bbox_to_anchor=(75 / 118 + 0.003, 0.91),
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
        fontsize=6.7, handlelength=1.25, borderaxespad=0.0, labelspacing=0.35,
    )

    energy_reduction = -float(pareto["energy_change_vs_reference_percent"])
    latency_reduction = -float(pareto["latency_change_vs_reference_percent"])
    parameter_change = (float(pareto["parameter_count"]) / float(reference["parameter_count"]) - 1) * 100
    accuracy_difference = float(pareto["actual_accuracy_percent"]) - float(reference["actual_accuracy_percent"])
    outcome.set_title("(b) Trained headline comparison", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    outcome.text(
        0.0, 0.97,
        f"Accuracy: {float(reference['actual_accuracy_percent']):.2f}% → {float(pareto['actual_accuracy_percent']):.2f}% ({accuracy_difference:+.2f} pp)\n"
        f"Parameters: {parameter_change:+.2f}%  |  n = {int(reference['valid_trials'])} trials/model",
        transform=outcome.transAxes, ha="left", va="top", fontsize=7.0, color="#334155", linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.4},
    )
    outcome.text(50, 0.46, f"{float(reference['mean_energy_mj']):.2f} → {float(pareto['mean_energy_mj']):.2f} mJ  ({energy_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=7.2, color="#172033")
    outcome.text(50, -0.54, f"{float(reference['mean_latency_ms']):.3f} → {float(pareto['mean_latency_ms']):.3f} ms  ({latency_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=7.2, color="#172033")

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.16)
    figure.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
