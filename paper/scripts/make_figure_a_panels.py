"""Create independently placeable panels for publication Figure A."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from make_figure_a_pareto_headline import PROJECT_ROOT, load_headline, load_pareto_data


OUTPUT_DIR = PROJECT_ROOT / "paper/figures/final"


def configure_style() -> None:
    """Apply the typography shared by the combined Figure A."""
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


def save(figure: plt.Figure, filename: str) -> None:
    """Save one panel in raster and vector formats."""
    prefix = OUTPUT_DIR / filename
    figure.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.16)
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {prefix.with_suffix('.pdf')}")


def make_landscape_panel() -> None:
    """Create panel (a): final ensemble Energy--Accuracy Pareto landscape."""
    candidates, pareto_mask, selected = load_pareto_data()
    figure, axis = plt.subplots(figsize=(6.85, 4.15))
    figure.subplots_adjust(left=0.13, right=0.985, top=0.91, bottom=0.16)

    axis.scatter(
        candidates["energy_mj"], candidates["accuracy_percent"],
        s=3.2, color="#94a3b8", alpha=0.16, linewidths=0,
        rasterized=True, label="All candidates (n = 50,000)",
    )
    frontier = candidates.loc[pareto_mask].sort_values("energy_mj")
    axis.plot(frontier["energy_mj"], frontier["accuracy_percent"], color="#172033", linewidth=1.8, label="Pareto frontier")
    axis.scatter(frontier["energy_mj"], frontier["accuracy_percent"], s=9, color="#172033", zorder=3)
    axis.scatter(
        [selected["energy_mj"]], [selected["accuracy_percent"]],
        s=38, color="#D55E00", linewidths=0, zorder=5, label="Headline model (C002793)",
    )
    axis.annotate(
        "C002793", xy=(selected["energy_mj"], selected["accuracy_percent"]), xytext=(6, -13),
        textcoords="offset points", fontsize=7.1, color="#A94400",
    )
    axis.axhline(85, color="#c3473c", linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9)
    axis.text(axis.get_xlim()[1] * 0.98, 85.11, "Accuracy = 85%", ha="right", va="bottom", fontsize=7.2, color="#a9362f")
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set(
        xlabel="Predicted inference energy (mJ, log scale)",
        ylabel="Predicted CIFAR-10 accuracy (%)",
        ylim=(candidates["accuracy_percent"].quantile(0.002) - 0.4, candidates["accuracy_percent"].max() + 1.20),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.legend(loc="lower right", frameon=False, fontsize=7.0, handlelength=1.8, labelspacing=0.35)
    axis.text(-0.09, 1.04, "(a)", transform=axis.transAxes, fontsize=11, fontweight="bold")
    save(figure, "Fig_Aa_pareto_landscape")


def make_headline_panel() -> None:
    """Create panel (b): trained reference-versus-headline comparison."""
    headline = load_headline()
    reference = headline.loc["reference"]
    pareto = headline.loc["pareto"]
    figure, axis = plt.subplots(figsize=(6.85, 3.00))
    figure.subplots_adjust(left=0.13, right=0.985, top=0.85, bottom=0.20)

    reference_percent = np.array([100.0, 100.0])
    pareto_percent = np.array(
        [
            float(pareto["mean_energy_mj"]) / float(reference["mean_energy_mj"]) * 100,
            float(pareto["mean_latency_ms"]) / float(reference["mean_latency_ms"]) * 100,
        ]
    )
    y_positions = np.array([0.15, -0.85])
    axis.barh(y_positions + 0.17, reference_percent, height=0.28, color="#64748B", label="Reference 0551")
    axis.barh(y_positions - 0.17, pareto_percent, height=0.28, color="#2563EB", label="Pareto 0530")
    axis.set(
        xlim=(0, 118), ylim=(-1.28, 1.22), yticks=y_positions,
        yticklabels=["Inf.\nEnergy", "Inf.\nLatency"],
        xlabel="Measured value (% of reference 0551)",
    )
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.axvline(100, color="#64748B", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7)
    axis.grid(axis="x", color="#e4e9f0", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.legend(
        loc="upper left", bbox_to_anchor=(75 / 118 + 0.003, 0.91),
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
        fontsize=6.7, handlelength=1.25, borderaxespad=0.0, labelspacing=0.35,
    )

    energy_reduction = -float(pareto["energy_change_vs_reference_percent"])
    latency_reduction = -float(pareto["latency_change_vs_reference_percent"])
    parameter_change = (float(pareto["parameter_count"]) / float(reference["parameter_count"]) - 1) * 100
    accuracy_difference = float(pareto["actual_accuracy_percent"]) - float(reference["actual_accuracy_percent"])
    arrow = "\N{RIGHTWARDS ARROW}"
    axis.set_title("(b) Trained headline comparison", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    axis.text(
        0.0, 0.97,
        f"Accuracy: {float(reference['actual_accuracy_percent']):.2f}% {arrow} {float(pareto['actual_accuracy_percent']):.2f}% ({accuracy_difference:+.2f} pp)\n"
        f"Parameters: {parameter_change:+.2f}%  |  n = {int(reference['valid_trials'])} trials/model",
        transform=axis.transAxes, ha="left", va="top", fontsize=7.0, color="#334155", linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.4},
    )
    axis.text(50, 0.46, f"{float(reference['mean_energy_mj']):.2f} {arrow} {float(pareto['mean_energy_mj']):.2f} mJ  ({energy_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=7.2, color="#172033")
    axis.text(50, -0.54, f"{float(reference['mean_latency_ms']):.3f} {arrow} {float(pareto['mean_latency_ms']):.3f} ms  ({latency_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=7.2, color="#172033")
    save(figure, "Fig_Ab_headline_comparison")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_landscape_panel()
    make_headline_panel()


if __name__ == "__main__":
    main()
