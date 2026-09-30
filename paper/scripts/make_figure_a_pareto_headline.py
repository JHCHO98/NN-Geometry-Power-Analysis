"""Create publication-ready Figure A: final Pareto search and headline outcome."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from select_certified_pareto import mean_pareto_mask  # noqa: E402


PREDICTIONS = PROJECT_ROOT / "measurements/search_2nd/candidate_predictions.csv"
HEADLINE = PROJECT_ROOT / "measurements/headline_comparison/analysis/headline_summary.csv"
OUTPUT_PREFIX = PROJECT_ROOT / "paper/figures/final/Fig_A_pareto_headline"


def load_pareto_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    frame = pd.read_csv(PREDICTIONS, encoding="utf-8-sig")
    # Use the same ensemble means as the interactive explorer and the final
    # Pareto-selection pipeline.  Point-prediction columns are intentionally
    # not used here: they can yield a different frontier.
    required = {"candidate_id", "ensemble_energy_mean_j", "ensemble_accuracy_mean_percent"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=required).copy()
    frame["energy_mj"] = pd.to_numeric(frame["ensemble_energy_mean_j"], errors="raise") * 1_000
    frame["accuracy_percent"] = pd.to_numeric(frame["ensemble_accuracy_mean_percent"], errors="raise")
    frame = frame[(frame["energy_mj"] > 0) & np.isfinite(frame["accuracy_percent"])].copy()
    mask = mean_pareto_mask(frame["accuracy_percent"].to_numpy(), frame["energy_mj"].to_numpy())
    selected = frame.loc[frame["candidate_id"].astype(str) == "C002793"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one C002793 row in final candidate predictions.")
    return frame, mask, selected.iloc[0]


def load_headline() -> pd.DataFrame:
    frame = pd.read_csv(HEADLINE, encoding="utf-8-sig")
    required = {
        "role", "structure_id", "parameter_count", "actual_accuracy_percent", "valid_trials",
        "mean_energy_mj", "std_energy_mj", "mean_latency_ms", "std_latency_ms",
        "energy_change_vs_reference_percent", "latency_change_vs_reference_percent",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Headline summary is missing columns: {sorted(missing)}")
    roles = set(frame["role"])
    if roles != {"reference", "pareto"}:
        raise ValueError("Headline summary must contain exactly reference and pareto rows.")
    return frame.set_index("role")


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

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.15, 3.35), gridspec_kw={"width_ratios": [1.52, 0.98]})
    figure.subplots_adjust(left=0.085, right=0.985, top=0.805, bottom=0.20, wspace=0.34)

    # (A) Final 50,000-candidate landscape. Values are the final-search predictions.
    left.scatter(
        candidates["energy_mj"], candidates["accuracy_percent"],
        s=3.2, color="#94a3b8", alpha=0.16, linewidths=0, rasterized=True, label="All candidates (n = 50,000)",
    )
    frontier = candidates.loc[pareto_mask].sort_values("energy_mj")
    left.plot(frontier["energy_mj"], frontier["accuracy_percent"], color="#172033", linewidth=1.8, label="Pareto frontier")
    left.scatter(frontier["energy_mj"], frontier["accuracy_percent"], s=9, color="#172033", zorder=3)
    # C002793 is the trained headline model shown in panel B.  It is retained
    # here as a reference point, not styled as a Pareto-optimal marker.
    left.scatter(
        [selected["energy_mj"]], [selected["accuracy_percent"]],
        s=38, color="#D55E00", linewidths=0, zorder=5, label="Headline model (C002793)",
    )
    left.annotate(
        "C002793", xy=(selected["energy_mj"], selected["accuracy_percent"]), xytext=(6, -13),
        textcoords="offset points", fontsize=7.1, color="#A94400",
    )
    left.axhline(85, color="#c3473c", linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9)
    left.text(left.get_xlim()[1] * 0.98, 85.11, "Accuracy = 85%", ha="right", va="bottom", fontsize=7.2, color="#a9362f")
    left.set_xscale("log")
    left.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set(xlabel="Predicted inference energy (mJ, log scale)", ylabel="Predicted CIFAR-10 accuracy (%)")
    left.set_ylim(candidates["accuracy_percent"].quantile(0.002) - 0.4, candidates["accuracy_percent"].max() + 1.20)
    left.spines[["top", "right"]].set_visible(False)
    left.tick_params(labelsize=8)
    left.legend(loc="lower right", frameon=False, fontsize=6.9, handlelength=1.8, labelspacing=0.35)
    left.text(-0.15, 1.03, "(A)", transform=left.transAxes, fontsize=11, fontweight="bold")

    # (B) Cost outcomes are normalized to the reference so Energy and Latency
    # can be compared within one compact panel. Raw means remain in annotations.
    reference_percent = np.array([100.0, 100.0])
    pareto_percent = np.array(
        [
            float(pareto["mean_energy_mj"]) / float(reference["mean_energy_mj"]) * 100,
            float(pareto["mean_latency_ms"]) / float(reference["mean_latency_ms"]) * 100,
        ]
    )
    y_positions = np.array([0.20, -0.80])
    right.barh(y_positions + 0.17, reference_percent, height=0.28, color="#64748B", label="Reference 0551")
    right.barh(y_positions - 0.17, pareto_percent, height=0.28, color="#2563EB", label="Pareto 0530")
    right.set(
        xlim=(0, 118), ylim=(-1.25, 1.35), yticks=y_positions, yticklabels=["Inference energy", "Inference latency"],
        xlabel="Measured value (% of reference 0551)",
    )
    right.set_xticks([0, 25, 50, 75, 100])
    right.axvline(100, color="#64748B", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7)
    right.grid(axis="x", color="#e4e9f0", linewidth=0.7)
    right.set_axisbelow(True)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.tick_params(labelsize=7.5)
    right.legend(loc="lower right", frameon=False, fontsize=7.0, handlelength=1.3, borderaxespad=0.1)

    energy_reduction = -float(pareto["energy_change_vs_reference_percent"])
    latency_reduction = -float(pareto["latency_change_vs_reference_percent"])
    parameter_change = (float(pareto["parameter_count"]) / float(reference["parameter_count"]) - 1) * 100
    accuracy_difference = float(pareto["actual_accuracy_percent"]) - float(reference["actual_accuracy_percent"])
    right.set_title("(B) Trained headline comparison", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    right.text(
        0.0, 0.97,
        f"Accuracy: {float(reference['actual_accuracy_percent']):.2f}% → {float(pareto['actual_accuracy_percent']):.2f}% "
        f"({accuracy_difference:+.2f} pp)\n"
        f"Parameters: {parameter_change:+.2f}%  |  n = {int(reference['valid_trials'])} trials/model",
        transform=right.transAxes, ha="left", va="top", fontsize=6.8, color="#334155", linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.4},
    )
    right.text(50, 0.51, f"{float(reference['mean_energy_mj']):.2f} → {float(pareto['mean_energy_mj']):.2f} mJ  ({energy_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=6.85, color="#172033")
    right.text(50, -0.49, f"{float(reference['mean_latency_ms']):.3f} → {float(pareto['mean_latency_ms']):.3f} ms  ({latency_reduction:.1f}% lower)", ha="center", va="bottom", fontsize=6.85, color="#172033")

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.16)
    figure.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
