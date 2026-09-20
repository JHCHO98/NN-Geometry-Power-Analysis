"""Summarize the trained 0551-versus-0530 ABBA benchmark comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


FIELDS = [
    "role",
    "structure_id",
    "parameter_count",
    "actual_accuracy_percent",
    "valid_trials",
    "mean_energy_mj",
    "std_energy_mj",
    "mean_latency_ms",
    "std_latency_ms",
    "energy_change_vs_reference_percent",
    "latency_change_vs_reference_percent",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-trials",
        type=Path,
        default=Path("measurements/headline_comparison/processed/energy_trials.csv"),
    )
    parser.add_argument(
        "--headline-models",
        type=Path,
        default=Path("measurements/headline_comparison/headline_models.csv"),
    )
    parser.add_argument("--pair-id", default="headline_0551_vs_0530")
    parser.add_argument("--benchmark-session-id", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/headline_comparison/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = {row["role"]: row for row in read_csv(args.headline_models)}
    if set(models) != {"reference", "pareto"}:
        raise ValueError("Headline model registry must contain exactly reference and pareto rows.")
    values: dict[str, dict[str, list[float]]] = {role: {"energy": [], "latency": []} for role in models}
    state_to_role = {f"{role}_{row['structure_id']}_trained": role for role, row in models.items()}
    for row in read_csv(args.energy_trials):
        if row.get("mode") != "inference" or row.get("status") != "valid" or row.get("pair_id") != args.pair_id:
            continue
        if args.benchmark_session_id and row.get("benchmark_session_id") != args.benchmark_session_id:
            continue
        role = state_to_role.get(row.get("weight_state", ""))
        if role is None:
            continue
        values[role]["energy"].append(float(row["net_energy_per_inference_j"]) * 1_000)
        values[role]["latency"].append(float(row["average_latency_ms"]))
    if any(not values[role]["energy"] for role in values):
        raise ValueError("No complete headline measurements found for both trained models.")

    rows: list[dict[str, object]] = []
    for role in ("reference", "pareto"):
        model = models[role]
        energy_values = values[role]["energy"]
        latency_values = values[role]["latency"]
        rows.append(
            {
                "role": role,
                "structure_id": model["structure_id"],
                "parameter_count": model["parameter_count"],
                "actual_accuracy_percent": model["actual_accuracy_percent"],
                "valid_trials": len(energy_values),
                "mean_energy_mj": mean(energy_values),
                "std_energy_mj": stdev(energy_values) if len(energy_values) > 1 else 0.0,
                "mean_latency_ms": mean(latency_values),
                "std_latency_ms": stdev(latency_values) if len(latency_values) > 1 else 0.0,
            }
        )
    reference, pareto = rows
    for metric, output_name in (("energy", "energy_change_vs_reference_percent"), ("latency", "latency_change_vs_reference_percent")):
        reference_value = float(reference[f"mean_{metric}_{'mj' if metric == 'energy' else 'ms'}"])
        pareto_value = float(pareto[f"mean_{metric}_{'mj' if metric == 'energy' else 'ms'}"])
        reference[output_name] = 0.0
        pareto[output_name] = (pareto_value - reference_value) / reference_value * 100

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "headline_summary.csv", rows)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    labels = ["Reference 0551", "Pareto 0530"]
    for axis, key, unit, title in (
        (axes[0], "energy", "mJ", "Trained inference energy"),
        (axes[1], "latency", "ms", "Trained inference latency"),
    ):
        means = [float(row[f"mean_{key}_{'mj' if key == 'energy' else 'ms'}"]) for row in rows]
        deviations = [float(row[f"std_{key}_{'mj' if key == 'energy' else 'ms'}"]) for row in rows]
        axis.bar(labels, means, yerr=deviations, capsize=5, color=["#64748b", "#2563eb"])
        axis.set(ylabel=unit, title=title)
    figure.savefig(args.output_dir / "headline_comparison.png", dpi=200)
    plt.close(figure)
    print(f"Wrote headline comparison for {pareto['valid_trials']} trials per model to {args.output_dir}")


if __name__ == "__main__":
    main()
