"""Select the next CNN measurement batch from candidate prediction results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def capacity_energy_pareto_mask(parameter_count: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Mark rows on the Parameter Count (max) / Energy (min) Pareto frontier."""
    # Scan largest models first. A later row is dominated when an equal-or-larger
    # model already achieved equal-or-lower conservative energy.
    order = np.lexsort((energy, -parameter_count))
    result = np.zeros(len(parameter_count), dtype=bool)
    best_second = np.inf
    for index in order:
        if energy[index] < best_second:
            result[index] = True
            best_second = energy[index]
    return result


def add_ranked(selected: list[int], reasons: dict[int, str], ordered: np.ndarray, count: int, reason: str) -> None:
    added = 0
    for index in ordered:
        index = int(index)
        if index not in reasons:
            selected.append(index)
            reasons[index] = reason
            added += 1
            if added == count:
                return


def structural_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric_columns = [column for column in frame if column.startswith("feature_") and not column.endswith("_pattern")]
    categorical_columns = [column for column in frame if column.endswith("_pattern")]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scale = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, 1.0)
    numeric = (numeric - numeric.median()) / scale
    categorical = pd.get_dummies(frame[categorical_columns].astype(str), dtype=float)
    return np.column_stack((numeric.to_numpy(float), categorical.to_numpy(float)))


def add_diverse(frame: pd.DataFrame, selected: list[int], reasons: dict[int, str], count: int) -> None:
    eligible = np.flatnonzero(
        (frame["energy_upper_j"] <= frame["energy_upper_j"].quantile(0.75))
        & (frame["parameter_count"] >= frame["parameter_count"].quantile(0.25))
    )
    eligible = np.array([index for index in eligible if int(index) not in reasons], dtype=int)
    if not count or not len(eligible):
        return
    matrix = structural_matrix(frame)
    if selected:
        chosen = np.array(selected, dtype=int)
        distances = np.sqrt(((matrix[eligible, None] - matrix[chosen]) ** 2).sum(axis=2)).min(axis=1)
    else:
        distances = np.full(len(eligible), np.inf)
    for _ in range(min(count, len(eligible))):
        position = int(np.argmin(frame.iloc[eligible]["capacity_energy_score"].to_numpy())) if np.isinf(distances).all() else int(np.argmax(distances))
        choice = int(eligible[position])
        selected.append(choice)
        reasons[choice] = "structural_diversity"
        distances = np.minimum(distances, np.sqrt(((matrix[eligible] - matrix[choice]) ** 2).sum(axis=1)))
        distances[position] = -np.inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("measurements/search/candidate_predictions.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/search/next_measurement_candidates.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("measurements/search/selection_summary.json"))
    parser.add_argument("--energy-count", type=int, default=15)
    parser.add_argument("--pareto-count", type=int, default=15, help="Parameter Capacity-Energy Pareto candidates.")
    parser.add_argument("--uncertainty-count", type=int, default=10)
    parser.add_argument("--diversity-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.energy_count, args.pareto_count, args.uncertainty_count, args.diversity_count) < 0:
        raise ValueError("Selection counts must be non-negative.")
    for path in (args.output_csv, args.summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}. Use --overwrite to replace it.")
    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")
    required = {"candidate_id", "parameter_count", "energy_upper_j", "latency_upper_ms", "combined_relative_uncertainty"}
    missing = required.difference(frame.columns)
    if missing or frame.empty or not frame["candidate_id"].is_unique:
        raise ValueError(f"Invalid prediction CSV; missing columns: {sorted(missing)}")

    parameter_scale = frame["parameter_count"].median()
    energy_scale = frame["energy_upper_j"].median()
    frame["capacity_energy_score"] = frame["energy_upper_j"] / energy_scale - frame["parameter_count"] / parameter_scale
    frame["is_capacity_energy_pareto"] = capacity_energy_pareto_mask(
        frame["parameter_count"].to_numpy(), frame["energy_upper_j"].to_numpy()
    )
    frame["is_latency_energy_pareto"] = capacity_energy_pareto_mask(
        -frame["latency_upper_ms"].to_numpy(), frame["energy_upper_j"].to_numpy()
    )

    selected: list[int] = []
    reasons: dict[int, str] = {}
    all_indices = frame.index.to_numpy()
    add_ranked(selected, reasons, all_indices[np.argsort(frame["energy_upper_j"].to_numpy())], args.energy_count, "low_energy_ucb")
    frontier = frame.loc[frame["is_capacity_energy_pareto"]].sort_values("parameter_count", ascending=False)
    positions = np.linspace(0, len(frontier) - 1, min(args.pareto_count, len(frontier))).round().astype(int) if len(frontier) else []
    add_ranked(selected, reasons, frontier.iloc[np.unique(positions)].index.to_numpy() if len(frontier) else np.array([], dtype=int), args.pareto_count, "capacity_energy_pareto")
    add_ranked(selected, reasons, all_indices[np.argsort(-frame["combined_relative_uncertainty"].to_numpy())], args.uncertainty_count, "high_uncertainty")
    add_diverse(frame, selected, reasons, args.diversity_count)
    desired = args.energy_count + args.pareto_count + args.uncertainty_count + args.diversity_count
    add_ranked(selected, reasons, all_indices[np.argsort(frame["capacity_energy_score"].to_numpy())], desired - len(selected), "balanced_fill")

    output = frame.loc[selected].copy()
    output.insert(0, "selection_rank", range(1, len(output) + 1))
    output.insert(1, "selection_reason", [reasons[index] for index in selected])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")
    summary = {
        "candidate_count": len(frame),
        "selected_count": len(output),
        "primary_pareto": "parameter_count (maximize) and energy_upper_j (minimize)",
        "capacity_energy_pareto_count": int(frame["is_capacity_energy_pareto"].sum()),
        "latency_energy_pareto_count": int(frame["is_latency_energy_pareto"].sum()),
        "selection_reason_counts": output["selection_reason"].value_counts().to_dict(),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_csv} ({len(output)} candidates).")
    print(f"Primary Parameter Capacity-Energy Pareto candidates: {summary['capacity_energy_pareto_count']}")


if __name__ == "__main__":
    main()
