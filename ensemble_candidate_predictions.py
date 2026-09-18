"""Blend first- and second-round CNN surrogate predictions for Pareto analysis.

The first surrogate is treated as a broad-coverage prior.  The second
surrogate is up-weighted for structures close to the 50 candidates measured
after the first search.  Those measured candidates themselves stay fixed at
their observed values; they are never averaged back into predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "accuracy": {
        "point": "predicted_accuracy_percent",
        "mean": "ensemble_accuracy_mean_percent",
        "std": "accuracy_uncertainty_std_percent",
        "lower": "accuracy_lower_percent",
        "upper": "accuracy_upper_percent",
        "relative": "accuracy_relative_uncertainty",
    },
    "energy": {
        "point": "predicted_energy_j",
        "mean": "ensemble_energy_mean_j",
        "std": "energy_uncertainty_std_j",
        "lower": "energy_lower_j",
        "upper": "energy_upper_j",
        "relative": "energy_relative_uncertainty",
    },
    "latency": {
        "point": "predicted_latency_ms",
        "mean": "ensemble_latency_mean_ms",
        "std": "latency_uncertainty_std_ms",
        "lower": "latency_lower_ms",
        "upper": "latency_upper_ms",
        "relative": "latency_relative_uncertainty",
    },
}

NUMERIC_DISTANCE_COLUMNS = ("depth", "pool_count", "parameter_count")
CATEGORICAL_DISTANCE_COLUMNS = ("pattern", "growth_pattern")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-predictions",
        type=Path,
        default=Path("measurements/search/candidate_predictions.csv"),
        help="First-round prediction table (the global prior).",
    )
    parser.add_argument(
        "--second-predictions",
        type=Path,
        default=Path("measurements/search_2nd/candidate_predictions.csv"),
        help="Second-round table with measured candidates already fixed as anchors.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("measurements/search_ensemble/candidate_predictions.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("measurements/search_ensemble/ensemble_summary.json"),
    )
    parser.add_argument(
        "--second-weight-floor",
        type=float,
        default=0.25,
        help="Minimum second-round weight for structures far from new measurements.",
    )
    parser.add_argument(
        "--second-weight-ceiling",
        type=float,
        default=0.85,
        help="Maximum second-round weight near new measurements.",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=None,
        help="Distance decay scale. Default: median nearest-anchor distance.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def measured_mask(frame: pd.DataFrame) -> np.ndarray:
    if "is_measured_candidate" not in frame.columns:
        raise ValueError("Second-round predictions must contain is_measured_candidate.")
    values = frame["is_measured_candidate"]
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes")).to_numpy()


def validate_inputs(first: pd.DataFrame, second: pd.DataFrame) -> None:
    for label, frame in (("first", first), ("second", second)):
        if "candidate_id" not in frame or not frame["candidate_id"].is_unique:
            raise ValueError(f"{label} predictions need unique candidate_id values.")
        for target, columns in TARGETS.items():
            missing = {columns["point"], columns["mean"], columns["std"]}.difference(frame.columns)
            if missing:
                raise ValueError(f"{label} predictions lack {target} columns: {sorted(missing)}")
    if set(first["candidate_id"]) != set(second["candidate_id"]):
        raise ValueError("First- and second-round prediction tables do not contain the same candidate IDs.")
    missing_structure = set(NUMERIC_DISTANCE_COLUMNS + CATEGORICAL_DISTANCE_COLUMNS).difference(second.columns)
    if missing_structure:
        raise ValueError(f"Second-round predictions lack distance features: {sorted(missing_structure)}")


def nearest_anchor_distance(frame: pd.DataFrame, anchors: np.ndarray) -> np.ndarray:
    """Return a simple Gower-style structural distance to the closest anchor."""
    pieces: list[np.ndarray] = []
    for column in NUMERIC_DISTANCE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if column == "parameter_count":
            values = np.log10(np.maximum(values, 1.0))
        scale = np.subtract(*np.nanpercentile(values, [75, 25]))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        difference = np.abs(values[:, None] - values[anchors][None, :]) / scale
        pieces.append(difference)
    for column in CATEGORICAL_DISTANCE_COLUMNS:
        values = frame[column].astype(str).to_numpy()
        pieces.append((values[:, None] != values[anchors][None, :]).astype(float))

    squared = sum(part**2 for part in pieces)
    return np.sqrt(squared / len(pieces)).min(axis=1)


def blend_target(
    output: pd.DataFrame,
    first: pd.DataFrame,
    second: pd.DataFrame,
    target: str,
    second_weight: np.ndarray,
    observed: np.ndarray,
) -> None:
    columns = TARGETS[target]
    first_point = pd.to_numeric(first[columns["point"]], errors="raise").to_numpy(float)
    second_point = pd.to_numeric(second[columns["point"]], errors="raise").to_numpy(float)
    first_mean = pd.to_numeric(first[columns["mean"]], errors="raise").to_numpy(float)
    second_mean = pd.to_numeric(second[columns["mean"]], errors="raise").to_numpy(float)
    first_std = pd.to_numeric(first[columns["std"]], errors="raise").to_numpy(float)
    second_std = pd.to_numeric(second[columns["std"]], errors="raise").to_numpy(float)

    first_weight = 1.0 - second_weight
    output[f"first_{columns['mean']}"] = first_mean
    output[f"second_{columns['mean']}"] = second_mean
    output[columns["point"]] = first_weight * first_point + second_weight * second_point
    output[columns["mean"]] = first_weight * first_mean + second_weight * second_mean
    output[columns["std"]] = np.sqrt((first_weight * first_std) ** 2 + (second_weight * second_std) ** 2)

    # The already measured models are Pareto anchors, not blended estimates.
    output.loc[observed, columns["point"]] = second_point[observed]
    output.loc[observed, columns["mean"]] = second_mean[observed]
    output.loc[observed, columns["std"]] = 0.0

    mean = output[columns["mean"]].to_numpy(float)
    std = output[columns["std"]].to_numpy(float)
    lower = mean - std
    if target in ("energy", "latency"):
        lower = np.maximum(lower, 1e-9)
    output[columns["lower"]] = lower
    output[columns["upper"]] = mean + std
    output[columns["relative"]] = std / np.maximum(np.abs(mean), np.finfo(float).eps)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.second_weight_floor <= args.second_weight_ceiling <= 1.0:
        raise ValueError("Second-round weights must satisfy 0 <= floor <= ceiling <= 1.")
    if args.distance_scale is not None and args.distance_scale <= 0:
        raise ValueError("--distance-scale must be positive.")
    for output_path in (args.output_csv, args.summary_json):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}. Use --overwrite to replace it.")

    first = pd.read_csv(args.first_predictions, encoding="utf-8-sig")
    second = pd.read_csv(args.second_predictions, encoding="utf-8-sig")
    validate_inputs(first, second)
    first = first.set_index("candidate_id").loc[second["candidate_id"]].reset_index()
    observed = measured_mask(second)
    anchors = np.flatnonzero(observed)
    if not len(anchors):
        raise ValueError("No measured anchors found in second-round predictions.")

    distance = nearest_anchor_distance(second, anchors)
    nonzero = distance[distance > 0]
    scale = args.distance_scale if args.distance_scale is not None else float(np.median(nonzero))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    second_weight = args.second_weight_floor + (
        args.second_weight_ceiling - args.second_weight_floor
    ) * np.exp(-distance / scale)
    second_weight[observed] = 1.0

    output = second.copy()
    output["distance_to_new_measurement"] = distance
    output["ensemble_second_weight"] = second_weight
    output["ensemble_first_weight"] = 1.0 - second_weight
    output["ensemble_method"] = "distance_weighted_round1_round2"
    for target in TARGETS:
        blend_target(output, first, second, target, second_weight, observed)
    relative_columns = [columns["relative"] for columns in TARGETS.values()]
    output["combined_relative_uncertainty"] = output[relative_columns].mean(axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")
    summary = {
        "candidate_count": len(output),
        "measured_anchor_count": int(observed.sum()),
        "method": "distance_weighted_round1_round2",
        "numeric_distance_features": list(NUMERIC_DISTANCE_COLUMNS),
        "categorical_distance_features": list(CATEGORICAL_DISTANCE_COLUMNS),
        "distance_scale": scale,
        "second_weight_floor": args.second_weight_floor,
        "second_weight_ceiling": args.second_weight_ceiling,
        "second_weight_min": float(second_weight.min()),
        "second_weight_median": float(np.median(second_weight)),
        "second_weight_max": float(second_weight.max()),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_csv} ({len(output):,} candidates, {int(observed.sum())} measured anchors).")
    print(f"Second-round weight: median={np.median(second_weight):.3f}, decay scale={scale:.3f}")


if __name__ == "__main__":
    main()
