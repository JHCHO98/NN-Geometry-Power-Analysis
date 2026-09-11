"""Convert cumulative energy analysis results into an XGBoost-ready model dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "model_id",
    "valid_trials",
    "mean_net_energy_per_inference_j",
    "std_net_energy_per_inference_j",
    "cv_percent",
    "mean_average_latency_ms",
    "mean_throughput_inferences_per_sec",
    "depth",
    "pattern",
    "growth_pattern",
    "channels",
    "pools",
    "parameter_count",
}


def parse_int_sequence(value: object) -> list[int]:
    """Parse dash-separated channel or pooling values from a structure CSV."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        return [int(part) for part in text.split("-")]
    except ValueError as error:
        raise ValueError(f"Expected a dash-separated integer sequence, got {value!r}.") from error


def engineering_features(
    channels: list[int], pools: list[int], image_size: int, max_depth: int
) -> dict[str, float]:
    """Calculate geometry features without using measured energy or latency."""
    if not channels:
        raise ValueError("A model must contain at least one channel value.")
    if len(channels) > max_depth:
        raise ValueError(f"Model depth {len(channels)} exceeds --max-depth {max_depth}.")

    features: dict[str, float] = {}
    for index in range(1, max_depth + 1):
        features[f"feature_channel_{index}"] = (
            float(channels[index - 1]) if index <= len(channels) else np.nan
        )
        features[f"feature_pool_after_{index}"] = float(index in pools)

    channel_values = np.asarray(channels, dtype=float)
    features.update(
        {
            "feature_channel_min": float(channel_values.min()),
            "feature_channel_max": float(channel_values.max()),
            "feature_channel_mean": float(channel_values.mean()),
            "feature_channel_std": float(channel_values.std(ddof=0)),
            "feature_channel_sum": float(channel_values.sum()),
            "feature_channel_range": float(channel_values.max() - channel_values.min()),
            "feature_channel_first": float(channel_values[0]),
            "feature_channel_last": float(channel_values[-1]),
            "feature_channel_last_first_ratio": float(channel_values[-1] / channel_values[0]),
            "feature_pool_count": float(len(pools)),
        }
    )

    height = image_size
    width = image_size
    input_channels = 3
    conv_macs = 0
    activation_elements = 0
    for block_index, output_channels in enumerate(channels, start=1):
        conv_macs += height * width * input_channels * output_channels * 3 * 3
        activation_elements += height * width * output_channels
        if block_index in pools:
            height //= 2
            width //= 2
        input_channels = output_channels

    features.update(
        {
            "feature_conv_macs": float(conv_macs),
            "feature_conv_flops": float(conv_macs * 2),
            "feature_activation_elements": float(activation_elements),
            "feature_final_feature_map_height": float(height),
            "feature_final_feature_map_width": float(width),
        }
    )
    return features


def build_dataset(summary: pd.DataFrame, image_size: int, max_depth: int) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"Energy summary is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for source_row in summary.to_dict(orient="records"):
        model_id = str(source_row["model_id"]).zfill(4)
        channels = parse_int_sequence(source_row["channels"])
        pools = parse_int_sequence(source_row["pools"])
        depth = int(source_row["depth"])
        if len(channels) != depth:
            raise ValueError(
                f"Model {model_id}: depth is {depth}, but channels contains {len(channels)} values."
            )

        row: dict[str, object] = {
            "model_id": model_id,
            "target_energy_j": float(source_row["mean_net_energy_per_inference_j"]),
            "target_energy_std_j": float(source_row["std_net_energy_per_inference_j"]),
            "target_energy_cv_percent": float(source_row["cv_percent"]),
            "target_latency_ms": float(source_row["mean_average_latency_ms"]),
            "target_throughput_inferences_per_sec": float(
                source_row["mean_throughput_inferences_per_sec"]
            ),
            "measurement_valid_trials": int(source_row["valid_trials"]),
            "feature_depth": depth,
            "feature_parameter_count": int(source_row["parameter_count"]),
            "feature_pattern": str(source_row["pattern"]),
            "feature_growth_pattern": str(source_row["growth_pattern"]),
        }
        row.update(engineering_features(channels, pools, image_size, max_depth))
        rows.append(row)

    dataset = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)
    if (dataset["target_energy_j"] <= 0).any():
        invalid_ids = dataset.loc[dataset["target_energy_j"] <= 0, "model_id"].tolist()
        raise ValueError(f"Net energy must be positive for XGBoost log targets. Invalid models: {invalid_ids}")
    if (dataset["target_latency_ms"] <= 0).any():
        invalid_ids = dataset.loc[dataset["target_latency_ms"] <= 0, "model_id"].tolist()
        raise ValueError(f"Latency must be positive. Invalid models: {invalid_ids}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-summary",
        type=Path,
        default=Path("measurements/processed/energy_summary.csv"),
        help="Cumulative output from analyze_energy.py.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("measurements/ml/model_dataset.csv")
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.max_depth <= 0:
        raise ValueError("--image-size and --max-depth must be positive.")
    summary = pd.read_csv(args.energy_summary, encoding="utf-8-sig")
    dataset = build_dataset(summary, args.image_size, args.max_depth)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Wrote {args.output_csv} with {len(dataset)} models and {len(dataset.columns)} columns.")


if __name__ == "__main__":
    main()
