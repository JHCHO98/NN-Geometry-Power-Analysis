"""Export selected candidate CNNs to ONNX and generate a training target CSV for Colab."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import onnx
import pandas as pd
import torch

from FlexibleCNN import FlexibleCNN, ModelConfig


CSV_FIELDS = [
    "id",
    "candidate_id",
    "selection_reason",
    "seed",
    "depth",
    "pattern",
    "growth_pattern",
    "noise_ratio",
    "min_channels",
    "max_channels",
    "channels",
    "pools",
    "parameter_count",
    "onnx_path",
    "onnx_size_bytes",
    "created_at_utc",
]


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


def export_candidate_onnx(
    config: ModelConfig,
    model_id_str: str,
    onnx_directory: Path,
) -> Path:
    """Export FlexibleCNN to ONNX and run onnx.checker validation."""
    onnx_directory.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_directory / f"{model_id_str}.onnx"

    if onnx_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {onnx_path}")

    torch.manual_seed(config.seed)
    model = FlexibleCNN(config).eval()
    dummy_input = torch.zeros(1, 3, 32, 32, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Validate ONNX file
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("measurements/search/next_measurement_candidates.csv"),
    )
    parser.add_argument(
        "--dataset-structure-csv",
        type=Path,
        default=Path("dataset_structure.csv"),
    )
    parser.add_argument(
        "--colab-train-csv",
        type=Path,
        default=Path("measurements/ml/candidates_to_train_50.csv"),
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("model_onnx"),
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=501,
        help="Starting integer ID for candidate models (default: 501)",
    )
    args = parser.parse_args()

    if not args.candidates_csv.exists():
        raise FileNotFoundError(f"Missing candidates CSV: {args.candidates_csv}")

    df = pd.read_csv(args.candidates_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} candidate models from {args.candidates_csv}.")

    records_for_colab = []
    records_for_structure = []

    current_id = args.start_id

    for _, row in df.iterrows():
        id_str = f"{current_id:04d}"
        channels = parse_int_sequence(row["channels"])
        pools = parse_int_sequence(row["pools"])

        config = ModelConfig(
            depth=int(row["depth"]),
            channels=channels,
            pools=pools,
            pattern=str(row["pattern"]),
            growth_pattern=str(row["growth_pattern"]),
            noise_ratio=float(row["noise_ratio"]),
            min_channels=int(row["min_channels"]),
            max_channels=int(row["max_channels"]),
            seed=int(row["seed"]),
            parameter_count=int(row["parameter_count"]),
        )

        onnx_path = export_candidate_onnx(config, id_str, args.onnx_dir)
        onnx_size = onnx_path.stat().st_size

        structure_record = {
            "id": id_str,
            "candidate_id": str(row["candidate_id"]),
            "selection_reason": str(row["selection_reason"]),
            "seed": config.seed,
            "depth": config.depth,
            "pattern": config.pattern,
            "growth_pattern": config.growth_pattern,
            "noise_ratio": config.noise_ratio,
            "min_channels": config.min_channels,
            "max_channels": config.max_channels,
            "channels": row["channels"],
            "pools": row["pools"],
            "parameter_count": config.parameter_count,
            "onnx_path": onnx_path.as_posix(),
            "onnx_size_bytes": onnx_size,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        records_for_structure.append(structure_record)

        # Record for Colab training (same schema as selected_50_models.csv)
        colab_record = {
            "id": id_str,
            "candidate_id": str(row["candidate_id"]),
            "selection_reason": str(row["selection_reason"]),
            "seed": config.seed,
            "depth": config.depth,
            "pattern": config.pattern,
            "growth_pattern": config.growth_pattern,
            "noise_ratio": config.noise_ratio,
            "min_channels": config.min_channels,
            "max_channels": config.max_channels,
            "channels": row["channels"],
            "pools": row["pools"],
            "parameter_count": config.parameter_count,
            "onnx_path": onnx_path.as_posix(),
            "onnx_size_bytes": onnx_size,
            "created_at_utc": structure_record["created_at_utc"],
            "synflow_score": row.get("feature_synflow_score", ""),
            "grad_norm_score": row.get("feature_grad_norm_score", ""),
            "jacob_cov_score": row.get("feature_jacob_cov_score", ""),
        }
        records_for_colab.append(colab_record)

        print(f"Exported [{id_str}] ({row['candidate_id']}) -> {onnx_path.name} ({config.parameter_count:,} params)")
        current_id += 1

    # Save training targets for Colab
    args.colab_train_csv.parent.mkdir(parents=True, exist_ok=True)
    colab_df = pd.DataFrame(records_for_colab)
    colab_df.to_csv(args.colab_train_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved Colab training task list: {args.colab_train_csv}")

    # Also save a candidate metadata map
    candidate_meta_csv = Path("measurements/search/candidate_onnx_metadata.csv")
    pd.DataFrame(records_for_structure).to_csv(candidate_meta_csv, index=False, encoding="utf-8-sig")
    print(f"Saved candidate ONNX metadata map: {candidate_meta_csv}")
    print(f"Successfully exported {len(records_for_structure)} models ({args.start_id:04d} ~ {current_id - 1:04d}.onnx).")


if __name__ == "__main__":
    main()
