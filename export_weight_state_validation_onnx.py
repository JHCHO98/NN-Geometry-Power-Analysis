"""Recreate untrained ONNX models for the weight-state validation sample.

The original generated ONNX files are reproducible from each row's structure
and seed.  This script exports those initial-weight models to a separate
location, leaving ``model_onnx/`` and the trained-model artifact directory
untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import pandas as pd
import torch

from FlexibleCNN import FlexibleCNN, ModelConfig


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


def make_config(row: pd.Series) -> ModelConfig:
    return ModelConfig(
        depth=int(row["depth"]),
        channels=parse_int_sequence(row["channels"]),
        pools=parse_int_sequence(row["pools"]),
        pattern=str(row["pattern"]),
        growth_pattern=str(row["growth_pattern"]),
        noise_ratio=float(row["noise_ratio"]),
        min_channels=int(row["min_channels"]),
        max_channels=int(row["max_channels"]),
        seed=int(row["seed"]),
        parameter_count=int(row["parameter_count"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("measurements/validation_weight_state/validation_sample_25.csv"),
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("measurements/validation_weight_state/onnx_untrained"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("measurements/validation_weight_state/untrained_onnx_metadata.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sample_csv.is_file():
        raise FileNotFoundError(f"Validation sample CSV not found: {args.sample_csv}")

    sample = pd.read_csv(args.sample_csv, encoding="utf-8-sig")
    required = {
        "id", "seed", "depth", "channels", "pools", "pattern", "growth_pattern",
        "noise_ratio", "min_channels", "max_channels", "parameter_count",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"Sample CSV is missing required columns: {sorted(missing)}")

    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    dummy_input = torch.zeros(1, 3, 32, 32, dtype=torch.float32)

    for row in sample.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        model_id = f"{int(row_series['id']):04d}"
        onnx_path = args.onnx_dir / f"{model_id}_untrained.onnx"
        if onnx_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing ONNX file: {onnx_path}")

        config = make_config(row_series)
        # This matches generate_dataset.export_and_record: architecture and
        # initial weights both come from the stored model seed.
        torch.manual_seed(config.seed)
        model = FlexibleCNN(config).eval()
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
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        rows.append(
            {
                "id": model_id,
                "weight_state": "untrained_seed_reconstructed",
                "source_seed": config.seed,
                "onnx_path": onnx_path.as_posix(),
                "onnx_size_bytes": onnx_path.stat().st_size,
            }
        )
        print(f"Exported and validated {onnx_path}")

    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.metadata_csv, index=False, encoding="utf-8-sig")
    print(f"Metadata: {args.metadata_csv}")


if __name__ == "__main__":
    main()
