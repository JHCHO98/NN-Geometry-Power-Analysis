"""Generate random CNN configurations, export them to ONNX, and record metadata."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random

import torch

from FlexibleCNN import (
    CHANNEL_PATTERNS,
    GROWTH_PATTERNS,
    FlexibleCNN,
    ModelConfig,
    random_config,
)


CSV_FIELDS = [
    "id",
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


def format_values(values: list[int]) -> str:
    return "-".join(map(str, values))


def available_patterns(depth: int) -> tuple[str, ...]:
    """Return channel patterns that remain distinct at the given depth."""
    if depth == 2:
        # With two layers, hourglass and inverse_hourglass collapse to the
        # same endpoint arrangements as decreasing and increasing.
        return ("increasing", "decreasing", "uniform")
    return tuple(sorted(CHANNEL_PATTERNS))


def next_model_id(csv_path: Path) -> int:
    """Return the next numeric ID, continuing an existing structure CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        ids = [int(row["id"]) for row in rows if row.get("id", "").isdigit()]
    return max(ids, default=0) + 1


def export_and_record(
    config: ModelConfig,
    model_id: int,
    onnx_directory: Path,
    csv_path: Path,
) -> Path:
    """Export one config to ``0001.onnx`` and append its structure record."""
    onnx_directory.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    id_text = f"{model_id:04d}"
    onnx_path = onnx_directory / f"{id_text}.onnx"
    if onnx_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {onnx_path}")

    # The model weights do not affect the architecture, but a fixed seed makes
    # each exported untrained model reproducible as well.
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
        # PyTorch 2.9 defaults to the dynamo exporter, which requires the
        # optional onnxscript package. The legacy exporter is sufficient here.
        dynamo=False,
    )

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row = {
        "id": id_text,
        "seed": config.seed,
        "depth": config.depth,
        "pattern": config.pattern,
        "growth_pattern": config.growth_pattern,
        "noise_ratio": config.noise_ratio,
        "min_channels": config.min_channels,
        "max_channels": config.max_channels,
        "channels": format_values(config.channels),
        "pools": format_values(config.pools),
        "parameter_count": config.parameter_count,
        "onnx_path": onnx_path.as_posix(),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return onnx_path


def generate_dataset(
    count: int,
    base_seed: int,
    onnx_directory: Path = Path("model_onnx"),
    csv_path: Path = Path("dataset_structure.csv"),
    min_parameters: int = 5_000,
    max_parameters: int = 2_000_000,
    no_pool_probability: float = 0.15,
) -> list[Path]:
    """Generate and export ``count`` randomly sampled CNNs."""
    if count <= 0:
        raise ValueError("count must be positive.")
    if not 0.0 <= no_pool_probability <= 1.0:
        raise ValueError("no_pool_probability must be in the range [0.0, 1.0].")

    selector = random.Random(base_seed)
    model_id = next_model_id(csv_path)
    exported_paths: list[Path] = []

    for _ in range(count):
        depth = selector.randint(2, 6)
        pool_count = (
            0
            if selector.random() < no_pool_probability
            else selector.randint(1, min(5, depth))
        )
        config = random_config(
            seed=selector.randrange(2**63),
            depth=depth,
            pattern=selector.choice(available_patterns(depth)),
            growth_pattern=selector.choice(sorted(GROWTH_PATTERNS)),
            noise_ratio=selector.choice([0.0, 0.05, 0.10]),
            pool_count=pool_count,
            min_parameters=min_parameters,
            max_parameters=max_parameters,
        )
        exported_paths.append(export_and_record(config, model_id, onnx_directory, csv_path))
        model_id += 1

    return exported_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="Number of models to generate.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--onnx-dir", type=Path, default=Path("model_onnx"))
    parser.add_argument("--csv-path", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument("--min-parameters", type=int, default=5_000)
    parser.add_argument("--max-parameters", type=int, default=2_000_000)
    parser.add_argument(
        "--no-pool-probability",
        type=float,
        default=0.15,
        help="Probability of generating a model without MaxPool layers (default: 0.15).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = generate_dataset(
        count=args.count,
        base_seed=args.base_seed,
        onnx_directory=args.onnx_dir,
        csv_path=args.csv_path,
        min_parameters=args.min_parameters,
        max_parameters=args.max_parameters,
        no_pool_probability=args.no_pool_probability,
    )
    for path in paths:
        print(f"Exported {path}")
