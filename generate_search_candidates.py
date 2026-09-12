"""Generate diverse, unmeasured CNN architectures for surrogate-guided search.

The output is a structure-only candidate table: it does not export ONNX files
or modify dataset_structure.csv.  That separation prevents thousands of
unselected candidates from filling model_onnx/ and lets the next scoring stage
choose a small measurement batch first.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random

import pandas as pd

from FlexibleCNN import GROWTH_PATTERNS, random_config
from generate_dataset import available_patterns, format_values
from prepare_xgboost_dataset import engineering_features


CONFIGURATION_FIELDS = [
    "candidate_id",
    "seed",
    "depth",
    "pattern",
    "growth_pattern",
    "noise_ratio",
    "pool_count",
    "pools",
    "min_channels",
    "max_channels",
    "channels",
    "parameter_count",
    "parameter_bin",
    "architecture_key",
]


def architecture_key(channels: list[int], pools: list[int]) -> str:
    """Return an identifier for the executable architecture, not its seed."""
    return f"{'-'.join(map(str, channels))}|{'-'.join(map(str, pools))}"


def existing_architectures(structure_csv: Path) -> set[str]:
    """Read structures already generated or measured, if the CSV exists."""
    if not structure_csv.exists():
        return set()
    with structure_csv.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            f"{row['channels'].strip()}|{row['pools'].strip()}"
            for row in csv.DictReader(file)
            if row.get("channels") and row.get("pools") is not None
        }


def parameter_bin_bounds(
    bin_index: int, bin_count: int, min_parameters: int, max_parameters: int
) -> tuple[int, int]:
    """Split the parameter range into logarithmically spaced inclusive bins."""
    lower_log = math.log(min_parameters)
    upper_log = math.log(max_parameters)
    lower = math.ceil(math.exp(lower_log + (upper_log - lower_log) * bin_index / bin_count))
    upper = math.floor(
        math.exp(lower_log + (upper_log - lower_log) * (bin_index + 1) / bin_count)
    )
    if bin_index == 0:
        lower = min_parameters
    if bin_index == bin_count - 1:
        upper = max_parameters
    return lower, upper


def sampling_cells(parameter_bins: int) -> list[tuple[int, str, str, float, int, int]]:
    """Create a balanced grid over categorical geometry choices and size bins."""
    cells: list[tuple[int, str, str, float, int, int]] = []
    for depth in range(2, 7):
        for pattern in available_patterns(depth):
            for growth_pattern in sorted(GROWTH_PATTERNS):
                for noise_ratio in (0.0, 0.05, 0.10):
                    for pool_count in range(0, min(5, depth) + 1):
                        for parameter_bin in range(parameter_bins):
                            cells.append(
                                (depth, pattern, growth_pattern, noise_ratio, pool_count, parameter_bin)
                            )
    return cells


def generate_candidates(args: argparse.Namespace) -> pd.DataFrame:
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.parameter_bins <= 0:
        raise ValueError("--parameter-bins must be positive.")
    if args.min_parameters <= 0 or args.max_parameters < args.min_parameters:
        raise ValueError("Parameter bounds must satisfy 0 < min <= max.")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output_csv}. Use --overwrite to replace it."
        )

    seen = existing_architectures(args.existing_structures)
    existing_count = len(seen)
    cells = sampling_cells(args.parameter_bins)
    selector = random.Random(args.base_seed)
    selector.shuffle(cells)
    rows: list[dict[str, object]] = []
    attempts = 0
    max_attempts = args.count * args.max_attempts_per_candidate

    while len(rows) < args.count and attempts < max_attempts:
        cell = cells[attempts % len(cells)]
        attempts += 1
        depth, pattern, growth_pattern, noise_ratio, pool_count, parameter_bin = cell
        lower, upper = parameter_bin_bounds(
            parameter_bin, args.parameter_bins, args.min_parameters, args.max_parameters
        )
        try:
            config = random_config(
                seed=selector.randrange(2**63),
                depth=depth,
                pattern=pattern,
                growth_pattern=growth_pattern,
                noise_ratio=noise_ratio,
                pool_count=pool_count,
                min_parameters=lower,
                max_parameters=upper,
                max_attempts=args.config_attempts,
            )
        except ValueError:
            # Some depth/size-bin combinations cannot exist under the current
            # channel limits. Other cells continue to supply balanced coverage.
            continue

        key = architecture_key(config.channels, config.pools)
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, object] = {
            "candidate_id": f"C{len(rows) + 1:06d}",
            "seed": config.seed,
            "depth": config.depth,
            "pattern": config.pattern,
            "growth_pattern": config.growth_pattern,
            "noise_ratio": config.noise_ratio,
            "pool_count": len(config.pools),
            "pools": format_values(config.pools),
            "min_channels": config.min_channels,
            "max_channels": config.max_channels,
            "channels": format_values(config.channels),
            "parameter_count": config.parameter_count,
            "parameter_bin": parameter_bin + 1,
            "architecture_key": key,
            # These names deliberately match model_dataset.csv so the saved
            # XGBoost preprocessors can score this table without remapping.
            "feature_depth": config.depth,
            "feature_parameter_count": config.parameter_count,
            "feature_pattern": config.pattern,
            "feature_growth_pattern": config.growth_pattern,
        }
        row.update(engineering_features(config.channels, config.pools, args.image_size, 6))
        rows.append(row)

    if len(rows) < args.count:
        raise RuntimeError(
            f"Generated only {len(rows)} unique candidates after {attempts} attempts. "
            "Reduce --count, widen the parameter range, or increase --max-attempts-per-candidate."
        )

    candidates = pd.DataFrame(rows)
    candidates.attrs["existing_count"] = existing_count
    candidates.attrs["attempts"] = attempts
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=20260911)
    parser.add_argument("--existing-structures", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/search/candidates.csv"))
    parser.add_argument("--min-parameters", type=int, default=5_000)
    parser.add_argument("--max-parameters", type=int, default=2_000_000)
    parser.add_argument("--parameter-bins", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--config-attempts", type=int, default=25)
    parser.add_argument("--max-attempts-per-candidate", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = generate_candidates(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(
        f"Wrote {args.output_csv} with {len(candidates)} unmeasured, unique candidates "
        f"({candidates.attrs['existing_count']} existing architectures excluded; "
        f"{candidates.attrs['attempts']} sampling attempts)."
    )


if __name__ == "__main__":
    main()
