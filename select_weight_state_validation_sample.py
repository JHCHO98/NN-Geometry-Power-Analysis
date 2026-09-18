"""Select a reproducible, structurally stratified sample for weight-state validation.

The sample represents the 500 original generated architectures rather than a
single promising region of the search space.  It balances the five channel
patterns and parameter-count quantiles, then spreads the selected models over
depth and pooling-ratio categories within those required strata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PATTERN_ORDER = [
    "uniform",
    "increasing",
    "decreasing",
    "hourglass",
    "inverse_hourglass",
]


def parse_pool_count(value: object) -> int:
    """Return the number of pooling operations encoded as ``1-3-5``."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    return len(text.split("-"))


def depth_category(depth: int) -> str:
    if depth <= 3:
        return "low_2_3"
    if depth <= 5:
        return "mid_4_5"
    return "high_6"


def pool_ratio_category(pool_ratio: float) -> str:
    if pool_ratio <= 1 / 3:
        return "low_0_0.33"
    if pool_ratio <= 2 / 3:
        return "mid_0.34_0.67"
    return "high_0.68_1.00"


def add_strata_columns(df: pd.DataFrame, parameter_bins: int) -> pd.DataFrame:
    """Attach reproducible structural strata without changing source columns."""
    result = df.copy()
    result["id"] = result["id"].astype(str).str.zfill(4)
    result["depth"] = result["depth"].astype(int)
    result["parameter_count"] = result["parameter_count"].astype(int)
    result["pool_count"] = result["pools"].map(parse_pool_count)
    result["pool_ratio"] = result["pool_count"] / result["depth"]
    result["depth_bin"] = result["depth"].map(depth_category)
    result["pool_ratio_bin"] = result["pool_ratio"].map(pool_ratio_category)

    # Quantiles are calculated within each channel pattern.  A global upper
    # parameter bin can contain no uniform models, whereas this validation
    # study needs every pattern to cover its own small-to-large range.
    result["parameter_bin"] = ""
    labels = [f"Q{index + 1}" for index in range(parameter_bins)]
    for pattern in PATTERN_ORDER:
        mask = result["pattern"] == pattern
        parameter_rank = result.loc[mask, "parameter_count"].rank(method="first")
        result.loc[mask, "parameter_bin"] = pd.qcut(
            parameter_rank,
            q=parameter_bins,
            labels=labels,
        ).astype(str)
    return result


def select_sample(df: pd.DataFrame, parameter_bins: int, seed: int) -> pd.DataFrame:
    """Select one model for every pattern-by-parameter-bin stratum."""
    rng = np.random.default_rng(seed)
    candidates = df.copy()
    candidates["tie_breaker"] = rng.random(len(candidates))
    depth_counts: dict[str, int] = {}
    pool_ratio_counts: dict[str, int] = {}
    exact_depth_counts: dict[int, int] = {}
    selected_rows: list[pd.Series] = []

    for parameter_bin in [f"Q{index + 1}" for index in range(parameter_bins)]:
        for pattern in PATTERN_ORDER:
            pool = candidates.loc[
                (candidates["pattern"] == pattern)
                & (candidates["parameter_bin"] == parameter_bin)
            ].copy()
            if pool.empty:
                raise ValueError(
                    f"No architecture is available for pattern={pattern}, "
                    f"parameter_bin={parameter_bin}."
                )

            # Pattern and parameter bin are hard strata.  The score distributes
            # the remaining dimensions globally across the selected sample.
            pool["coverage_score"] = (
                8 * pool["depth_bin"].map(depth_counts).fillna(0)
                + 8 * pool["pool_ratio_bin"].map(pool_ratio_counts).fillna(0)
                + 2 * pool["depth"].map(exact_depth_counts).fillna(0)
            )
            chosen = pool.sort_values(
                ["coverage_score", "tie_breaker", "id"],
                kind="stable",
            ).iloc[0]
            selected_rows.append(chosen)
            depth_counts[chosen["depth_bin"]] = depth_counts.get(chosen["depth_bin"], 0) + 1
            pool_ratio_counts[chosen["pool_ratio_bin"]] = (
                pool_ratio_counts.get(chosen["pool_ratio_bin"], 0) + 1
            )
            exact_depth_counts[int(chosen["depth"])] = exact_depth_counts.get(
                int(chosen["depth"]), 0
            ) + 1

    selected = pd.DataFrame(selected_rows).drop(columns=["tie_breaker", "coverage_score"])
    selected.insert(0, "selection_order", range(1, len(selected) + 1))
    selected.insert(1, "selection_reason", "weight_state_validation_stratified")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-csv", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("measurements/validation_weight_state/validation_sample_25.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("measurements/validation_weight_state/validation_sample_summary.json"),
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--min-model-id", type=int, default=1)
    parser.add_argument("--max-model-id", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.count % len(PATTERN_ORDER) != 0:
        raise ValueError(f"--count must be a positive multiple of {len(PATTERN_ORDER)}.")
    if args.min_model_id <= 0 or args.max_model_id < args.min_model_id:
        raise ValueError("Model ID bounds must satisfy 0 < min <= max.")
    if not args.structure_csv.is_file():
        raise FileNotFoundError(f"Structure CSV not found: {args.structure_csv}")

    source = pd.read_csv(args.structure_csv, encoding="utf-8-sig")
    required_columns = {"id", "depth", "pattern", "pools", "parameter_count"}
    missing = required_columns.difference(source.columns)
    if missing:
        raise ValueError(f"Structure CSV is missing required columns: {sorted(missing)}")

    numeric_ids = pd.to_numeric(source["id"], errors="coerce")
    source = source.loc[numeric_ids.between(args.min_model_id, args.max_model_id)].copy()
    source = source.loc[source["pattern"].isin(PATTERN_ORDER)].copy()
    if source.empty:
        raise ValueError("No source models remain after applying the ID bounds.")

    bins_per_pattern = args.count // len(PATTERN_ORDER)
    stratified = add_strata_columns(source, parameter_bins=bins_per_pattern)
    selected = select_sample(stratified, parameter_bins=bins_per_pattern, seed=args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "source_model_id_range": [args.min_model_id, args.max_model_id],
        "source_models": int(len(stratified)),
        "selected_models": int(len(selected)),
        "selection_method": "pattern x within-pattern parameter-count quantile, with depth and pool-ratio coverage balancing",
        "parameter_bin_scope": "within each channel pattern",
        "seed": args.seed,
        "pattern_counts": selected["pattern"].value_counts().sort_index().to_dict(),
        "parameter_bin_counts": selected["parameter_bin"].value_counts().sort_index().to_dict(),
        "depth_counts": selected["depth"].value_counts().sort_index().to_dict(),
        "pool_count_counts": selected["pool_count"].value_counts().sort_index().to_dict(),
        "pool_ratio_bin_counts": selected["pool_ratio_bin"].value_counts().sort_index().to_dict(),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Selected {len(selected)} models from {len(stratified)} source architectures.")
    print(f"Sample CSV: {args.output_csv}")
    print(f"Summary: {args.summary_json}")
    print("Pattern counts:", summary["pattern_counts"])
    print("Parameter-bin counts:", summary["parameter_bin_counts"])
    print("Depth counts:", summary["depth_counts"])
    print("Pool-count counts:", summary["pool_count_counts"])


if __name__ == "__main__":
    main()
