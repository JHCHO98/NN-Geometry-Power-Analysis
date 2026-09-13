"""Compute resumable Zero-Cost proxy features directly in a structure CSV."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from FlexibleCNN import ModelConfig
from prepare_xgboost_dataset import parse_int_sequence
from zero_cost_proxies import compute_zero_cost_proxies


PROXY_NAMES = ("synflow_score", "grad_norm_score", "jacob_cov_score")


def config_from_row(row: pd.Series) -> ModelConfig:
    """Restore an executable model configuration from one candidate CSV row."""
    channels = parse_int_sequence(row["channels"])
    pools = parse_int_sequence(row["pools"])
    return ModelConfig(
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


def atomic_save(frame: pd.DataFrame, path: Path) -> None:
    """Replace the target only after a complete checkpoint CSV has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def score_one(payload: tuple[int, ModelConfig, str]) -> tuple[int, dict[str, float]]:
    """Worker entry point kept at module level for Windows multiprocessing."""
    index, config, device = payload
    if device == "cpu":
        torch.set_num_threads(1)
    return index, compute_zero_cost_proxies(config, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("measurements/search/candidates.csv"))
    parser.add_argument("--id-column", default="candidate_id", help="Unique row identifier, for example candidate_id or id.")
    parser.add_argument("--column-prefix", default="feature_", help="Prefix for proxy columns; use an empty string for accuracy results.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU processes (0: use up to 8 automatically; CUDA always uses 1).",
    )
    parser.add_argument("--save-every", type=int, default=500, help="Persist after this many newly scored candidates.")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many missing rows; useful for a smoke test.")
    parser.add_argument("--start", type=int, default=1, help="First candidate row to consider (1-based, inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last candidate row to consider (1-based, inclusive).")
    parser.add_argument("--force", action="store_true", help="Recalculate rows that already have proxy values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.save_every <= 0
        or args.workers < 0
        or args.start <= 0
        or (args.end is not None and args.end <= 0)
        or (args.limit is not None and args.limit <= 0)
    ):
        raise ValueError("--save-every, --limit, --start, and --end must be positive; --workers cannot be negative.")
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    workers = 1 if device == "cuda" else (args.workers or min(8, os.cpu_count() or 1))

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    required = {args.id_column, "seed", "depth", "pattern", "growth_pattern", "noise_ratio", "pools", "min_channels", "max_channels", "channels", "parameter_count"}
    missing = required.difference(frame.columns)
    if missing or frame.empty or not frame[args.id_column].is_unique:
        raise ValueError(f"Invalid candidate CSV; missing columns: {sorted(missing)}")
    proxy_columns = tuple(f"{args.column_prefix}{name}" for name in PROXY_NAMES)
    for column in proxy_columns:
        if column not in frame:
            frame[column] = np.nan

    end = args.end or len(frame)
    if args.start > end or end > len(frame):
        raise ValueError(f"Requested range {args.start}..{end} is outside 1..{len(frame)}.")
    range_indices = frame.index[args.start - 1 : end]
    complete = frame[list(proxy_columns)].notna().all(axis=1)
    pending = range_indices.to_list() if args.force else [index for index in range_indices if not complete.loc[index]]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"Using {device} with {workers} worker(s). "
        f"Range {args.start:,}..{end:,}; {complete.loc[range_indices].sum():,}/{len(range_indices):,} rows already have all proxy values."
    )
    if not pending:
        print("Nothing to calculate.")
        return

    started = time.perf_counter()
    executor = None
    if workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except (OSError, PermissionError) as error:
            print(f"Notice: could not start CPU workers ({error}). Falling back to one worker.")
            workers = 1
    try:
        for start in range(0, len(pending), args.save_every):
            batch_indices = pending[start : start + args.save_every]
            payloads = [(index, config_from_row(frame.loc[index]), device) for index in batch_indices]
            results = map(score_one, payloads) if executor is None else executor.map(score_one, payloads)
            for index, scores in results:
                frame.loc[index, list(proxy_columns)] = [
                    scores["synflow_score"], scores["grad_norm_score"], scores["jacob_cov_score"]
                ]
            done = start + len(batch_indices)
            atomic_save(frame, args.csv)
            elapsed = time.perf_counter() - started
            print(f"Saved {done:,}/{len(pending):,} new proxies ({elapsed / done:.2f}s/model).")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    print(f"Completed {len(pending):,} Zero-Cost proxy rows in {time.perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()
