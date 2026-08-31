"""Match HWiNFO power logs to benchmark trials and calculate inference energy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TRIAL_FIELDS = [
    "run_row",
    "mode",
    "model_id",
    "trial",
    "started_at_local",
    "finished_at_local",
    "actual_duration_sec",
    "inference_count",
    "sample_count",
    "average_power_w",
    "gross_energy_j",
    "gross_energy_per_inference_j",
    "baseline_power_w",
    "baseline_source",
    "net_energy_j",
    "net_energy_per_inference_j",
    "average_latency_ms",
    "throughput_inferences_per_sec",
    "depth",
    "pattern",
    "growth_pattern",
    "channels",
    "pools",
    "parameter_count",
    "status",
]
SUMMARY_FIELDS = [
    "model_id",
    "valid_trials",
    "mean_net_energy_per_inference_j",
    "std_net_energy_per_inference_j",
    "cv_percent",
    "mean_gross_energy_per_inference_j",
    "mean_average_latency_ms",
    "mean_throughput_inferences_per_sec",
    "depth",
    "pattern",
    "growth_pattern",
    "channels",
    "pools",
    "parameter_count",
]


@dataclass
class Trial:
    run_row: int
    mode: str
    model_id: str
    trial: str
    started_at: datetime
    finished_at: datetime
    duration_sec: float
    inference_count: int
    average_latency_ms: float | None
    throughput: float | None


def canonical_model_id(value: str) -> str:
    try:
        return f"{int(value):04d}"
    except (TypeError, ValueError):
        return value


def read_csv_with_detected_encoding(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except (UnicodeError, pd.errors.ParserError) as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError(f"Could not read {path}. Attempts: {'; '.join(errors)}")


def find_column(columns: list[str], requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Column '{requested}' was not found. Available: {columns}")
        return requested

    normalized = {column.strip().casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return normalized[candidate.casefold()]
    raise ValueError(f"Could not identify a required column. Available: {columns}")


def parse_hwinfolog(
    path: Path,
    timezone: ZoneInfo,
    power_column: str | None,
    timestamp_column: str | None,
    date_column: str | None,
    time_column: str | None,
) -> pd.DataFrame:
    frame = read_csv_with_detected_encoding(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    columns = list(frame.columns)
    power_name = find_column(
        columns,
        power_column,
        ("CPU Package Power [W]", "CPU Package Power", "CPU Package Power (W)"),
    )

    if timestamp_column:
        timestamp_name = find_column(columns, timestamp_column, ())
        timestamp_text = frame[timestamp_name].astype(str)
    else:
        try:
            date_name = find_column(columns, date_column, ("Date",))
            time_name = find_column(columns, time_column, ("Time",))
        except ValueError:
            timestamp_name = find_column(columns, None, ("Timestamp", "Date Time", "Date/Time"))
            timestamp_text = frame[timestamp_name].astype(str)
        else:
            timestamp_text = frame[date_name].astype(str) + " " + frame[time_name].astype(str)

    timestamps = pd.to_datetime(timestamp_text, errors="coerce")
    if timestamps.isna().all():
        raise ValueError("Could not parse any HWiNFO timestamps. Specify timestamp columns explicitly.")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)

    power = pd.to_numeric(
        frame[power_name].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    result = pd.DataFrame({"timestamp": timestamps, "power_w": power}).dropna()
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if len(result) < 2:
        raise ValueError("HWiNFO log needs at least two valid timestamped power samples.")
    return result.reset_index(drop=True)


def parse_trials(path: Path) -> list[Trial]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"mode", "started_at_local", "finished_at_local", "actual_duration_sec", "inference_count"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Benchmark CSV is missing required fields: {sorted(required)}")

    trials: list[Trial] = []
    for run_row, row in enumerate(rows, start=1):
        try:
            trials.append(
                Trial(
                    run_row=run_row,
                    mode=row["mode"],
                    model_id=canonical_model_id(row.get("model_id", "")),
                    trial=row.get("trial", ""),
                    started_at=datetime.fromisoformat(row["started_at_local"]),
                    finished_at=datetime.fromisoformat(row["finished_at_local"]),
                    duration_sec=float(row["actual_duration_sec"]),
                    inference_count=int(row["inference_count"]),
                    average_latency_ms=float(row["average_latency_ms"]) if row.get("average_latency_ms") else None,
                    throughput=float(row["throughput_inferences_per_sec"]) if row.get("throughput_inferences_per_sec") else None,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark row {run_row}: {error}") from error
    return trials


def read_structures(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {canonical_model_id(row["id"]): row for row in rows if row.get("id")}


def integrate_trial(trace: pd.DataFrame, trial: Trial) -> tuple[int, float, float]:
    """Integrate power with linearly interpolated values at exact boundaries."""
    start_seconds = trial.started_at.timestamp()
    end_seconds = trial.finished_at.timestamp()
    times = trace["timestamp"].map(lambda value: value.timestamp()).to_numpy(dtype=float)
    powers = trace["power_w"].to_numpy(dtype=float)

    if times[0] > start_seconds or times[-1] < end_seconds:
        raise ValueError("HWiNFO log does not cover this trial's full time interval.")

    internal = (times > start_seconds) & (times < end_seconds)
    interval_times = np.concatenate(([start_seconds], times[internal], [end_seconds]))
    interval_powers = np.interp(interval_times, times, powers)
    gross_energy_j = float(np.trapezoid(interval_powers, interval_times))
    average_power_w = gross_energy_j / trial.duration_sec
    return int(internal.sum()), average_power_w, gross_energy_j


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_idle_blocks(
    rows: list[dict[str, object]], trial_lookup: dict[int, Trial]
) -> list[dict[str, object]]:
    """Combine each contiguous group of idle trials into one baseline block."""
    blocks: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def finish_block() -> None:
        if not current:
            return
        energy = sum(float(row["gross_energy_j"]) for row in current)
        duration = sum(float(row["actual_duration_sec"]) for row in current)
        first_trial = trial_lookup[int(current[0]["run_row"])]
        last_trial = trial_lookup[int(current[-1]["run_row"])]
        blocks.append(
            {
                "started_at": first_trial.started_at,
                "finished_at": last_trial.finished_at,
                "power_w": energy / duration,
            }
        )

    for row in rows:
        if row["mode"] == "idle" and row["status"] == "valid":
            current.append(row)
        else:
            finish_block()
            current = []
    finish_block()
    return blocks


def interpolated_baseline(
    trial: Trial, blocks: list[dict[str, object]]
) -> tuple[float, str] | None:
    """Use the closest idle blocks before/after a trial, interpolating if both exist."""
    before = [block for block in blocks if block["finished_at"] <= trial.started_at]
    after = [block for block in blocks if block["started_at"] >= trial.finished_at]
    if before and after:
        left = before[-1]
        right = after[0]
        left_time = left["finished_at"].timestamp()
        right_time = right["started_at"].timestamp()
        trial_time = (trial.started_at.timestamp() + trial.finished_at.timestamp()) / 2
        fraction = (trial_time - left_time) / (right_time - left_time)
        fraction = max(0.0, min(1.0, fraction))
        power = left["power_w"] + fraction * (right["power_w"] - left["power_w"])
        return power, "interpolated_adjacent_idle_blocks"
    if before:
        return before[-1]["power_w"], "previous_idle_block"
    if after:
        return after[0]["power_w"], "next_idle_block"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwinfolog", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, default=Path("pilot_benchmark_runs.csv"))
    parser.add_argument("--structure-csv", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/processed"))
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--power-column")
    parser.add_argument("--timestamp-column")
    parser.add_argument("--date-column")
    parser.add_argument("--time-column")
    parser.add_argument(
        "--baseline-power-w",
        type=float,
        help="Use a known idle baseline instead of idle rows in the benchmark CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timezone = ZoneInfo(args.timezone)
    trace = parse_hwinfolog(
        args.hwinfolog,
        timezone,
        args.power_column,
        args.timestamp_column,
        args.date_column,
        args.time_column,
    )
    trials = parse_trials(args.benchmark_csv)
    trial_lookup = {trial.run_row: trial for trial in trials}
    structures = read_structures(args.structure_csv)

    extracted: list[dict[str, object]] = []
    for trial in trials:
        row: dict[str, object] = {
            "run_row": trial.run_row,
            "mode": trial.mode,
            "model_id": trial.model_id,
            "trial": trial.trial,
            "started_at_local": trial.started_at.isoformat(),
            "finished_at_local": trial.finished_at.isoformat(),
            "actual_duration_sec": trial.duration_sec,
            "inference_count": trial.inference_count,
            "average_latency_ms": trial.average_latency_ms or "",
            "throughput_inferences_per_sec": trial.throughput or "",
            "sample_count": "",
            "average_power_w": "",
            "gross_energy_j": "",
            "gross_energy_per_inference_j": "",
            "baseline_power_w": "",
            "baseline_source": "",
            "net_energy_j": "",
            "net_energy_per_inference_j": "",
            "status": "",
        }
        structure = structures.get(trial.model_id, {})
        for field in ("depth", "pattern", "growth_pattern", "channels", "pools", "parameter_count"):
            row[field] = structure.get(field, "")
        try:
            sample_count, average_power, gross_energy = integrate_trial(trace, trial)
        except ValueError as error:
            row["status"] = f"excluded: {error}"
        else:
            row.update(
                {
                    "sample_count": sample_count,
                    "average_power_w": average_power,
                    "gross_energy_j": gross_energy,
                    "gross_energy_per_inference_j": (
                        gross_energy / trial.inference_count if trial.inference_count else ""
                    ),
                    "status": "valid",
                }
            )
        extracted.append(row)

    idle_blocks = build_idle_blocks(extracted, trial_lookup)
    used_baselines: list[float] = []

    for row in extracted:
        if row["mode"] != "inference" or row["status"] != "valid":
            continue
        if args.baseline_power_w is not None:
            baseline = (args.baseline_power_w, "provided_baseline_power")
        else:
            baseline = interpolated_baseline(trial_lookup[int(row["run_row"])], idle_blocks)
        if baseline is None:
            row["status"] = "valid_gross_only: no idle baseline"
            continue
        baseline_power, baseline_source = baseline
        used_baselines.append(baseline_power)
        net_energy = float(row["gross_energy_j"]) - baseline_power * float(row["actual_duration_sec"])
        row["baseline_power_w"] = baseline_power
        row["baseline_source"] = baseline_source
        row["net_energy_j"] = net_energy
        row["net_energy_per_inference_j"] = net_energy / int(row["inference_count"])

    summaries: list[dict[str, object]] = []
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in extracted:
        if row["mode"] == "inference" and row["status"] == "valid" and row["net_energy_per_inference_j"] != "":
            by_model.setdefault(str(row["model_id"]), []).append(row)
    for model_id, rows in sorted(by_model.items()):
        net_values = [float(row["net_energy_per_inference_j"]) for row in rows]
        gross_values = [float(row["gross_energy_per_inference_j"]) for row in rows]
        latency_values = [float(row["average_latency_ms"]) for row in rows if row["average_latency_ms"] != ""]
        throughput_values = [float(row["throughput_inferences_per_sec"]) for row in rows if row["throughput_inferences_per_sec"] != ""]
        net_mean = mean(net_values)
        net_std = stdev(net_values) if len(net_values) > 1 else 0.0
        exemplar = rows[0]
        summaries.append(
            {
                "model_id": model_id,
                "valid_trials": len(rows),
                "mean_net_energy_per_inference_j": net_mean,
                "std_net_energy_per_inference_j": net_std,
                "cv_percent": net_std / net_mean * 100 if net_mean else "",
                "mean_gross_energy_per_inference_j": mean(gross_values),
                "mean_average_latency_ms": mean(latency_values) if latency_values else "",
                "mean_throughput_inferences_per_sec": mean(throughput_values) if throughput_values else "",
                "depth": exemplar["depth"],
                "pattern": exemplar["pattern"],
                "growth_pattern": exemplar["growth_pattern"],
                "channels": exemplar["channels"],
                "pools": exemplar["pools"],
                "parameter_count": exemplar["parameter_count"],
            }
        )

    trial_output = args.output_dir / "energy_trials.csv"
    summary_output = args.output_dir / "energy_summary.csv"
    write_csv(trial_output, TRIAL_FIELDS, extracted)
    write_csv(summary_output, SUMMARY_FIELDS, summaries)
    print(f"Wrote {trial_output}")
    print(f"Wrote {summary_output}")
    if not used_baselines:
        print("No valid idle baseline was found: only gross energy is available.")
    else:
        print(f"Mean baseline power applied: {mean(used_baselines):.6f} W")


if __name__ == "__main__":
    main()
