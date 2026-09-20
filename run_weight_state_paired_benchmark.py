"""Benchmark any paired ONNX states in a balanced ABBA/BAAB order.

Start HWiNFO logging before this program. Each pair is measured in alternating
ABBA/BAAB blocks, limiting time-drift bias in the within-pair comparison. The
benchmark CSV carries all pairing metadata; the accompanying HWiNFO CSV is one
continuous log for the same session.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import onnxruntime as ort

from benchmark_onnx import (
    append_result,
    configure_process,
    make_input,
    make_session,
    run_idle_trial,
    run_inference_trial,
)


REQUIRED_FIELDS = {"pair_id", "structure_id", "weight_state", "onnx_path"}
DEFAULT_PROTOCOL_ID = "paired_abba_v1"


@dataclass(frozen=True)
class ModelState:
    pair_id: str
    structure_id: str
    weight_state: str
    onnx_path: Path
    training_run_id: str


def read_manifest(
    path: Path, state_a_name: str, state_b_name: str
) -> dict[str, dict[str, ModelState]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain {sorted(REQUIRED_FIELDS)}: {path}")
        pairs: dict[str, dict[str, ModelState]] = {}
        for row in reader:
            state = ModelState(
                pair_id=row["pair_id"],
                structure_id=row["structure_id"].zfill(4),
                weight_state=row["weight_state"],
                onnx_path=Path(row["onnx_path"]),
                training_run_id=row.get("training_run_id", ""),
            )
            if not state.pair_id or not state.weight_state or not str(state.onnx_path):
                raise ValueError(f"Invalid manifest row: {row}")
            if state.weight_state in pairs.setdefault(state.pair_id, {}):
                raise ValueError(f"Duplicate state '{state.weight_state}' in {state.pair_id}")
            pairs[state.pair_id][state.weight_state] = state
    for pair_id, states in pairs.items():
        expected = {state_a_name, state_b_name}
        if set(states) != expected:
            raise ValueError(f"{pair_id} needs exactly states {sorted(expected)}, found {sorted(states)}")
    return pairs


def recorded_keys(path: Path, session_id: str) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {
            (row.get("pair_id", ""), row.get("weight_state", ""), row.get("pair_trial", ""))
            for row in rows
            if row.get("benchmark_session_id", "") == session_id and row.get("mode") == "inference"
        }


def append_idle(
    args: argparse.Namespace,
    cpu_core_applied: str,
    priority_applied: bool,
    run_order: int,
    idle_role: str,
) -> None:
    started_at, finished_at, actual_duration, inference_count = run_idle_trial(args.idle_duration_sec)
    append_result(
        args.result_csv,
        {
            "mode": "idle",
            "model_id": "",
            "model_path": "",
            "trial": "",
            "warmup_count": 0,
            "target_duration_sec": args.idle_duration_sec,
            "actual_duration_sec": f"{actual_duration:.6f}",
            "inference_count": inference_count,
            "average_latency_ms": "",
            "throughput_inferences_per_sec": "",
            "input_seed": "",
            "intra_op_threads": "",
            "cpu_core_requested": args.cpu_core if args.cpu_core is not None else "",
            "cpu_core_applied": cpu_core_applied,
            "high_priority_requested": args.high_priority,
            "high_priority_applied": priority_applied,
            "started_at_local": started_at.isoformat(),
            "finished_at_local": finished_at.isoformat(),
            "benchmark_session_id": args.benchmark_session_id,
            "pair_id": "",
            "structure_id": "",
            "weight_state": "",
            "training_run_id": "",
            "protocol_id": args.protocol_id,
            "run_order": run_order,
            "pair_trial": "",
            "idle_role": idle_role,
        },
    )


def run_state(
    args: argparse.Namespace,
    state: ModelState,
    pair_trial: int,
    run_order: int,
    cpu_core_applied: str,
    priority_applied: bool,
) -> None:
    session: ort.InferenceSession = make_session(state.onnx_path, args.intra_op_threads)
    input_name, input_array = make_input(session, args.input_seed)
    started_at, finished_at, actual_duration, inference_count = run_inference_trial(
        session, input_name, input_array, args.warmup_count, args.duration_sec
    )
    append_result(
        args.result_csv,
        {
            "mode": "inference",
            "model_id": state.structure_id,
            "model_path": str(state.onnx_path),
            "trial": pair_trial,
            "warmup_count": args.warmup_count,
            "target_duration_sec": args.duration_sec,
            "actual_duration_sec": f"{actual_duration:.6f}",
            "inference_count": inference_count,
            "average_latency_ms": actual_duration / inference_count * 1_000,
            "throughput_inferences_per_sec": inference_count / actual_duration,
            "input_seed": args.input_seed,
            "intra_op_threads": args.intra_op_threads,
            "cpu_core_requested": args.cpu_core if args.cpu_core is not None else "",
            "cpu_core_applied": cpu_core_applied,
            "high_priority_requested": args.high_priority,
            "high_priority_applied": priority_applied,
            "started_at_local": started_at.isoformat(),
            "finished_at_local": finished_at.isoformat(),
            "benchmark_session_id": args.benchmark_session_id,
            "pair_id": state.pair_id,
            "structure_id": state.structure_id,
            "weight_state": state.weight_state,
            "training_run_id": state.training_run_id,
            "protocol_id": args.protocol_id,
            "run_order": run_order,
            "pair_trial": pair_trial,
            "idle_role": "",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("measurements/validation_weight_state/pair_manifest.csv"),
    )
    parser.add_argument("--benchmark-session-id", required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=200)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--cooldown-sec", type=float, default=60.0)
    parser.add_argument("--idle-duration-sec", type=float, default=60.0)
    parser.add_argument("--idle-every-pairs", type=int, default=5)
    parser.add_argument("--initial-idle-trials", type=int, default=3)
    parser.add_argument("--final-idle-trials", type=int, default=3)
    parser.add_argument("--ready-wait-sec", type=float, default=5.0)
    parser.add_argument("--input-seed", type=int, default=20260824)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--cpu-core", type=int)
    parser.add_argument("--high-priority", action="store_true")
    parser.add_argument(
        "--state-a",
        default="untrained_seed_reconstructed",
        help="First state in the first ABBA block.",
    )
    parser.add_argument(
        "--state-b",
        default="trained_best_accuracy",
        help="Second state in the first ABBA block.",
    )
    parser.add_argument(
        "--abba-blocks",
        type=int,
        default=1,
        help="Number of alternating ABBA/BAAB blocks. Each block records two trials per state.",
    )
    parser.add_argument(
        "--idle-between-abba-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert an idle trial between repeated ABBA blocks for the same pair.",
    )
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--start-pair", type=int, default=1)
    parser.add_argument("--end-pair", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_count < 0 or args.duration_sec <= 0 or args.idle_duration_sec <= 0:
        raise ValueError("Warmup must be non-negative; inference and idle durations must be positive.")
    if args.cooldown_sec < 0 or args.ready_wait_sec < 0 or args.idle_every_pairs < 0:
        raise ValueError("Cooldown, ready wait, and idle interval must be non-negative.")
    if args.abba_blocks <= 0 or args.state_a == args.state_b:
        raise ValueError("abba-blocks must be positive and --state-a/--state-b must differ.")

    pairs = read_manifest(args.pair_manifest, args.state_a, args.state_b)
    pair_ids = sorted(pairs)
    selected = pair_ids[args.start_pair - 1 : args.end_pair]
    if not selected:
        raise ValueError("No pairs selected; check --start-pair and --end-pair.")
    missing_paths = [str(state.onnx_path) for pair_id in selected for state in pairs[pair_id].values() if not state.onnx_path.is_file()]
    if missing_paths:
        raise FileNotFoundError("ONNX files missing: " + ", ".join(missing_paths))
    if args.result_csv.exists() and not args.resume:
        raise FileExistsError(f"Result CSV already exists: {args.result_csv}. Use --resume to append only missing ABBA trials.")

    existing = recorded_keys(args.result_csv, args.benchmark_session_id) if args.resume else set()
    cpu_core_applied, priority_applied = configure_process(args.cpu_core, args.high_priority)
    print(f"Ready for {len(selected)} pairs. Start or confirm HWiNFO logging; measurement starts in {args.ready_wait_sec:g} seconds.")
    time.sleep(args.ready_wait_sec)
    run_order = 0
    for _ in range(args.initial_idle_trials):
        run_order += 1
        append_idle(args, cpu_core_applied, priority_applied, run_order, "pre_session_idle")

    for pair_index, pair_id in enumerate(selected, start=1):
        states = pairs[pair_id]
        print(f"Pair {pair_index}/{len(selected)}: {pair_id} ({args.abba_blocks} balanced ABBA block(s))")
        for block_index in range(1, args.abba_blocks + 1):
            first_trial = 2 * block_index - 1
            second_trial = 2 * block_index
            state_a = states[args.state_a]
            state_b = states[args.state_b]
            schedule = (
                ((state_a, first_trial), (state_b, first_trial), (state_b, second_trial), (state_a, second_trial))
                if block_index % 2
                else ((state_b, first_trial), (state_a, first_trial), (state_a, second_trial), (state_b, second_trial))
            )
            for state, pair_trial in schedule:
                key = (state.pair_id, state.weight_state, str(pair_trial))
                if key in existing:
                    print(f"Skipping recorded {key}.")
                    continue
                run_order += 1
                run_state(args, state, pair_trial, run_order, cpu_core_applied, priority_applied)
                if args.cooldown_sec:
                    print(f"Cooldown: {args.cooldown_sec:g} seconds.")
                    time.sleep(args.cooldown_sec)
            if args.idle_between_abba_blocks and block_index < args.abba_blocks:
                run_order += 1
                append_idle(args, cpu_core_applied, priority_applied, run_order, "between_abba_blocks_idle")
        if args.idle_every_pairs and pair_index % args.idle_every_pairs == 0:
            run_order += 1
            append_idle(args, cpu_core_applied, priority_applied, run_order, "between_pair_block_idle")

    for _ in range(args.final_idle_trials):
        run_order += 1
        append_idle(args, cpu_core_applied, priority_applied, run_order, "post_session_idle")
    print(f"Completed paired benchmark session: {args.benchmark_session_id}")


if __name__ == "__main__":
    main()
