"""Run reproducible ONNX CPU-inference or idle-baseline benchmark trials.

HWiNFO should log CPU Package Power continuously while this script runs. The
CSV written by this script contains timezone-aware start/end timestamps and the
exact number of inferences, so its rows can later be matched to HWiNFO logs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import psutil


RESULT_FIELDS = [
    "mode",
    "model_id",
    "model_path",
    "trial",
    "warmup_count",
    "target_duration_sec",
    "actual_duration_sec",
    "inference_count",
    "average_latency_ms",
    "throughput_inferences_per_sec",
    "input_seed",
    "intra_op_threads",
    "cpu_core_requested",
    "cpu_core_applied",
    "high_priority_requested",
    "high_priority_applied",
    "started_at_local",
    "finished_at_local",
]


def append_result(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def configure_process(cpu_core: int | None, high_priority: bool) -> tuple[str, bool]:
    """Apply optional CPU affinity and Windows high priority to this process."""
    process = psutil.Process(os.getpid())
    applied_core = ""
    priority_applied = False

    if cpu_core is not None:
        available_cores = process.cpu_affinity()
        if cpu_core not in available_cores:
            raise ValueError(
                f"CPU core {cpu_core} is unavailable. Available cores: {available_cores}"
            )
        process.cpu_affinity([cpu_core])
        applied_core = str(cpu_core)

    if high_priority:
        if os.name != "nt":
            raise RuntimeError("--high-priority is currently supported only on Windows.")
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        priority_applied = True

    return applied_core, priority_applied


def resolve_model_path(model_path: Path | None, model_id: str | None, model_dir: Path) -> Path:
    if model_path is not None:
        return model_path
    if model_id is None:
        raise ValueError("Inference mode requires --model-path or --model-id.")
    try:
        numeric_id = int(model_id)
    except ValueError:
        filename = f"{model_id}.onnx"
    else:
        filename = f"{numeric_id:04d}.onnx"
    return model_dir / filename


def make_session(model_path: Path, intra_op_threads: int) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model was not found: {model_path}")
    if intra_op_threads <= 0:
        raise ValueError("intra_op_threads must be positive.")

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def make_input(session: ort.InferenceSession, input_seed: int) -> tuple[str, np.ndarray]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected exactly one model input, found {len(inputs)}.")
    input_info = inputs[0]
    shape: list[int] = []
    for index, dimension in enumerate(input_info.shape):
        if isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        elif index == 0:
            shape.append(1)  # dynamic batch dimension
        else:
            raise ValueError(f"Unsupported dynamic input shape: {input_info.shape}")
    if input_info.type != "tensor(float)":
        raise ValueError(f"Expected float32 model input, found {input_info.type}.")

    rng = np.random.default_rng(input_seed)
    return input_info.name, rng.random(shape, dtype=np.float32)


def run_inference_trial(
    session: ort.InferenceSession,
    input_name: str,
    input_array: np.ndarray,
    warmup_count: int,
    duration_sec: float,
) -> tuple[datetime, datetime, float, int]:
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive.")

    for _ in range(warmup_count):
        session.run(None, {input_name: input_array})

    print("MEASUREMENT_START")
    started_at = datetime.now().astimezone()
    start = time.perf_counter()
    inference_count = 0
    while time.perf_counter() - start < duration_sec:
        session.run(None, {input_name: input_array})
        inference_count += 1
    actual_duration = time.perf_counter() - start
    finished_at = datetime.now().astimezone()
    print("MEASUREMENT_END")
    return started_at, finished_at, actual_duration, inference_count


def run_idle_trial(duration_sec: float) -> tuple[datetime, datetime, float, int]:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive.")

    print("MEASUREMENT_START (IDLE)")
    started_at = datetime.now().astimezone()
    start = time.perf_counter()
    time.sleep(duration_sec)
    actual_duration = time.perf_counter() - start
    finished_at = datetime.now().astimezone()
    print("MEASUREMENT_END (IDLE)")
    return started_at, finished_at, actual_duration, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "idle"), default="inference")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-path", type=Path)
    model_group.add_argument("--model-id")
    parser.add_argument("--model-dir", type=Path, default=Path("model_onnx"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-count", type=int, default=200)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--cooldown-sec", type=float, default=60.0)
    parser.add_argument("--ready-wait-sec", type=float, default=5.0)
    parser.add_argument("--input-seed", type=int, default=20260824)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--cpu-core", type=int)
    parser.add_argument("--high-priority", action="store_true")
    parser.add_argument("--result-csv", type=Path, default=Path("pilot_benchmark_runs.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive.")
    if args.cooldown_sec < 0 or args.ready_wait_sec < 0:
        raise ValueError("cooldown_sec and ready_wait_sec must be non-negative.")

    cpu_core_applied, high_priority_applied = configure_process(args.cpu_core, args.high_priority)
    model_path: Path | None = None
    session: ort.InferenceSession | None = None
    input_name = ""
    input_array: np.ndarray | None = None

    if args.mode == "inference":
        model_path = resolve_model_path(args.model_path, args.model_id, args.model_dir)
        session = make_session(model_path, args.intra_op_threads)
        input_name, input_array = make_input(session, args.input_seed)

    print(f"Ready. Start or confirm HWiNFO logging; measurement starts in {args.ready_wait_sec:g} seconds.")
    time.sleep(args.ready_wait_sec)

    for trial in range(1, args.trials + 1):
        print(f"Trial {trial}/{args.trials}")
        if args.mode == "inference":
            assert session is not None and input_array is not None
            started_at, finished_at, actual_duration, inference_count = run_inference_trial(
                session,
                input_name,
                input_array,
                args.warmup_count,
                args.duration_sec,
            )
        else:
            started_at, finished_at, actual_duration, inference_count = run_idle_trial(args.duration_sec)

        average_latency_ms = (
            actual_duration / inference_count * 1_000 if inference_count else ""
        )
        throughput = inference_count / actual_duration if inference_count else ""
        append_result(
            args.result_csv,
            {
                "mode": args.mode,
                "model_id": args.model_id or "",
                "model_path": str(model_path) if model_path else "",
                "trial": trial,
                "warmup_count": args.warmup_count if args.mode == "inference" else 0,
                "target_duration_sec": args.duration_sec,
                "actual_duration_sec": f"{actual_duration:.6f}",
                "inference_count": inference_count,
                "average_latency_ms": average_latency_ms,
                "throughput_inferences_per_sec": throughput,
                "input_seed": args.input_seed if args.mode == "inference" else "",
                "intra_op_threads": args.intra_op_threads if args.mode == "inference" else "",
                "cpu_core_requested": args.cpu_core if args.cpu_core is not None else "",
                "cpu_core_applied": cpu_core_applied,
                "high_priority_requested": args.high_priority,
                "high_priority_applied": high_priority_applied,
                "started_at_local": started_at.isoformat(),
                "finished_at_local": finished_at.isoformat(),
            },
        )
        print(
            f"Recorded trial {trial}: {inference_count} inferences in "
            f"{actual_duration:.3f} s."
        )
        if trial < args.trials and args.cooldown_sec:
            print(f"Cooldown: {args.cooldown_sec:g} seconds.")
            time.sleep(args.cooldown_sec)


if __name__ == "__main__":
    main()
