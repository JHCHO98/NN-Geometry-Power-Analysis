"""Build the trained/untrained ONNX pairing manifest for weight-state validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "pair_id",
    "structure_id",
    "weight_state",
    "training_run_id",
    "onnx_path",
    "checkpoint_path",
    "expected_best_accuracy",
    "artifact_available",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--untrained-metadata",
        type=Path,
        default=Path("measurements/validation_weight_state/untrained_onnx_metadata.csv"),
    )
    parser.add_argument(
        "--trained-results",
        type=Path,
        default=Path("measurements/validation_weight_state/trained_accuracy_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("measurements/validation_weight_state/pair_manifest.csv"),
    )
    parser.add_argument("--training-run-id", default="cifar10_trained_validation_v1")
    parser.add_argument(
        "--include-headline",
        action="store_true",
        help="Add 0551 and C002793/0530 to the untrained-versus-trained validation set.",
    )
    parser.add_argument(
        "--headline-models",
        type=Path,
        default=Path("measurements/headline_comparison/headline_models.csv"),
    )
    parser.add_argument("--require-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    untrained = {row["id"].zfill(4): row for row in read_rows(args.untrained_metadata) if row.get("id")}
    trained = {row["id"].zfill(4): row for row in read_rows(args.trained_results) if row.get("id")}
    missing = sorted(set(untrained).symmetric_difference(trained))
    if missing:
        raise ValueError(f"Each structure needs both states; unmatched IDs: {', '.join(missing)}")

    rows: list[dict[str, str]] = []
    unavailable: list[str] = []
    for structure_id in sorted(untrained):
        pair_id = f"wsval_{structure_id}"
        states = (
            ("untrained_seed_reconstructed", untrained[structure_id]["onnx_path"], "", ""),
            (
                "trained_best_accuracy",
                trained[structure_id].get("trained_onnx_path", ""),
                trained[structure_id].get("trained_checkpoint_path", ""),
                trained[structure_id].get("best_accuracy", ""),
            ),
        )
        for weight_state, onnx_path, checkpoint_path, accuracy in states:
            available = Path(onnx_path).is_file()
            if not available:
                unavailable.append(f"{structure_id}:{weight_state}")
            rows.append(
                {
                    "pair_id": pair_id,
                    "structure_id": structure_id,
                    "weight_state": weight_state,
                    "training_run_id": args.training_run_id if weight_state.startswith("trained") else "",
                    "onnx_path": onnx_path,
                    "checkpoint_path": checkpoint_path,
                    "expected_best_accuracy": accuracy,
                    "artifact_available": str(available).lower(),
                }
            )
    if args.include_headline:
        if not args.headline_models.is_file():
            raise FileNotFoundError(
                f"Create the headline registry first: {args.headline_models}"
            )
        for model in read_rows(args.headline_models):
            structure_id = model["structure_id"].zfill(4)
            if structure_id in untrained:
                raise ValueError(f"Headline model duplicates a validation structure: {structure_id}")
            pair_id = f"wsval_{structure_id}"
            states = (
                ("untrained_seed_reconstructed", model["untrained_onnx_path"], "", ""),
                (
                    "trained_best_accuracy",
                    model["trained_onnx_path"],
                    model.get("trained_checkpoint_path", ""),
                    model.get("actual_accuracy_percent", ""),
                ),
            )
            for weight_state, onnx_path, checkpoint_path, accuracy in states:
                available = Path(onnx_path).is_file()
                if not available:
                    unavailable.append(f"{structure_id}:{weight_state}")
                rows.append(
                    {
                        "pair_id": pair_id,
                        "structure_id": structure_id,
                        "weight_state": weight_state,
                        "training_run_id": args.training_run_id if weight_state.startswith("trained") else "",
                        "onnx_path": onnx_path,
                        "checkpoint_path": checkpoint_path,
                        "expected_best_accuracy": accuracy,
                        "artifact_available": str(available).lower(),
                    }
                )
    if args.require_artifacts and unavailable:
        raise FileNotFoundError("Missing ONNX artifacts: " + ", ".join(unavailable))
    write_rows(args.output, rows)
    print(f"Wrote {args.output} with {len(rows) // 2} pairs / {len(rows)} state rows.")
    if unavailable:
        print(f"ONNX artifacts not currently local ({len(unavailable)}): {', '.join(unavailable[:5])}" + (" ..." if len(unavailable) > 5 else ""))


if __name__ == "__main__":
    main()
