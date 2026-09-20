"""Prepare manifests and artifact locations for the final 0551-versus-0530 comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_FIELDS = [
    "role",
    "structure_id",
    "candidate_id",
    "display_name",
    "parameter_count",
    "actual_accuracy_percent",
    "untrained_onnx_path",
    "trained_onnx_path",
    "trained_checkpoint_path",
    "trained_onnx_available",
]
MANIFEST_FIELDS = [
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


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("measurements/reference_models/reference_accuracy_results.csv"),
    )
    parser.add_argument(
        "--candidate-results",
        type=Path,
        default=Path("measurements/evaluation/candidate_evaluation_results.csv"),
    )
    parser.add_argument("--candidate-id", default="C002793")
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/headline_comparison"))
    parser.add_argument("--training-run-id", default="headline_cifar10_v1")
    parser.add_argument(
        "--headline-training-results",
        type=Path,
        default=Path("measurements/headline_comparison/headline_training_results.csv"),
        help="Optional result CSV from train_candidate_accuracy.py; overrides older recorded accuracies.",
    )
    parser.add_argument("--require-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_rows = read_rows(args.reference_results)
    if len(reference_rows) != 1:
        raise ValueError("Reference results must contain exactly one model.")
    reference = reference_rows[0]
    candidate = next(
        (row for row in read_rows(args.candidate_results) if row.get("candidate_id") == args.candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Candidate ID was not found: {args.candidate_id}")

    output_dir = args.output_dir
    trained_dir = output_dir / "onnx"
    checkpoint_dir = output_dir / "checkpoints"
    models = [
        {
            "role": "reference",
            "structure_id": str(reference["id"]).zfill(4),
            "candidate_id": reference.get("candidate_id", ""),
            "display_name": "VGG-inspired reference 0551",
            "parameter_count": reference["parameter_count"],
            "actual_accuracy_percent": reference.get("best_accuracy", ""),
            "untrained_onnx_path": reference["onnx_path"],
        },
        {
            "role": "pareto",
            "structure_id": str(candidate["model_id"]).zfill(4),
            "candidate_id": candidate["candidate_id"],
            "display_name": f"Final Pareto candidate {candidate['candidate_id']}",
            "parameter_count": candidate["parameter_count"],
            "actual_accuracy_percent": candidate.get("actual_accuracy_percent", ""),
            "untrained_onnx_path": f"model_onnx/{str(candidate['model_id']).zfill(4)}.onnx",
        },
    ]
    for model in models:
        model_id = str(model["structure_id"])
        model["trained_onnx_path"] = str(trained_dir / f"{model_id}_trained_best.onnx")
        model["trained_checkpoint_path"] = str(checkpoint_dir / f"{model_id}_best.pt")
        model["trained_onnx_available"] = str(Path(str(model["trained_onnx_path"])).is_file()).lower()
    if args.headline_training_results.is_file():
        training_rows = {
            str(row["id"]).zfill(4): row
            for row in read_rows(args.headline_training_results)
            if row.get("id")
        }
        required_ids = {str(model["structure_id"]) for model in models}
        missing = sorted(required_ids - set(training_rows))
        if missing:
            raise ValueError(
                "Headline training results are incomplete; missing IDs: " + ", ".join(missing)
            )
        for model in models:
            training = training_rows[str(model["structure_id"])]
            model["actual_accuracy_percent"] = training.get("best_accuracy", "")
            model["trained_onnx_path"] = training.get("trained_onnx_path", model["trained_onnx_path"])
            model["trained_checkpoint_path"] = training.get(
                "trained_checkpoint_path", model["trained_checkpoint_path"]
            )
            model["trained_onnx_available"] = str(
                Path(str(model["trained_onnx_path"])).is_file()
            ).lower()
    if args.require_artifacts and not all(model["trained_onnx_available"] == "true" for model in models):
        raise FileNotFoundError("Both trained headline ONNX files must be present under " + str(trained_dir))

    manifest_rows = []
    for model in models:
        state = f"{model['role']}_{model['structure_id']}_trained"
        manifest_rows.append(
            {
                "pair_id": "headline_0551_vs_0530",
                "structure_id": model["structure_id"],
                "weight_state": state,
                "training_run_id": args.training_run_id,
                "onnx_path": model["trained_onnx_path"],
                "checkpoint_path": model["trained_checkpoint_path"],
                "expected_best_accuracy": model["actual_accuracy_percent"],
                "artifact_available": model["trained_onnx_available"],
            }
        )
    write_rows(output_dir / "headline_models.csv", MODEL_FIELDS, models)
    write_rows(output_dir / "headline_manifest.csv", MANIFEST_FIELDS, manifest_rows)
    print(f"Wrote headline registry and manifest to {output_dir}.")
    print("Expected trained ONNX files:")
    for model in models:
        print(f"  {model['trained_onnx_path']}")


if __name__ == "__main__":
    main()
