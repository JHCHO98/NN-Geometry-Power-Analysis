"""Prepare a second-round Pareto table with measured candidates as fixed anchors.

The 50 candidates measured as models 0501–0550 remain in the full Pareto
comparison, but their predictions are replaced with ground truth.  The output
is intended for ``measurements/search_2nd/`` and is never written over the
first-round prediction table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("measurements/search_2nd/candidate_predictions_raw.csv"),
        help="Second-round predictions for all 50,000 candidates.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("measurements/search/candidate_onnx_metadata.csv"),
        help="Mapping between candidate_id and measured model IDs 0501–0550.",
    )
    parser.add_argument(
        "--energy-summary",
        type=Path,
        default=Path("measurements/processed/energy_summary.csv"),
    )
    parser.add_argument(
        "--accuracy-results",
        type=Path,
        default=Path("measurements/ml/candidate_accuracy_50_results.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("measurements/search_2nd/candidate_predictions.csv"),
    )
    parser.add_argument(
        "--known-output-csv",
        type=Path,
        default=Path("measurements/search_2nd/known_measured_candidates.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.predictions, args.metadata, args.energy_summary, args.accuracy_results):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig")
    metadata = pd.read_csv(args.metadata, encoding="utf-8-sig")
    energy = pd.read_csv(args.energy_summary, encoding="utf-8-sig")
    accuracy = pd.read_csv(args.accuracy_results, encoding="utf-8-sig")
    if not predictions["candidate_id"].is_unique:
        raise ValueError("Prediction candidate IDs must be unique.")
    if not metadata["candidate_id"].is_unique:
        raise ValueError("Candidate metadata IDs must be unique.")

    metadata = metadata[["candidate_id", "id"]].copy()
    metadata["model_id"] = metadata["id"].astype(str).str.zfill(4)
    energy["model_id"] = energy["model_id"].astype(str).str.zfill(4)
    accuracy["model_id"] = accuracy["id"].astype(str).str.zfill(4)
    accuracy_column = "best_accuracy" if "best_accuracy" in accuracy.columns else "final_accuracy"

    known = metadata.merge(
        energy[["model_id", "mean_net_energy_per_inference_j", "mean_average_latency_ms"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    ).merge(
        accuracy[["model_id", accuracy_column]],
        on="model_id",
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "mean_net_energy_per_inference_j": "actual_energy_j",
            "mean_average_latency_ms": "actual_latency_ms",
            accuracy_column: "actual_accuracy_percent",
        }
    )
    required_actual = ["actual_energy_j", "actual_latency_ms", "actual_accuracy_percent"]
    incomplete = known[known[required_actual].isna().any(axis=1)]
    if not incomplete.empty:
        raise ValueError(
            "Measured candidate metadata lacks energy, latency, or accuracy for: "
            f"{incomplete['candidate_id'].tolist()}"
        )

    output = predictions.copy()
    lookup = known.set_index("candidate_id")
    output["is_measured_candidate"] = output["candidate_id"].isin(lookup.index)
    output["pareto_value_source"] = "predicted"
    output.loc[output["is_measured_candidate"], "pareto_value_source"] = "measured"
    output["measured_model_id"] = output["candidate_id"].map(lookup["model_id"])

    value_columns = {
        "predicted_accuracy_percent": "actual_accuracy_percent",
        "ensemble_accuracy_mean_percent": "actual_accuracy_percent",
        "predicted_energy_j": "actual_energy_j",
        "ensemble_energy_mean_j": "actual_energy_j",
        "predicted_latency_ms": "actual_latency_ms",
        "ensemble_latency_mean_ms": "actual_latency_ms",
    }
    measured_mask = output["is_measured_candidate"]
    for prediction_column, actual_column in value_columns.items():
        if prediction_column in output.columns:
            output.loc[measured_mask, prediction_column] = output.loc[measured_mask, "candidate_id"].map(
                lookup[actual_column]
            )
    for column in output.columns:
        if column.endswith("uncertainty_std_j") or column.endswith("uncertainty_std_ms") or column.endswith("uncertainty_std_percent"):
            output.loc[measured_mask, column] = 0.0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.known_output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")
    known.to_csv(args.known_output_csv, index=False, encoding="utf-8")
    print(
        f"Wrote {args.output_csv} with {len(output):,} candidates and "
        f"{int(measured_mask.sum())} measured Pareto anchors."
    )


if __name__ == "__main__":
    main()
