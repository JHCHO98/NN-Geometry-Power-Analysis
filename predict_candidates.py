"""Predict energy, latency, and accuracy with bootstrap uncertainty for CNN candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = {
    "energy": "target_energy_j",
    "latency": "target_latency_ms",
    "accuracy": "target_accuracy_percent",
}


@dataclass
class ModelArtifact:
    name: str
    model: object
    preprocessor: object
    features: list[str]


def import_dependencies() -> dict[str, object]:
    try:
        import joblib
        from sklearn.base import clone
    except ImportError as error:
        raise RuntimeError(
            "Missing ML dependencies. Install them with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn xgboost joblib"
        ) from error
    return {"joblib": joblib, "clone": clone}


def load_artifact(path: Path, name: str, dependencies: dict[str, object]) -> ModelArtifact:
    content = dependencies["joblib"].load(path)
    required = {"model", "preprocessor", "features"}
    missing = required.difference(content)
    if missing:
        raise ValueError(f"{path} is missing artifact fields: {sorted(missing)}")
    return ModelArtifact(name, content["model"], content["preprocessor"], list(content["features"]))


def predict_ensemble(
    artifact: ModelArtifact,
    measured: pd.DataFrame,
    candidates: pd.DataFrame,
    ensemble_size: int,
    random_state: int,
    dependencies: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return saved-model prediction and bootstrap mean/standard deviation."""
    features = artifact.features
    use_log = artifact.name in ("energy", "latency")

    measured_sub = measured[measured[TARGET_COLUMNS[artifact.name]].notna()].copy()
    if measured_sub.empty:
        raise ValueError(f"No valid measured data found for target {artifact.name}.")

    for col in features:
        if col not in candidates.columns:
            candidates[col] = np.nan
        if col not in measured_sub.columns:
            measured_sub[col] = np.nan

    raw_pred = artifact.model.predict(artifact.preprocessor.transform(candidates[features]))
    point = np.exp(raw_pred) if use_log else raw_pred

    target_raw = measured_sub[TARGET_COLUMNS[artifact.name]].to_numpy(dtype=float)
    target = np.log(target_raw) if use_log else target_raw

    if use_log and not np.isfinite(target).all():
        raise ValueError(f"Measured {TARGET_COLUMNS[artifact.name]} must be positive and finite.")

    rng = np.random.default_rng(random_state)
    replica_predictions = np.empty((ensemble_size, len(candidates)), dtype=float)
    estimator_count = max(1, int(getattr(artifact.model, "best_iteration", 0)) + 1)

    for replica in range(ensemble_size):
        sampled = rng.integers(0, len(measured_sub), size=len(measured_sub))
        preprocessor = dependencies["clone"](artifact.preprocessor)
        train = preprocessor.fit_transform(measured_sub.iloc[sampled][features])
        model = dependencies["clone"](artifact.model)
        model.set_params(
            n_estimators=estimator_count,
            random_state=random_state + replica,
            early_stopping_rounds=None,
        )
        model.fit(train, target[sampled], verbose=False)
        sub_pred = model.predict(preprocessor.transform(candidates[features]))
        replica_predictions[replica] = np.exp(sub_pred) if use_log else sub_pred

    return point, replica_predictions.mean(axis=0), replica_predictions.std(axis=0, ddof=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("measurements/search/candidates.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("measurements/ml/model_dataset.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("measurements/ml"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/search/candidate_predictions.csv"))
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=20260911)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 2 or args.uncertainty_weight < 0:
        raise ValueError("--ensemble-size must be at least 2 and --uncertainty-weight must be non-negative.")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}. Use --overwrite to replace it.")

    dependencies = import_dependencies()
    candidates = pd.read_csv(args.candidates, encoding="utf-8-sig")
    measured = pd.read_csv(args.dataset, encoding="utf-8-sig")
    if candidates.empty or measured.empty or not candidates["candidate_id"].is_unique:
        raise ValueError("Candidate IDs must be unique, and both input CSVs must be non-empty.")

    predictions = candidates.copy()
    w = args.uncertainty_weight

    for offset, name in enumerate(TARGET_COLUMNS):
        model_path = args.model_dir / f"{name}_xgboost.joblib"
        if not model_path.exists():
            print(f"Notice: {model_path} not found. Skipping {name} prediction.")
            continue

        artifact = load_artifact(model_path, name, dependencies)
        point, mean, std = predict_ensemble(
            artifact, measured, predictions, args.ensemble_size, args.random_state + offset * 10_000, dependencies
        )

        if name == "energy":
            predictions["predicted_energy_j"] = point
            predictions["ensemble_energy_mean_j"] = mean
            predictions["energy_uncertainty_std_j"] = std
            predictions["energy_lower_j"] = np.maximum(1e-9, mean - w * std)
            predictions["energy_upper_j"] = mean + w * std
            predictions["energy_relative_uncertainty"] = std / np.maximum(mean, np.finfo(float).eps)
        elif name == "latency":
            predictions["predicted_latency_ms"] = point
            predictions["ensemble_latency_mean_ms"] = mean
            predictions["latency_uncertainty_std_ms"] = std
            predictions["latency_lower_ms"] = np.maximum(1e-9, mean - w * std)
            predictions["latency_upper_ms"] = mean + w * std
            predictions["latency_relative_uncertainty"] = std / np.maximum(mean, np.finfo(float).eps)
        elif name == "accuracy":
            predictions["predicted_accuracy_percent"] = point
            predictions["ensemble_accuracy_mean_percent"] = mean
            predictions["accuracy_uncertainty_std_percent"] = std
            predictions["accuracy_lower_percent"] = mean - w * std
            predictions["accuracy_upper_percent"] = mean + w * std
            predictions["accuracy_relative_uncertainty"] = std / np.maximum(mean, np.finfo(float).eps)

    unc_cols = [c for c in predictions.columns if c.endswith("_relative_uncertainty")]
    if unc_cols:
        predictions["combined_relative_uncertainty"] = predictions[unc_cols].mean(axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Wrote {args.output_csv} ({len(predictions)} candidates, {args.ensemble_size} bootstrap replicas).")


if __name__ == "__main__":
    main()
