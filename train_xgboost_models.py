"""Train regularized XGBoost models for CNN inference energy and latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "energy": "target_energy_j",
    "latency": "target_latency_ms",
}


def import_ml_dependencies():
    """Import optional ML dependencies with an actionable installation error."""
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "Missing ML dependencies. Install them in the project environment with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn xgboost joblib"
        ) from error
    return {
        "joblib": joblib,
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "XGBRegressor": XGBRegressor,
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, ml: dict[str, object]) -> dict[str, float]:
    return {
        "mae": float(ml["mean_absolute_error"](actual, predicted)),
        "rmse": float(ml["mean_squared_error"](actual, predicted) ** 0.5),
        "r2": float(ml["r2_score"](actual, predicted)),
    }


def train_target(
    name: str,
    target_column: str,
    features: pd.DataFrame,
    target_values: pd.Series,
    split: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    ml: dict[str, object],
    random_state: int,
) -> tuple[object, object, pd.Series, dict[str, dict[str, float]], pd.DataFrame]:
    """Fit one log-target model and return predictions, metrics, and importances."""
    preprocessor = ml["ColumnTransformer"](
        transformers=[
            (
                "numeric",
                ml["Pipeline"]([("imputer", ml["SimpleImputer"](strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                ml["Pipeline"](
                    [
                        ("imputer", ml["SimpleImputer"](strategy="most_frequent")),
                        ("onehot", ml["OneHotEncoder"](handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    train_mask = split == "train"
    validation_mask = split == "validation"
    transformed_train = preprocessor.fit_transform(features.loc[train_mask])
    transformed_validation = preprocessor.transform(features.loc[validation_mask])
    transformed_all = preprocessor.transform(features)

    target = np.log(target_values.to_numpy(dtype=float))
    model = ml["XGBRegressor"](
        objective="reg:squarederror",
        n_estimators=1_000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        transformed_train,
        target[train_mask],
        eval_set=[(transformed_validation, target[validation_mask])],
        verbose=False,
    )
    predictions = pd.Series(np.exp(model.predict(transformed_all)), index=features.index)
    actual = target_values.to_numpy(dtype=float)
    metrics = {
        subset: regression_metrics(
            actual[split.to_numpy() == subset],
            predictions.to_numpy()[split.to_numpy() == subset],
            ml,
        )
        for subset in ("train", "validation", "test")
    }
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            f"{name}_importance": model.feature_importances_,
        }
    ).sort_values(f"{name}_importance", ascending=False)
    return model, preprocessor, predictions, metrics, importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("measurements/ml/model_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/ml"))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 0.5 or not 0.0 < args.validation_size < 0.5:
        raise ValueError("--test-size and --validation-size must be between 0 and 0.5.")
    ml = import_ml_dependencies()
    dataset = pd.read_csv(args.dataset, encoding="utf-8-sig")
    required = {"model_id", *TARGETS.values()}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(dataset) < 30:
        raise ValueError("At least 30 valid models are required for train/validation/test splitting.")

    feature_columns = [column for column in dataset.columns if column.startswith("feature_")]
    categorical_columns = [
        column for column in feature_columns if dataset[column].dtype == object or column.endswith("_pattern")
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    if not numeric_columns or not categorical_columns:
        raise ValueError("Expected both numeric and categorical structure features in model_dataset.csv.")
    features = dataset[feature_columns].copy()

    indices = np.arange(len(dataset))

# Stratification groups: depth × pattern
    stratify_groups = (
        dataset["feature_depth"].astype(str)
        + "_"
        + dataset["feature_pattern"].astype(str)
    )

    # First split: 70% train+validation / 15% test
    train_validation, test = ml["train_test_split"](
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_groups,
    )

    # Second split: 70% train / 15% validation
    validation_share = args.validation_size / (1.0 - args.test_size)

    train, validation = ml["train_test_split"](
        train_validation,
        test_size=validation_share,
        random_state=args.random_state,
        stratify=stratify_groups.iloc[train_validation],
    )
    split = pd.Series("test", index=dataset.index, name="split")
    split.iloc[train] = "train"
    split.iloc[validation] = "validation"

    energy_model, energy_preprocessor, energy_predictions, energy_metrics, energy_importance = train_target(
        "energy",
        TARGETS["energy"],
        features,
        dataset[TARGETS["energy"]],
        split,
        numeric_columns,
        categorical_columns,
        ml,
        args.random_state,
    )
    latency_model, latency_preprocessor, latency_predictions, latency_metrics, latency_importance = train_target(
        "latency",
        TARGETS["latency"],
        features,
        dataset[TARGETS["latency"]],
        split,
        numeric_columns,
        categorical_columns,
        ml,
        args.random_state + 1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = dataset[["model_id", *TARGETS.values(), "target_energy_cv_percent"]].copy()
    predictions["split"] = split
    predictions["predicted_energy_j"] = energy_predictions
    predictions["predicted_latency_ms"] = latency_predictions
    predictions["energy_residual_j"] = predictions["target_energy_j"] - predictions["predicted_energy_j"]
    predictions["latency_residual_ms"] = predictions["target_latency_ms"] - predictions["predicted_latency_ms"]
    predictions.to_csv(args.output_dir / "model_predictions.csv", index=False, encoding="utf-8")

    importance = energy_importance.merge(latency_importance, on="feature", how="outer").fillna(0)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8")
    metrics = {"energy": energy_metrics, "latency": latency_metrics}
    (args.output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    ml["joblib"].dump(
        {"model": energy_model, "preprocessor": energy_preprocessor, "features": feature_columns},
        args.output_dir / "energy_xgboost.joblib",
    )
    ml["joblib"].dump(
        {"model": latency_model, "preprocessor": latency_preprocessor, "features": feature_columns},
        args.output_dir / "latency_xgboost.joblib",
    )
    print(f"Wrote {args.output_dir / 'model_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'feature_importance.csv'}")
    print(f"Wrote {args.output_dir / 'model_metrics.json'}")
    print("Test metrics:")
    for name, values in metrics.items():
        print(f"  {name}: MAE={values['test']['mae']:.6g}, RMSE={values['test']['rmse']:.6g}, R²={values['test']['r2']:.4f}")


if __name__ == "__main__":
    main()
