"""Train regularized XGBoost models for CNN inference energy, latency, and accuracy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "energy": "target_energy_j",
    "latency": "target_latency_ms",
    "accuracy": "target_accuracy_percent",
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
    if len(actual) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    return {
        "mae": float(ml["mean_absolute_error"](actual, predicted)),
        "rmse": float(ml["mean_squared_error"](actual, predicted) ** 0.5),
        "r2": float(ml["r2_score"](actual, predicted)),
    }


def split_labeled_accuracy_indices(
    indices: np.ndarray,
    pattern_groups: pd.Series,
    test_fraction: float,
    validation_fraction: float,
    random_state: int,
    ml: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a three-way accuracy split, preserving every pattern when feasible."""
    class_count = int(pattern_groups.nunique())
    group_counts = pattern_groups.value_counts()
    total = len(indices)
    test_count = max(math.ceil(total * test_fraction), class_count)
    # With the first stratified holdout removed, sklearn can omit a small class
    # from a validation split sized exactly at the number of classes.  One extra
    # validation row keeps every pattern represented for the current small set.
    validation_count = max(math.ceil(total * validation_fraction), class_count + 1)
    can_stratify = (
        group_counts.min() >= 3
        and total - test_count - validation_count >= class_count
    )

    if can_stratify:
        train_val, test = ml["train_test_split"](
            indices,
            test_size=test_count,
            random_state=random_state,
            stratify=pattern_groups,
        )
        train, validation = ml["train_test_split"](
            train_val,
            test_size=validation_count,
            random_state=random_state,
            stratify=pattern_groups.iloc[train_val],
        )
        return train, validation, test

    # Tiny or imbalanced labeled sets cannot place every pattern in all three splits.
    # Keep the workflow runnable while explicitly reporting the weaker validation design.
    test_count = max(1, math.ceil(total * test_fraction))
    validation_count = max(1, math.ceil(total * validation_fraction))
    if total - test_count - validation_count < 2:
        raise ValueError("Not enough annotated models for an accuracy train/validation/test split.")
    print(
        "Warning: accuracy labels are too small or imbalanced for a three-way "
        "pattern-stratified split; using a reproducible unstratified split."
    )
    train_val, test = ml["train_test_split"](
        indices, test_size=test_count, random_state=random_state
    )
    train, validation = ml["train_test_split"](
        train_val, test_size=validation_count, random_state=random_state
    )
    return train, validation, test


def train_target(
    name: str,
    target_column: str,
    features: pd.DataFrame,
    target_values: pd.Series,
    split: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    ml: dict[str, object],
    random_state: int,
    use_log_transform: bool = True,
) -> tuple[object, object, pd.Series, dict[str, dict[str, float]], pd.DataFrame]:
    """Fit one model and return predictions, metrics, and importances."""
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

    target_raw = target_values.to_numpy(dtype=float)
    target = np.log(target_raw) if use_log_transform else target_raw

    # Parameter tuning based on dataset size
    n_samples = len(features.loc[train_mask])
    max_depth = 3 if n_samples > 100 else 2
    n_estimators = 1_000 if n_samples > 100 else 300
    lr = 0.03 if n_samples > 100 else 0.05

    model = ml["XGBRegressor"](
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        min_child_weight=2 if n_samples < 100 else 3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0 if n_samples < 100 else 3.0,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30 if n_samples < 100 else 50,
    )
    model.fit(
        transformed_train,
        target[train_mask],
        eval_set=[(transformed_validation, target[validation_mask])],
        verbose=False,
    )

    pred_raw = model.predict(transformed_all)
    predictions = pd.Series(np.exp(pred_raw) if use_log_transform else pred_raw, index=features.index)

    metrics = {
        subset: regression_metrics(
            target_raw[split.to_numpy() == subset],
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

    required = {"model_id", TARGETS["energy"], TARGETS["latency"]}
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

    # Split for Energy & Latency (all models)
    indices = np.arange(len(dataset))
    stratify_groups = dataset["feature_depth"].astype(str) + "_" + dataset["feature_pattern"].astype(str)

    train_val, test = ml["train_test_split"](
        indices, test_size=args.test_size, random_state=args.random_state, stratify=stratify_groups
    )
    val_share = args.validation_size / (1.0 - args.test_size)
    train, val = ml["train_test_split"](
        train_val, test_size=val_share, random_state=args.random_state, stratify=stratify_groups.iloc[train_val]
    )

    split = pd.Series("test", index=dataset.index, name="split")
    split.iloc[train] = "train"
    split.iloc[val] = "validation"

    # Train Energy Model
    energy_model, energy_preproc, energy_preds, energy_metrics, energy_imp = train_target(
        "energy",
        TARGETS["energy"],
        features,
        dataset[TARGETS["energy"]],
        split,
        numeric_columns,
        categorical_columns,
        ml,
        args.random_state,
        use_log_transform=True,
    )

    # Train Latency Model
    latency_model, latency_preproc, latency_preds, latency_metrics, latency_imp = train_target(
        "latency",
        TARGETS["latency"],
        features,
        dataset[TARGETS["latency"]],
        split,
        numeric_columns,
        categorical_columns,
        ml,
        args.random_state + 1,
        use_log_transform=True,
    )

    # Train Accuracy Model (if target_accuracy_percent exists and has valid rows)
    accuracy_metrics = None
    accuracy_preds = pd.Series(np.nan, index=dataset.index)
    accuracy_imp = pd.DataFrame(columns=["feature", "accuracy_importance"])

    if TARGETS["accuracy"] in dataset.columns:
        acc_mask = dataset[TARGETS["accuracy"]].notna()
        acc_dataset = dataset[acc_mask].copy()

        if len(acc_dataset) >= 15:
            print(f"\nTraining Accuracy Model on {len(acc_dataset)} annotated models...")
            acc_features = features.loc[acc_mask].copy()
            acc_stratify = acc_dataset["feature_pattern"].astype(str)

            acc_indices = np.arange(len(acc_dataset))
            acc_train, acc_val, acc_test = split_labeled_accuracy_indices(
                acc_indices,
                acc_stratify,
                args.test_size,
                args.validation_size,
                args.random_state,
                ml,
            )
            print(
                "Accuracy split: "
                f"train={len(acc_train)}, validation={len(acc_val)}, test={len(acc_test)} "
                "(pattern-stratified when feasible)."
            )

            acc_split = pd.Series("test", index=acc_dataset.index, name="split")
            acc_split.iloc[acc_train] = "train"
            acc_split.iloc[acc_val] = "validation"

            accuracy_model, accuracy_preproc, acc_sub_preds, accuracy_metrics, accuracy_imp = train_target(
                "accuracy",
                TARGETS["accuracy"],
                acc_features,
                acc_dataset[TARGETS["accuracy"]],
                acc_split,
                numeric_columns,
                categorical_columns,
                ml,
                args.random_state + 2,
                use_log_transform=False,
            )
            
            # Predict for ALL models in dataset
            all_acc_preds = accuracy_model.predict(accuracy_preproc.transform(features))
            accuracy_preds = pd.Series(all_acc_preds, index=dataset.index)

            ml["joblib"].dump(
                {"model": accuracy_model, "preprocessor": accuracy_preproc, "features": feature_columns},
                args.output_dir / "accuracy_xgboost.joblib",
            )

    # Save Output Artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = dataset[["model_id", TARGETS["energy"], TARGETS["latency"]]].copy()
    predictions["split"] = split
    predictions["predicted_energy_j"] = energy_preds
    predictions["predicted_latency_ms"] = latency_preds

    if TARGETS["accuracy"] in dataset.columns:
        predictions[TARGETS["accuracy"]] = dataset[TARGETS["accuracy"]]
        predictions["predicted_accuracy_percent"] = accuracy_preds

    predictions.to_csv(args.output_dir / "model_predictions.csv", index=False, encoding="utf-8")

    importance = energy_imp.merge(latency_imp, on="feature", how="outer").fillna(0)
    if not accuracy_imp.empty:
        importance = importance.merge(accuracy_imp, on="feature", how="outer").fillna(0)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8")

    metrics = {"energy": energy_metrics, "latency": latency_metrics}
    if accuracy_metrics is not None:
        metrics["accuracy"] = accuracy_metrics

    (args.output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ml["joblib"].dump(
        {"model": energy_model, "preprocessor": energy_preproc, "features": feature_columns},
        args.output_dir / "energy_xgboost.joblib",
    )
    ml["joblib"].dump(
        {"model": latency_model, "preprocessor": latency_preproc, "features": feature_columns},
        args.output_dir / "latency_xgboost.joblib",
    )

    print(f"\nWrote artifacts to {args.output_dir}:")
    print(f"  - model_predictions.csv")
    print(f"  - feature_importance.csv")
    print(f"  - model_metrics.json")
    print(f"  - energy_xgboost.joblib, latency_xgboost.joblib" + (", accuracy_xgboost.joblib" if accuracy_metrics else ""))

    print("\n--- Test Metrics Summary ---")
    for name, values in metrics.items():
        print(
            f"  {name.upper()}: MAE={values['test']['mae']:.4f}, RMSE={values['test']['rmse']:.4f}, R²={values['test']['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
