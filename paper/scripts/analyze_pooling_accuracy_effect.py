"""Test the adjusted association between CNN pooling and measured accuracy.

Only structures with an observed CIFAR-10 accuracy label are included. The
result is an association within this evaluated subset, not a causal effect or
an estimate for every generated architecture, because accuracy labels were
collected for a targeted subset of candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TARGET = "target_accuracy_percent"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_NUMERIC = [
    "feature_depth",
    "feature_parameter_count",
    "feature_channel_mean",
    "feature_channel_std",
    "feature_channel_last_first_ratio",
]
CATEGORICAL = ["feature_pattern", "feature_growth_pattern"]


def build_design(frame: pd.DataFrame, pooling_feature: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    data = frame.copy()
    data["log_parameter_count"] = np.log(data["feature_parameter_count"])
    numeric = [pooling_feature, "feature_depth", "log_parameter_count", *BASE_NUMERIC[2:]]
    categories = pd.get_dummies(data[CATEGORICAL], prefix=CATEGORICAL, drop_first=True, dtype=float)
    features = pd.concat([data[numeric].astype(float), categories], axis=1)
    return (
        np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)]),
        ["intercept", *features.columns.tolist()],
        data[TARGET].to_numpy(dtype=float),
    )


def fit_hc3_ols(x: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    n_obs, n_params = x.shape
    degrees_freedom = n_obs - n_params
    inverse_xtx = np.linalg.inv(x.T @ x)
    leverage = np.sum((x @ inverse_xtx) * x, axis=1)
    weights = (residual / np.maximum(1 - leverage, 1e-12)) ** 2
    covariance = inverse_xtx @ ((x.T * weights) @ x) @ inverse_xtx
    standard_error = np.sqrt(np.diag(covariance))
    statistic = beta / standard_error
    p_values = 2 * stats.t.sf(np.abs(statistic), degrees_freedom)
    critical = stats.t.ppf(0.975, degrees_freedom)
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - float(np.sum(residual**2)) / total_sum_squares
    table = pd.DataFrame(
        {
            "term": names,
            "coefficient_accuracy_percentage_points": beta,
            "robust_se_hc3": standard_error,
            "t_statistic": statistic,
            "p_value": p_values,
            "ci95_lower_percentage_points": beta - critical * standard_error,
            "ci95_upper_percentage_points": beta + critical * standard_error,
        }
    )
    return table, {
        "n_observations": float(n_obs),
        "n_parameters": float(n_params),
        "degrees_freedom": float(degrees_freedom),
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (n_obs - 1) / degrees_freedom,
    }


def partial_residuals(x: np.ndarray, y: np.ndarray, variable_index: int) -> tuple[np.ndarray, np.ndarray]:
    control_indices = [index for index in range(x.shape[1]) if index != variable_index]
    controls = x[:, control_indices]
    y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    variable = x[:, variable_index]
    variable_residual = variable - controls @ np.linalg.lstsq(controls, variable, rcond=None)[0]
    return variable_residual, y_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "measurements/ml/model_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper/results/pooling_effects/accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset, encoding="utf-8-sig")
    required = {TARGET, "feature_pool_count", *BASE_NUMERIC, *CATEGORICAL}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame.dropna(subset=required).copy()
    frame["feature_pool_ratio"] = frame["feature_pool_count"] / frame["feature_depth"]
    if len(frame) < 30:
        raise ValueError("At least 30 labeled structures are required for this regression.")

    all_tables: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float | bool | str]] = {}
    partial: tuple[np.ndarray, np.ndarray] | None = None
    for analysis, pooling_feature in (("primary_pool_count", "feature_pool_count"), ("sensitivity_pool_ratio", "feature_pool_ratio")):
        x, names, y = build_design(frame, pooling_feature)
        table, model_stats = fit_hc3_ols(x, y, names)
        table.insert(0, "analysis", analysis)
        all_tables.append(table)
        pooling_row = table.loc[table["term"] == pooling_feature].iloc[0]
        summaries[analysis] = {
            **model_stats,
            "pooling_feature": pooling_feature,
            "coefficient_percentage_points_per_unit": float(pooling_row["coefficient_accuracy_percentage_points"]),
            "ci95_lower_percentage_points": float(pooling_row["ci95_lower_percentage_points"]),
            "ci95_upper_percentage_points": float(pooling_row["ci95_upper_percentage_points"]),
            "p_value_hc3": float(pooling_row["p_value"]),
            "significant_at_0_05": bool(pooling_row["p_value"] < 0.05),
        }
        if analysis == "primary_pool_count":
            partial = partial_residuals(x, y, names.index(pooling_feature))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_tables, ignore_index=True).to_csv(
        args.output_dir / "pooling_accuracy_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    with (args.output_dir / "pooling_accuracy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    assert partial is not None
    x_partial, y_partial = partial
    slope, intercept, r_value, _, _ = stats.linregress(x_partial, y_partial)
    order = np.argsort(x_partial)
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    axis.scatter(x_partial, y_partial, alpha=0.65, color="#7c3aed", edgecolors="none")
    axis.plot(x_partial[order], intercept + slope * x_partial[order], color="#dc2626", linewidth=2)
    axis.set(
        xlabel="Residualized pooling count",
        ylabel="Residualized accuracy (percentage points)",
        title="Adjusted association of pooling count with measured accuracy",
    )
    axis.text(0.03, 0.96, f"Partial r = {r_value:.3f}", transform=axis.transAxes, va="top")
    figure.savefig(args.output_dir / "pooling_accuracy_partial_residual.png", dpi=220)
    plt.close(figure)

    primary = summaries["primary_pool_count"]
    print(
        "Primary pooling-count result: "
        f"{primary['coefficient_percentage_points_per_unit']:.3f} percentage points per additional pool "
        f"(95% CI {primary['ci95_lower_percentage_points']:.3f} to {primary['ci95_upper_percentage_points']:.3f}, "
        f"HC3 p={primary['p_value_hc3']:.4g})."
    )
    print(f"Wrote pooling-accuracy outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
