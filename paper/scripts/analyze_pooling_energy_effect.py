"""Test the adjusted association between CNN pooling and inference energy.

The primary model regresses log inference energy on pooling count while
controlling for depth, parameter count, channel geometry, channel pattern, and
growth pattern. HC3 robust standard errors are used because energy variability
is heteroscedastic across the sampled CNN structures. A pool-ratio model is
reported as a prespecified sensitivity analysis; it is not fit jointly with
pool count because the two quantities are mechanically related.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TARGET = "target_energy_j"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_NUMERIC = [
    "feature_depth",
    "feature_parameter_count",
    "feature_channel_mean",
    "feature_channel_std",
    "feature_channel_last_first_ratio",
]
CATEGORICAL = ["feature_pattern", "feature_growth_pattern"]


def add_design_columns(frame: pd.DataFrame, pooling_feature: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    data = frame.copy()
    data["log_parameter_count"] = np.log(data["feature_parameter_count"])
    numeric = [pooling_feature, "feature_depth", "log_parameter_count", *BASE_NUMERIC[2:]]
    controls = data[numeric].astype(float)
    categories = pd.get_dummies(data[CATEGORICAL], prefix=CATEGORICAL, drop_first=True, dtype=float)
    design = pd.concat([controls, categories], axis=1)
    names = ["intercept", *design.columns.tolist()]
    return np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)]), names, np.log(data[TARGET].to_numpy(dtype=float))


def hc3_ols(x: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    n_obs, n_params = x.shape
    inverse_xtx = np.linalg.inv(x.T @ x)
    leverage = np.sum((x @ inverse_xtx) * x, axis=1)
    hc3_weights = (residual / np.maximum(1 - leverage, 1e-12)) ** 2
    covariance = inverse_xtx @ ((x.T * hc3_weights) @ x) @ inverse_xtx
    standard_error = np.sqrt(np.diag(covariance))
    statistic = beta / standard_error
    degrees_freedom = n_obs - n_params
    p_values = 2 * stats.t.sf(np.abs(statistic), degrees_freedom)
    critical = stats.t.ppf(0.975, degrees_freedom)
    coefficients = pd.DataFrame(
        {
            "term": names,
            "coefficient_log_energy": beta,
            "robust_se_hc3": standard_error,
            "t_statistic": statistic,
            "p_value": p_values,
            "ci95_lower_log_energy": beta - critical * standard_error,
            "ci95_upper_log_energy": beta + critical * standard_error,
        }
    )
    coefficients["percent_change_per_unit"] = (np.exp(coefficients["coefficient_log_energy"]) - 1) * 100
    coefficients["ci95_lower_percent"] = (np.exp(coefficients["ci95_lower_log_energy"]) - 1) * 100
    coefficients["ci95_upper_percent"] = (np.exp(coefficients["ci95_upper_log_energy"]) - 1) * 100
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    model_stats = {
        "n_observations": float(n_obs),
        "n_parameters": float(n_params),
        "r_squared": 1 - float(np.sum(residual**2)) / total_sum_squares,
        "adjusted_r_squared": 1 - (1 - (1 - float(np.sum(residual**2)) / total_sum_squares)) * (n_obs - 1) / degrees_freedom,
        "degrees_freedom": float(degrees_freedom),
    }
    return coefficients, model_stats, residual


def partial_residuals(x: np.ndarray, y: np.ndarray, pooling_index: int) -> tuple[np.ndarray, np.ndarray]:
    control_indices = [index for index in range(x.shape[1]) if index != pooling_index]
    controls = x[:, control_indices]
    y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    pooling = x[:, pooling_index]
    pooling_residual = pooling - controls @ np.linalg.lstsq(controls, pooling, rcond=None)[0]
    return pooling_residual, y_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "measurements/ml/model_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper/results/pooling_effects/energy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset, encoding="utf-8-sig")
    required = {TARGET, "feature_pool_count", *BASE_NUMERIC, *CATEGORICAL}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame.dropna(subset=required).copy()
    frame = frame[frame[TARGET] > 0].copy()
    frame["feature_pool_ratio"] = frame["feature_pool_count"] / frame["feature_depth"]
    if len(frame) < 30:
        raise ValueError("At least 30 valid measured structures are required.")

    all_coefficients: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float | str | bool]] = {}
    plot_data: tuple[np.ndarray, np.ndarray, str] | None = None
    for analysis_name, pooling_feature in (("primary_pool_count", "feature_pool_count"), ("sensitivity_pool_ratio", "feature_pool_ratio")):
        x, names, y = add_design_columns(frame, pooling_feature)
        coefficients, model_stats, _ = hc3_ols(x, y, names)
        coefficients.insert(0, "analysis", analysis_name)
        all_coefficients.append(coefficients)
        pooling_row = coefficients.loc[coefficients["term"] == pooling_feature].iloc[0]
        summaries[analysis_name] = {
            **model_stats,
            "pooling_feature": pooling_feature,
            "coefficient_log_energy": float(pooling_row["coefficient_log_energy"]),
            "percent_change_per_unit": float(pooling_row["percent_change_per_unit"]),
            "ci95_lower_percent": float(pooling_row["ci95_lower_percent"]),
            "ci95_upper_percent": float(pooling_row["ci95_upper_percent"]),
            "p_value_hc3": float(pooling_row["p_value"]),
            "significant_at_0_05": bool(pooling_row["p_value"] < 0.05),
        }
        if analysis_name == "primary_pool_count":
            plot_data = (*partial_residuals(x, y, names.index(pooling_feature)), pooling_feature)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_table = pd.concat(all_coefficients, ignore_index=True)
    coefficient_table.to_csv(args.output_dir / "pooling_energy_coefficients.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "pooling_energy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    assert plot_data is not None
    x_partial, y_partial, label = plot_data
    slope, intercept, r_value, _, _ = stats.linregress(x_partial, y_partial)
    order = np.argsort(x_partial)
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    axis.scatter(x_partial, y_partial, alpha=0.55, color="#2563eb", edgecolors="none")
    axis.plot(x_partial[order], intercept + slope * x_partial[order], color="#dc2626", linewidth=2)
    axis.set(
        xlabel="Residualized pooling count",
        ylabel="Residualized log inference energy",
        title="Adjusted association of pooling count with inference energy",
    )
    axis.text(0.03, 0.96, f"Partial r = {r_value:.3f}", transform=axis.transAxes, va="top")
    figure.savefig(args.output_dir / "pooling_energy_partial_residual.png", dpi=220)
    plt.close(figure)

    primary = summaries["primary_pool_count"]
    print(
        "Primary pooling-count result: "
        f"{primary['percent_change_per_unit']:.2f}% per additional pool "
        f"(95% CI {primary['ci95_lower_percent']:.2f}% to {primary['ci95_upper_percent']:.2f}%, "
        f"HC3 p={primary['p_value_hc3']:.4g})."
    )
    print(f"Wrote pooling-effect outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
