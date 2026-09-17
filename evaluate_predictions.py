"""Evaluate candidate predictions against ground-truth measurements (Energy, Latency, Accuracy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute regression evaluation metrics."""
    valid_mask = np.isfinite(actual) & np.isfinite(predicted)
    y_true = actual[valid_mask]
    y_pred = predicted[valid_mask]

    if len(y_true) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan, "mape_percent": np.nan}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    # Pearson correlation
    corr_matrix = np.corrcoef(y_true, y_pred)
    pearson_r = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else np.nan

    # MAPE
    nonzero_mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "mape_percent": mape,
    }


def compute_interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute the percentage of actual values falling inside [lower, upper]."""
    valid_mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid_mask):
        return np.nan
    in_bounds = (actual[valid_mask] >= lower[valid_mask]) & (actual[valid_mask] <= upper[valid_mask])
    return float(np.mean(in_bounds) * 100.0)


def compute_pareto_mask(x: np.ndarray, y: np.ndarray, maximize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
    """Standard 2D Pareto frontier mask."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better_or_equal = (x[j] >= x[i]) if maximize_x else (x[j] <= x[i])
            y_better_or_equal = (y[j] <= y[i]) if minimize_y else (y[j] >= y[i])
            x_strictly_better = (x[j] > x[i]) if maximize_x else (x[j] < x[i])
            y_strictly_better = (y[j] < y[i]) if minimize_y else (y[j] > y[i])

            if x_better_or_equal and y_better_or_equal and (x_strictly_better or y_strictly_better):
                is_pareto[i] = False
                break
    return is_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("measurements/search/next_measurement_candidates.csv"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("measurements/search/candidate_onnx_metadata.csv"),
    )
    parser.add_argument(
        "--energy-summary-csv",
        type=Path,
        default=Path("measurements/processed/energy_summary.csv"),
    )
    parser.add_argument(
        "--accuracy-results-csv",
        type=Path,
        default=Path("measurements/ml/candidate_accuracy_50_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("measurements/evaluation"),
    )
    args = parser.parse_args()

    if not args.candidates_csv.exists():
        raise FileNotFoundError(f"Missing candidates CSV: {args.candidates_csv}")
    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {args.metadata_csv}")

    candidates_df = pd.read_csv(args.candidates_csv, encoding="utf-8-sig")
    metadata_df = pd.read_csv(args.metadata_csv, encoding="utf-8-sig")

    # Map candidate_id -> model_id (e.g. C028658 -> 0501)
    meta_map = dict(zip(metadata_df["candidate_id"], metadata_df["id"].astype(str).str.zfill(4)))
    candidates_df["model_id"] = candidates_df["candidate_id"].map(meta_map)

    # 1. Merge Actual Energy & Latency if available
    has_energy = False
    if args.energy_summary_csv.exists():
        energy_df = pd.read_csv(args.energy_summary_csv, encoding="utf-8-sig")
        energy_df["model_id"] = energy_df["model_id"].astype(str).str.zfill(4)
        cols_to_merge = ["model_id", "mean_net_energy_per_inference_j", "mean_average_latency_ms", "valid_trials"]
        merged_energy = energy_df[[c for c in cols_to_merge if c in energy_df.columns]]
        candidates_df = pd.merge(candidates_df, merged_energy, on="model_id", how="left")
        if candidates_df["mean_net_energy_per_inference_j"].notna().any():
            has_energy = True
            print(f"Loaded actual energy/latency for {candidates_df['mean_net_energy_per_inference_j'].notna().sum()}/{len(candidates_df)} models.")

    # 2. Merge Actual Accuracy if available
    has_accuracy = False
    if args.accuracy_results_csv.exists():
        acc_df = pd.read_csv(args.accuracy_results_csv, encoding="utf-8-sig")
        if "id" in acc_df.columns:
            acc_df["model_id"] = acc_df["id"].astype(str).str.zfill(4)
        acc_col = "best_accuracy" if "best_accuracy" in acc_df.columns else "final_accuracy"
        cols_to_merge = ["model_id", acc_col, "train_time_sec"]
        merged_acc = acc_df[[c for c in cols_to_merge if c in acc_df.columns]].rename(
            columns={acc_col: "actual_accuracy_percent"}
        )
        candidates_df = pd.merge(candidates_df, merged_acc, on="model_id", how="left")
        if candidates_df["actual_accuracy_percent"].notna().any():
            has_accuracy = True
            print(f"Loaded actual accuracy for {candidates_df['actual_accuracy_percent'].notna().sum()}/{len(candidates_df)} models.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Candidate Models Prediction vs. Ground-Truth Evaluation\n"]
    metrics_summary = {}

    # --- Evaluate Accuracy ---
    if has_accuracy and candidates_df["actual_accuracy_percent"].notna().any():
        actual_acc = candidates_df["actual_accuracy_percent"].to_numpy()
        pred_acc = candidates_df["predicted_accuracy_percent"].to_numpy()
        acc_metrics = compute_metrics(actual_acc, pred_acc)

        # Coverage
        if "accuracy_lower_percent" in candidates_df.columns and "accuracy_upper_percent" in candidates_df.columns:
            acc_cov = compute_interval_coverage(
                actual_acc,
                candidates_df["accuracy_lower_percent"].to_numpy(),
                candidates_df["accuracy_upper_percent"].to_numpy(),
            )
            acc_metrics["coverage_percent"] = acc_cov

        metrics_summary["accuracy"] = acc_metrics
        report_lines.append("## 1. Accuracy Surrogate Model Evaluation")
        report_lines.append(f"- **Evaluated Models**: {int(np.sum(np.isfinite(actual_acc)))} / {len(candidates_df)}")
        report_lines.append(f"- **Pearson Correlation ($r$)**: {acc_metrics['pearson_r']:.4f}")
        report_lines.append(f"- **$R^2$ Score**: {acc_metrics['r2']:.4f}")
        report_lines.append(f"- **MAE**: {acc_metrics['mae']:.2f}%")
        report_lines.append(f"- **RMSE**: {acc_metrics['rmse']:.2f}%")
        if "coverage_percent" in acc_metrics:
            report_lines.append(f"- **Interval Coverage [LCB, UCB]**: {acc_metrics['coverage_percent']:.1f}%\n")

    # --- Evaluate Energy ---
    if has_energy and candidates_df["mean_net_energy_per_inference_j"].notna().any():
        actual_energy = candidates_df["mean_net_energy_per_inference_j"].to_numpy()
        pred_energy = candidates_df["predicted_energy_j"].to_numpy()
        energy_metrics = compute_metrics(actual_energy, pred_energy)

        if "energy_lower_j" in candidates_df.columns and "energy_upper_j" in candidates_df.columns:
            energy_cov = compute_interval_coverage(
                actual_energy,
                candidates_df["energy_lower_j"].to_numpy(),
                candidates_df["energy_upper_j"].to_numpy(),
            )
            energy_metrics["coverage_percent"] = energy_cov

        metrics_summary["energy"] = energy_metrics
        report_lines.append("## 2. Energy Surrogate Model Evaluation")
        report_lines.append(f"- **Evaluated Models**: {int(np.sum(np.isfinite(actual_energy)))} / {len(candidates_df)}")
        report_lines.append(f"- **Pearson Correlation ($r$)**: {energy_metrics['pearson_r']:.4f}")
        report_lines.append(f"- **$R^2$ Score**: {energy_metrics['r2']:.4f}")
        report_lines.append(f"- **MAE**: {energy_metrics['mae'] * 1000:.3f} mJ")
        report_lines.append(f"- **MAPE**: {energy_metrics['mape_percent']:.2f}%")
        if "coverage_percent" in energy_metrics:
            report_lines.append(f"- **Interval Coverage [LCB, UCB]**: {energy_metrics['coverage_percent']:.1f}%\n")

    # --- Evaluate Latency ---
    if has_energy and candidates_df["mean_average_latency_ms"].notna().any():
        actual_lat = candidates_df["mean_average_latency_ms"].to_numpy()
        pred_lat = candidates_df["predicted_latency_ms"].to_numpy()
        lat_metrics = compute_metrics(actual_lat, pred_lat)

        if "latency_lower_ms" in candidates_df.columns and "latency_upper_ms" in candidates_df.columns:
            lat_cov = compute_interval_coverage(
                actual_lat,
                candidates_df["latency_lower_ms"].to_numpy(),
                candidates_df["latency_upper_ms"].to_numpy(),
            )
            lat_metrics["coverage_percent"] = lat_cov

        metrics_summary["latency"] = lat_metrics
        report_lines.append("## 3. Latency Surrogate Model Evaluation")
        report_lines.append(f"- **Evaluated Models**: {int(np.sum(np.isfinite(actual_lat)))} / {len(candidates_df)}")
        report_lines.append(f"- **Pearson Correlation ($r$)**: {lat_metrics['pearson_r']:.4f}")
        report_lines.append(f"- **$R^2$ Score**: {lat_metrics['r2']:.4f}")
        report_lines.append(f"- **MAE**: {lat_metrics['mae']:.3f} ms")
        report_lines.append(f"- **MAPE**: {lat_metrics['mape_percent']:.2f}%")
        if "coverage_percent" in lat_metrics:
            report_lines.append(f"- **Interval Coverage [LCB, UCB]**: {lat_metrics['coverage_percent']:.1f}%\n")

    # --- Pareto Frontier Verification ---
    if has_accuracy and has_energy:
        valid_both = candidates_df["actual_accuracy_percent"].notna() & candidates_df["mean_net_energy_per_inference_j"].notna()
        sub = candidates_df.loc[valid_both].copy()
        if len(sub) >= 5:
            actual_acc_sub = sub["actual_accuracy_percent"].to_numpy()
            actual_energy_sub = sub["mean_net_energy_per_inference_j"].to_numpy()

            actual_pareto = compute_pareto_mask(actual_acc_sub, actual_energy_sub, maximize_x=True, minimize_y=True)
            sub["is_actual_pareto"] = actual_pareto
            candidates_df.loc[valid_both, "is_actual_pareto"] = actual_pareto

            pareto_reasons = {"mean_acc_energy_pareto", "strict_acc_energy_pareto"}
            predicted_pareto_mask = sub["selection_reason"].isin(pareto_reasons)
            n_pred_pareto = int(predicted_pareto_mask.sum())
            n_actual_pareto = int(actual_pareto.sum())
            n_intersection = int((predicted_pareto_mask & actual_pareto).sum())

            precision = (n_intersection / n_pred_pareto * 100.0) if n_pred_pareto > 0 else 0.0
            recall = (n_intersection / n_actual_pareto * 100.0) if n_actual_pareto > 0 else 0.0

            report_lines.append("## 4. Empirical Pareto Frontier Analysis")
            report_lines.append(f"- **Ground-Truth Pareto Models Identified**: {n_actual_pareto} out of {len(sub)}")
            report_lines.append(f"- **Models Selected via Predicted Pareto**: {n_pred_pareto}")
            report_lines.append(f"- **Matched Ground-Truth Pareto**: {n_intersection}")
            report_lines.append(f"- **Pareto Hit Rate (Precision)**: {precision:.1f}%")
            report_lines.append(f"- **Pareto Coverage (Recall)**: {recall:.1f}%\n")

    # Save detailed evaluation table
    eval_csv = args.output_dir / "candidate_evaluation_results.csv"
    candidates_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    report_lines.append(f"Detailed model-by-model evaluation saved to: `{eval_csv}`")

    # Save Markdown report & JSON
    report_md = args.output_dir / "evaluation_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = args.output_dir / "evaluation_metrics.json"
    summary_json.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
