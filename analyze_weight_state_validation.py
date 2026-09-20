"""Analyze paired trained/untrained inference measurements.

The input is ``energy_trials.csv`` produced by analyze_energy.py from one or
more ABBA paired-benchmark sessions. It reports state-to-state correlation,
Bland-Altman agreement, and a paired TOST equivalence test on relative change.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METRICS = {
    "energy": ("net_energy_per_inference_j", 1_000.0, "Inference energy (mJ)"),
    "latency": ("average_latency_ms", 1.0, "Latency (ms)"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_values(
    rows: list[dict[str, str]],
    metric_column: str,
    scale: float,
    min_valid_trials: int,
    session_id: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("mode") != "inference" or row.get("status") != "valid":
            continue
        if session_id and row.get("benchmark_session_id") != session_id:
            continue
        pair_id = row.get("pair_id", "")
        state = row.get("weight_state", "")
        value = row.get(metric_column, "")
        if not pair_id or state not in {"untrained_seed_reconstructed", "trained_best_accuracy"} or not value:
            continue
        try:
            grouped.setdefault((pair_id, row.get("structure_id", ""), state), []).append(float(value) * scale)
        except ValueError:
            continue

    pairs: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (pair_id, structure_id, state), values in grouped.items():
        pairs.setdefault((pair_id, structure_id), {})[state] = values
    output: list[dict[str, object]] = []
    for (pair_id, structure_id), states in sorted(pairs.items()):
        untrained = states.get("untrained_seed_reconstructed", [])
        trained = states.get("trained_best_accuracy", [])
        if len(untrained) < min_valid_trials or len(trained) < min_valid_trials:
            continue
        before, after = mean(untrained), mean(trained)
        difference = after - before
        output.append(
            {
                "pair_id": pair_id,
                "structure_id": structure_id,
                "untrained_mean": before,
                "trained_mean": after,
                "untrained_trials": len(untrained),
                "trained_trials": len(trained),
                "untrained_std": stdev(untrained) if len(untrained) > 1 else 0.0,
                "trained_std": stdev(trained) if len(trained) > 1 else 0.0,
                "mean_of_states": (before + after) / 2,
                "difference_trained_minus_untrained": difference,
                "relative_difference_percent": difference / before * 100 if before else np.nan,
            }
        )
    return output


def tost(relative_differences: np.ndarray, bound: float, alpha: float) -> dict[str, object]:
    count = len(relative_differences)
    if count < 2:
        return {"n_pairs": count, "equivalent": False, "reason": "At least two paired structures are required."}
    sample_mean = float(np.mean(relative_differences))
    sample_sd = float(np.std(relative_differences, ddof=1))
    standard_error = sample_sd / np.sqrt(count)
    df = count - 1
    if standard_error == 0:
        lower_p = 0.0 if sample_mean > -bound else 1.0
        upper_p = 0.0 if sample_mean < bound else 1.0
        ci_lower = ci_upper = sample_mean
    else:
        lower_p = float(stats.t.sf((sample_mean + bound) / standard_error, df))
        upper_p = float(stats.t.cdf((sample_mean - bound) / standard_error, df))
        critical = float(stats.t.ppf(1 - alpha, df))  # 90% CI for alpha=.05
        ci_lower = sample_mean - critical * standard_error
        ci_upper = sample_mean + critical * standard_error
    return {
        "n_pairs": count,
        "equivalence_bounds_percent": f"±{bound:g}",
        "mean_relative_difference_percent": sample_mean,
        "ci90_lower_percent": ci_lower,
        "ci90_upper_percent": ci_upper,
        "p_lower": lower_p,
        "p_upper": upper_p,
        "equivalent": lower_p < alpha and upper_p < alpha,
    }


def metric_summary(pairs: list[dict[str, object]], metric: str, bound: float, alpha: float) -> dict[str, object]:
    untrained = np.array([float(row["untrained_mean"]) for row in pairs])
    trained = np.array([float(row["trained_mean"]) for row in pairs])
    differences = trained - untrained
    relative = np.array([float(row["relative_difference_percent"]) for row in pairs])
    correlation = stats.pearsonr(untrained, trained)
    spearman = stats.spearmanr(untrained, trained)
    paired_test = stats.ttest_rel(trained, untrained)
    bias = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    result: dict[str, object] = {
        "metric": metric,
        "n_pairs": len(pairs),
        "untrained_mean": float(np.mean(untrained)),
        "trained_mean": float(np.mean(trained)),
        "mean_difference_trained_minus_untrained": bias,
        "mean_relative_difference_percent": float(np.mean(relative)),
        "pearson_r": float(correlation.statistic),
        "pearson_p": float(correlation.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "paired_t_statistic": float(paired_test.statistic),
        "paired_t_p": float(paired_test.pvalue),
        "bland_altman_bias": bias,
        "bland_altman_sd": difference_sd,
        "bland_altman_lower_loa": bias - 1.96 * difference_sd,
        "bland_altman_upper_loa": bias + 1.96 * difference_sd,
    }
    result.update(tost(relative, bound, alpha))
    return result


def plot_correlation(metric_pairs: dict[str, list[dict[str, object]]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (metric, pairs) in zip(axes, metric_pairs.items()):
        _, _, label = METRICS[metric]
        before = np.array([float(row["untrained_mean"]) for row in pairs])
        after = np.array([float(row["trained_mean"]) for row in pairs])
        axis.scatter(before, after, color="#2563eb", alpha=0.8)
        low, high = min(before.min(), after.min()), max(before.max(), after.max())
        margin = (high - low) * 0.05 or 1.0
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#64748b", label="Identity")
        pearson = stats.pearsonr(before, after).statistic
        axis.set(xlabel=f"Untrained {label}", ylabel=f"Trained {label}", title=f"{metric.title()} agreement")
        axis.text(0.04, 0.95, f"Pearson r = {pearson:.3f}", transform=axis.transAxes, va="top")
        axis.legend()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_bland_altman(metric_pairs: dict[str, list[dict[str, object]]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (metric, pairs) in zip(axes, metric_pairs.items()):
        _, _, label = METRICS[metric]
        state_means = np.array([float(row["mean_of_states"]) for row in pairs])
        differences = np.array([float(row["difference_trained_minus_untrained"]) for row in pairs])
        bias = float(np.mean(differences))
        sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        axis.scatter(state_means, differences, color="#7c3aed", alpha=0.8)
        axis.axhline(bias, color="#111827", label=f"Bias: {bias:.3g}")
        axis.axhline(bias - 1.96 * sd, color="#ef4444", linestyle="--", label="95% limits of agreement")
        axis.axhline(bias + 1.96 * sd, color="#ef4444", linestyle="--")
        axis.set(xlabel=f"Mean {label}", ylabel=f"Trained - untrained {label}", title=f"{metric.title()} Bland-Altman")
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-trials",
        type=Path,
        default=Path("measurements/validation_weight_state/processed/energy_trials.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurements/validation_weight_state/analysis"))
    parser.add_argument("--benchmark-session-id", default="")
    parser.add_argument("--min-valid-trials", type=int, default=2)
    parser.add_argument("--energy-equivalence-percent", type=float, default=5.0)
    parser.add_argument("--latency-equivalence-percent", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_valid_trials < 1 or args.alpha <= 0 or args.alpha >= 0.5:
        raise ValueError("min-valid-trials must be positive and alpha must be between 0 and 0.5.")
    rows = read_rows(args.energy_trials)
    metric_pairs: dict[str, list[dict[str, object]]] = {}
    summaries: list[dict[str, object]] = []
    bounds = {"energy": args.energy_equivalence_percent, "latency": args.latency_equivalence_percent}
    for metric, (column, scale, _) in METRICS.items():
        pairs = paired_values(rows, column, scale, args.min_valid_trials, args.benchmark_session_id)
        if len(pairs) < 2:
            raise ValueError(f"{metric}: only {len(pairs)} complete pairs; at least two are required.")
        metric_pairs[metric] = pairs
        for row in pairs:
            row["metric"] = metric
        summaries.append(metric_summary(pairs, metric, bounds[metric], args.alpha))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = [row for pairs in metric_pairs.values() for row in pairs]
    pair_fields = list(paired_rows[0])
    write_csv(args.output_dir / "paired_measurements.csv", pair_fields, paired_rows)
    summary_fields = list(summaries[0])
    write_csv(args.output_dir / "metric_summary.csv", summary_fields, summaries)
    write_csv(args.output_dir / "tost_results.csv", summary_fields, summaries)
    plot_correlation(metric_pairs, args.output_dir / "weight_state_correlation.png")
    plot_bland_altman(metric_pairs, args.output_dir / "weight_state_bland_altman.png")
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print(f"Wrote paired analysis for {len(metric_pairs['energy'])} structures to {args.output_dir}")


if __name__ == "__main__":
    main()
