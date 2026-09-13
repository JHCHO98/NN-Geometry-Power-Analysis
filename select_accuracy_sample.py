"""Extract 50 representative models for accuracy training using Zero-Cost Proxies and Diversity Sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from FlexibleCNN import ModelConfig
from zero_cost_proxies import compute_zero_cost_proxies


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


def select_representative_models(
    structure_csv: Path, output_csv: Path, n_samples: int = 50, seed: int = 42
) -> pd.DataFrame:
    df = pd.read_csv(structure_csv)
    print(f"Loaded {len(df)} models from {structure_csv}.")

    print("Computing Zero-Cost Proxies for all models...")
    scores_list: list[dict[str, float]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _, row in df.iterrows():
        channels = parse_int_sequence(row["channels"])
        pools = parse_int_sequence(row["pools"])
        config = ModelConfig(
            depth=int(row["depth"]),
            channels=channels,
            pools=pools,
            pattern=str(row["pattern"]),
            growth_pattern=str(row["growth_pattern"]),
            noise_ratio=float(row["noise_ratio"]),
            min_channels=int(row["min_channels"]),
            max_channels=int(row["max_channels"]),
            seed=int(row["seed"]),
            parameter_count=int(row["parameter_count"]),
        )
        scores = compute_zero_cost_proxies(config, device=device)
        scores_list.append(scores)

    scores_df = pd.DataFrame(scores_list)
    df = pd.concat([df, scores_df], axis=1)

    print(f"Performing Stratified Diversity Sampling ({n_samples} models)...")
    np.random.seed(seed)
    random.seed(seed)

    selected_ids: list[str] = []
    patterns = df["pattern"].unique()
    per_pattern = n_samples // len(patterns)  # 10 models per pattern type

    for pat in patterns:
        pat_df = df[df["pattern"] == pat].copy()
        
        # Features for clustering
        feature_cols = ["parameter_count", "depth", "synflow_score", "jacob_cov_score"]
        X = pat_df[feature_cols].copy()
        X = StandardScaler().fit_transform(X)

        n_clusters = min(per_pattern, len(pat_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        pat_df["cluster"] = kmeans.fit_predict(X)

        # Pick model closest to center of each cluster
        for c in range(n_clusters):
            cluster_indices = pat_df[pat_df["cluster"] == c].index
            center = kmeans.cluster_centers_[c]
            distances = np.linalg.norm(X[pat_df.index.get_indexer(cluster_indices)] - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_ids.append(pat_df.loc[closest_idx, "id"])

    # If any remaining slots to reach n_samples
    if len(selected_ids) < n_samples:
        remaining = df[~df["id"].isin(selected_ids)].sample(n_samples - len(selected_ids), random_state=seed)
        selected_ids.extend(remaining["id"].tolist())

    selected_df = df[df["id"].isin(selected_ids)].sort_values("id").reset_index(drop=True)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(selected_df)} representative models to {output_csv}.")
    return selected_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-csv", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/ml/selected_50_models.csv"))
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    select_representative_models(args.structure_csv, args.output_csv, args.n_samples, args.seed)
