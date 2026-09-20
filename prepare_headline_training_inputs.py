"""Create the exact 0551 and 0530 input CSV for trained headline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HEADLINE_MODELS = {
    551: ("vgg_inspired_reference", "headline_reference_0551"),
    530: ("C002793", "headline_pareto_0530"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-csv", type=Path, default=Path("dataset_structure.csv"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("measurements/headline_comparison/headline_training_candidates.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structures = pd.read_csv(args.structure_csv, encoding="utf-8-sig")
    selected = structures[structures["id"].isin(HEADLINE_MODELS)].copy()
    if len(selected) != len(HEADLINE_MODELS):
        found = set(selected["id"])
        missing = sorted(set(HEADLINE_MODELS) - found)
        raise ValueError(f"Missing headline structures in {args.structure_csv}: {missing}")
    selected["id"] = selected["id"].astype(int).map(lambda value: f"{value:04d}")
    selected["candidate_id"] = selected["id"].map(
        {f"{model_id:04d}": candidate_id for model_id, (candidate_id, _) in HEADLINE_MODELS.items()}
    )
    selected["selection_reason"] = selected["id"].map(
        {f"{model_id:04d}": reason for model_id, (_, reason) in HEADLINE_MODELS.items()}
    )
    selected = selected.sort_values("id")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output_csv} with {len(selected)} models: {', '.join(selected['id'])}")


if __name__ == "__main__":
    main()
