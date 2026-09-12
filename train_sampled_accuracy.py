"""Train selected representative models on CIFAR-10 and collect accuracy results."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from FlexibleCNN import FlexibleCNN, ModelConfig
from load_data import get_dataloaders


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


def train_single_model(
    config: ModelConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = "cuda",
) -> dict[str, float]:
    """Train a single FlexibleCNN model on CIFAR-10 with AMP and Cosine Annealing."""
    model = FlexibleCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device == "cuda"))

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Evaluate at epoch end
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        val_loss /= total
        if val_acc > best_acc:
            best_acc = val_acc

    elapsed_sec = time.time() - start_time
    return {
        "final_accuracy": val_acc,
        "best_accuracy": best_acc,
        "final_val_loss": val_loss,
        "train_time_sec": elapsed_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, default=Path("measurements/ml/selected_50_models.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("measurements/ml/accuracy_50_results.csv"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if not args.selected_csv.exists():
        raise FileNotFoundError(f"Input file {args.selected_csv} not found. Run select_accuracy_sample.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    selected_df = pd.read_csv(args.selected_csv)
    print(f"Loaded {len(selected_df)} models for training ({args.epochs} epochs each)...")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    results: list[dict[str, object]] = []

    for index, row in selected_df.iterrows():
        model_id = str(row["id"]).zfill(4)
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

        print(f"[{index+1}/{len(selected_df)}] Training Model {model_id} (Params: {config.parameter_count:,})...")
        metrics = train_single_model(
            config, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device
        )
        print(
            f"  --> Acc: {metrics['best_accuracy']:.2f}% | Time: {metrics['train_time_sec']:.1f}s"
        )

        res_row = row.to_dict()
        res_row.update(metrics)
        results.append(res_row)

    res_df = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSuccessfully trained {len(res_df)} models. Results saved to {args.output_csv}.")


if __name__ == "__main__":
    main()
