"""Train selected candidate CNNs on CIFAR-10 using 100% GPU in-memory acceleration (RTX 4090)."""

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
from load_data import load_data


torch.backends.cudnn.benchmark = True


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


class PureGpuCIFAR10:
    """Zero-CPU-bottleneck CIFAR-10 dataset residing completely in GPU VRAM."""

    def __init__(self, data_dict: dict, device: str = "cuda", is_train: bool = True):
        self.device = device
        self.is_train = is_train

        # Shape: (N, 3, 32, 32), uint8 -> float32 on GPU normalized to [0, 1]
        raw_data = torch.from_numpy(data_dict["data"]).float().to(device) / 255.0

        # Pre-normalize on GPU (Mean: 0.4914, 0.4822, 0.4465 / Std: 0.2023, 0.1994, 0.2010)
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
        self.data = (raw_data - mean) / std

        self.labels = torch.tensor(data_dict["labels"], dtype=torch.long, device=device)
        self.num_samples = len(self.labels)

    def get_batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)

        for start_idx in range(0, self.num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_x = self.data[batch_indices]
            batch_y = self.labels[batch_indices]

            # GPU-native random horizontal flip for training
            if self.is_train and torch.rand(1, device=self.device).item() > 0.5:
                batch_x = torch.flip(batch_x, dims=[3])

            yield batch_x, batch_y


def train_single_model(
    config: ModelConfig,
    train_gpu_data: PureGpuCIFAR10,
    test_gpu_data: PureGpuCIFAR10,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = "cuda",
) -> dict[str, float]:
    torch.manual_seed(config.seed)
    model = FlexibleCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device == "cuda"))

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_gpu_data.get_batches(batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Evaluation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_gpu_data.get_batches(batch_size=batch_size, shuffle=False):
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

    return {
        "final_accuracy": val_acc,
        "best_accuracy": best_acc,
        "final_val_loss": val_loss,
        "train_time_sec": time.time() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("measurements/ml/candidates_to_train_50.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("measurements/ml/candidate_accuracy_50_results.csv"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if not args.candidates_csv.exists():
        raise FileNotFoundError(f"Candidate file {args.candidates_csv} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Training Environment: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    df = pd.read_csv(args.candidates_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} candidate models ({args.epochs} epochs each, batch size {args.batch_size}).")

    print("Loading CIFAR-10 entirely into 24GB GPU VRAM (Zero CPU Bottleneck)...")
    train_raw, test_raw = load_data()
    train_gpu_data = PureGpuCIFAR10(train_raw, device=device, is_train=True)
    test_gpu_data = PureGpuCIFAR10(test_raw, device=device, is_train=False)
    print("VRAM Data transfer completed! GPU is ready at 100% throughput.")

    results = []
    total_start = time.time()

    for idx, row in df.iterrows():
        id_str = str(row["id"])
        cand_id = str(row["candidate_id"])
        reason = str(row["selection_reason"])
        cfg = ModelConfig(
            depth=int(row["depth"]),
            channels=parse_int_sequence(row["channels"]),
            pools=parse_int_sequence(row["pools"]),
            pattern=str(row["pattern"]),
            growth_pattern=str(row["growth_pattern"]),
            noise_ratio=float(row["noise_ratio"]),
            min_channels=int(row["min_channels"]),
            max_channels=int(row["max_channels"]),
            seed=int(row["seed"]),
            parameter_count=int(row["parameter_count"]),
        )

        print(f"[{idx+1:02d}/{len(df)}] Model {id_str} ({cand_id}, {reason}, Params: {cfg.parameter_count:,})...", end=" ", flush=True)
        metrics = train_single_model(
            cfg,
            train_gpu_data,
            test_gpu_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        print(f"Acc: {metrics['best_accuracy']:.2f}% ({metrics['train_time_sec']:.1f}s)")

        res_row = row.to_dict()
        res_row.update(metrics)
        results.append(res_row)

    total_time = time.time() - total_start
    res_df = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"\n==================================================")
    print(f"All 50 models trained in {total_time/60:.2f} minutes!")
    print(f"Results saved to: {args.output_csv}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
