"""Train selected candidate CNNs on CIFAR-10 with 100% identical protocol to Colab training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import onnx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.v2 as v2

from FlexibleCNN import FlexibleCNN, ModelConfig
from load_data import load_data


torch.backends.cudnn.benchmark = True


def parse_int_sequence(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("-")]


class GpuCIFAR10ExactProtocol:
    """Exact reproduction of Colab CIFAR-10 augmentation & normalization on GPU.
    
    Transforms (100% identical to load_data.py & train_sampled_accuracy.py):
      - RandomCrop(32, padding=4)
      - RandomHorizontalFlip(p=0.5)
      - Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    """

    def __init__(self, data_dict: dict, device: str = "cuda", is_train: bool = True):
        self.device = device
        self.is_train = is_train

        # Shape: (N, 3, 32, 32), uint8 -> float32 on GPU normalized to [0, 1]
        self.raw_data = torch.from_numpy(data_dict["data"]).float().to(device) / 255.0
        self.labels = torch.tensor(data_dict["labels"], dtype=torch.long, device=device)
        self.num_samples = len(self.labels)

        # Exact transforms using Torchvision V2 GPU-native operators
        if is_train:
            self.transform = torch.nn.Sequential(
                v2.Pad(4, padding_mode="constant", fill=0.0),
                v2.RandomCrop(32),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            )
        else:
            self.transform = v2.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            )

    def get_batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)

        for start_idx in range(0, self.num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_x = self.raw_data[batch_indices]
            batch_y = self.labels[batch_indices]

            # Apply identical transform on GPU batch
            batch_x = self.transform(batch_x)
            yield batch_x, batch_y


def train_single_model(
    config: ModelConfig,
    train_gpu_data: GpuCIFAR10ExactProtocol,
    test_gpu_data: GpuCIFAR10ExactProtocol,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cuda",
) -> tuple[dict[str, float | int], dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    model = FlexibleCNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device == "cuda"))

    start_time = time.time()
    best_acc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

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
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training ended without a validation checkpoint.")

    return {
        "final_accuracy": val_acc,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_loss": val_loss,
        "train_time_sec": time.time() - start_time,
    }, best_state


def save_trained_artifacts(
    config: ModelConfig,
    model_id: str,
    best_state: dict[str, torch.Tensor],
    metrics: dict[str, float | int],
    artifact_dir: Path,
    training_settings: dict[str, float | int],
) -> dict[str, str | int]:
    """Save the best validation checkpoint and an equivalent static ONNX model."""
    checkpoint_dir = artifact_dir / "checkpoints"
    onnx_dir = artifact_dir / "onnx"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{model_id}_best.pt"
    onnx_path = onnx_dir / f"{model_id}_trained_best.onnx"
    if checkpoint_path.exists() or onnx_path.exists():
        raise FileExistsError(
            "Refusing to overwrite trained artifacts for model "
            f"{model_id}. Use a new --trained-artifact-dir or remove the complete prior run."
        )

    checkpoint = {
        "model_id": model_id,
        "model_config": asdict(config),
        "model_state_dict": best_state,
        "selection_metric": "best_accuracy",
        "best_accuracy": metrics["best_accuracy"],
        "best_epoch": metrics["best_epoch"],
        "training_settings": training_settings,
    }
    torch.save(checkpoint, checkpoint_path)

    export_model = FlexibleCNN(config).cpu().eval()
    export_model.load_state_dict(best_state)
    dummy_input = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    return {
        "trained_checkpoint_path": checkpoint_path.as_posix(),
        "trained_onnx_path": onnx_path.as_posix(),
        "trained_onnx_size_bytes": onnx_path.stat().st_size,
        "checkpoint_selection_metric": "best_accuracy",
    }


def normalized_model_id(value: object) -> str:
    """Return the four-digit ID shared by ONNX, checkpoints, and measurement logs."""
    return f"{int(value):04d}"


def write_results(results: list[dict], output_csv: Path) -> None:
    """Persist completed rows so an interrupted long training run can resume."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")


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
    parser.add_argument("--batch-size", type=int, default=128)  # Exact match to Colab: 128
    parser.add_argument("--lr", type=float, default=0.001)       # Exact match to Colab: 0.001
    parser.add_argument(
        "--trained-artifact-dir",
        type=Path,
        default=None,
        help=(
            "When provided, save each best-validation checkpoint and a matching trained ONNX "
            "model under this directory. Existing artifact names are never overwritten."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed model IDs already present in --output-csv.",
    )
    args = parser.parse_args()

    if not args.candidates_csv.exists():
        raise FileNotFoundError(f"Candidate file {args.candidates_csv} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Training Environment: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: 100% Exact Match to Colab (RandomCrop Pad=4, HorizontalFlip, Normalize, Batch=128)")
    print(f"==================================================")

    df = pd.read_csv(args.candidates_csv, encoding="utf-8-sig")
    if "id" not in df.columns:
        raise ValueError("Candidate CSV must contain an 'id' column from dataset_structure.csv.")
    print(f"Loaded {len(df)} candidate models ({args.epochs} epochs each, batch size {args.batch_size}).")

    train_raw, test_raw = load_data()
    train_gpu_data = GpuCIFAR10ExactProtocol(train_raw, device=device, is_train=True)
    test_gpu_data = GpuCIFAR10ExactProtocol(test_raw, device=device, is_train=False)
    print("Exact protocol data loaders ready.")

    results: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume and args.output_csv.exists():
        existing = pd.read_csv(args.output_csv, encoding="utf-8-sig")
        if "id" not in existing.columns:
            raise ValueError(f"Cannot resume: {args.output_csv} has no 'id' column.")
        results = existing.to_dict("records")
        completed_ids = {normalized_model_id(value) for value in existing["id"]}
        print(f"Resuming: keeping {len(completed_ids)} completed result rows.")
    total_start = time.time()

    for idx, row in df.iterrows():
        id_str = normalized_model_id(row["id"])
        if id_str in completed_ids:
            print(f"[{idx+1:02d}/{len(df)}] Model {id_str} already completed; skipping.")
            continue
        cand_id = str(row.get("candidate_id", id_str))
        reason = str(row.get("selection_reason", "accuracy_training"))
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
        metrics, best_state = train_single_model(
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
        res_row["id"] = id_str
        res_row.update(metrics)
        if args.trained_artifact_dir is not None:
            artifact_fields = save_trained_artifacts(
                cfg,
                id_str,
                best_state,
                metrics,
                args.trained_artifact_dir,
                {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": 1e-4,
                    "protocol_version": 1,
                },
            )
            res_row.update(artifact_fields)
            print(f"  Saved checkpoint and trained ONNX for {id_str}.")
        results.append(res_row)
        write_results(results, args.output_csv)

    total_time = time.time() - total_start
    write_results(results, args.output_csv)

    print(f"\n==================================================")
    print(f"Training run completed in {total_time/60:.2f} minutes.")
    print(f"Results saved to: {args.output_csv}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
