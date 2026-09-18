# VGG-inspired reference model

The fixed comparison model is a VGG-inspired CNN, not an implementation of
the original VGG architecture. Its convolutional channel geometry is:

```text
64 → 64 → MaxPool → 128 → 128 → MaxPool → 256 → 256
```

It has six convolutional layers, MaxPool after blocks 2 and 4, and the same
classifier rule as the generated CNNs: `AdaptiveAvgPool2d(1) → Linear(256, 10)`.

## Create and validate the ONNX model

```powershell
.\.venv\Scripts\python.exe generate_reference_models.py
```

This creates `model_onnx/0551.onnx`, appends its structure to
`dataset_structure.csv`, and writes `reference_models.csv` in this directory.

## Train CIFAR-10 accuracy

Use the identical protocol used for the selected candidates:

```powershell
.\.venv\Scripts\python.exe train_candidate_accuracy.py `
  --candidates-csv measurements\reference_models\reference_models.csv `
  --output-csv measurements\reference_models\reference_accuracy_results.csv
```

## Later CPU measurement

When the measurement device is available, start HWiNFO logging and run the
normal benchmark protocol for ID 551. Keep its raw logs separate by using a
reference-specific run filename before calling `analyze_energy.py`.

```powershell
.\run_production_benchmark.bat 551 551
```
