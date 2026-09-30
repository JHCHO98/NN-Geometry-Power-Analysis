# NN Geometry Power Analysis

This repository investigates how CNN geometry affects CPU inference energy on
edge-oriented hardware. Rather than treating parameter count as the only model
cost, the study varies convolutional depth, channel distribution, and pooling
placement, then measures their relationships with inference energy, latency,
and CIFAR-10 accuracy.

## Study overview

- **Architecture space:** generated CNNs using 3 x 3 convolutions and a fixed
  classifier structure; depth, channel pattern, and pooling geometry vary.
- **Measurement:** ONNX Runtime CPU inference is benchmarked while HWiNFO logs
  CPU Package Power. `analyze_energy.py` subtracts an idle baseline to report
  net energy per inference in joules and latency in milliseconds.
- **Surrogates and search:** XGBoost models predict Energy, Latency, and
  Accuracy, allowing 50,000 valid CNN candidates to be screened with an
  Energy-Accuracy Pareto criterion.
- **Validation:** paired trained/untrained ONNX benchmarks assess whether the
  large untrained measurement set preserves relative structural cost.

The current dataset contains energy and latency measurements for 550 CNN
architectures and CIFAR-10 accuracy labels for 100 architectures. Paper-ready
figures, manuscripts, and frozen analysis results are maintained in `paper/`.

## Repository layout

```text
FlexibleCNN.py                  CNN architecture definition
generate_dataset.py             Generate CNN metadata and ONNX models
benchmark_onnx.py               ONNX Runtime inference benchmark
run_production_benchmark.bat    Windows production measurement batch
analyze_energy.py               Integrate HWiNFO logs into net-energy trials

prepare_xgboost_dataset.py      Build the surrogate-training dataset
train_xgboost_models.py         Train Energy, Latency, and Accuracy surrogates
predict_candidates.py           Score generated search candidates
prepare_second_search.py        Combine predictions with known measurements
select_certified_pareto.py      Select mean Energy-Accuracy Pareto candidates
serve_pareto_explorer.py        Serve the local interactive Pareto explorer

measurements/                   Raw logs, processed trials, ML, and searches
paper/                          Manuscripts, final figures, tables, and scripts
model_onnx/                     Generated ONNX models (not versioned)
```

## Core workflow

Use the project virtual environment on Windows. Start HWiNFO logging before a
production benchmark and keep the measurement environment fixed.

```powershell
# 1. Generate models and measure an ID range.
.\.venv\Scripts\python.exe generate_dataset.py --count 10
.\run_production_benchmark.bat 1 10

# 2. Convert a HWiNFO log plus the benchmark run CSV into cumulative trials.
.\.venv\Scripts\python.exe analyze_energy.py `
  --hwinfolog measurements\raw_hwinf\production_1_10.csv `
  --benchmark-csv measurements\benchmark_runs\production_runs_1_10.csv

# 3. Prepare labels and train the surrogate models.
.\.venv\Scripts\python.exe prepare_xgboost_dataset.py
.\.venv\Scripts\python.exe train_xgboost_models.py
```

`analyze_energy.py` appends only new trial identities to processed outputs; do
not delete raw logs merely to force a rerun.

## Second-round Pareto search

The current final-search artifacts are isolated in `measurements/search_2nd/`.

```powershell
.\.venv\Scripts\python.exe predict_candidates.py `
  --output-csv measurements\search_2nd\candidate_predictions_raw.csv `
  --overwrite
.\.venv\Scripts\python.exe prepare_second_search.py
.\.venv\Scripts\python.exe select_certified_pareto.py `
  --predictions measurements\search_2nd\candidate_predictions.csv `
  --output-csv measurements\search_2nd\next_measurement_candidates.csv `
  --summary-json measurements\search_2nd\selection_summary.json `
  --output-html measurements\search_2nd\pareto_structures.html `
  --overwrite
```

Open the interactive explorer locally:

```powershell
.\.venv\Scripts\python.exe serve_pareto_explorer.py `
  --search-dir measurements\search_2nd `
  --predictions measurements\search_2nd\candidate_predictions.csv
```

## Current headline results

With the final trained ONNX comparison protocol (four trials per model), the
selected candidate C002793/0530 used 2.66% more parameters than the
VGG-inspired reference 0551 but reduced CPU inference energy by **52.23%**
and latency by **53.77%**, with an accuracy change of **-0.11 percentage
points**. See `measurements/headline_comparison/analysis/` and `paper/` for
the underlying summaries and publication figures.

## Reproducibility notes

- Preserve HWiNFO raw logs and benchmark CSVs. They are the source of all
  energy calculations.
- Keep power mode, CPU affinity, HWiNFO sensor configuration, warm-up,
  cooldown, and idle-baseline procedures consistent across sessions.
- Generated ONNX files, virtual environments, caches, and large raw logs are
  intentionally excluded from version control.
- Run `git diff --check` after code or documentation changes.
