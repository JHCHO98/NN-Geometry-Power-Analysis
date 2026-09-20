# Weight-State Paired Validation

This directory keeps the trained/untrained comparison reproducible. One
`pair_id` represents one CNN structure and must have exactly two ONNX rows in
`pair_manifest.csv`:

- `untrained_seed_reconstructed`: deterministic reconstruction from the saved seed.
- `trained_best_accuracy`: the ONNX exported from the selected trained checkpoint.

The base sample has 25 stratified structures from IDs 1-500. Add the required
0551 reference and C002793/0530 final candidate to form the 27-structure
validation set. Use one continuous HWiNFO log per `benchmark_session_id`; do
not split raw logs by weight state. State identity, order, and protocol are
recorded in the matching benchmark CSV.

On the measurement PC, prepare the headline registry, copy the trained ONNX
files, and rebuild the manifest. The default one ABBA block records two trials
per state. Use two blocks for four trials per state, with ABBA followed by
BAAB and an idle interval between blocks:

```powershell
 .\.venv\Scripts\python.exe prepare_headline_comparison.py --require-artifacts
 .\.venv\Scripts\python.exe build_weight_state_pair_manifest.py --include-headline --require-artifacts
.\.venv\Scripts\python.exe run_weight_state_paired_benchmark.py `
  --benchmark-session-id wsval_20260920_s01 `
  --result-csv measurements\benchmark_runs\weight_state_validation_wsval_20260920_s01.csv `
  --cpu-core 0 --high-priority --abba-blocks 2
```

Start HWiNFO logging immediately before the runner starts. Save its raw CSV as
`measurements/raw_hwinf/weight_state_validation_wsval_20260920_s01.csv`.
Afterward, integrate energy and run the paired analysis:

```powershell
.\.venv\Scripts\python.exe analyze_energy.py `
  --hwinfolog measurements\raw_hwinf\weight_state_validation_wsval_20260920_s01.csv `
  --benchmark-csv measurements\benchmark_runs\weight_state_validation_wsval_20260920_s01.csv `
  --output-dir measurements\validation_weight_state\processed
.\.venv\Scripts\python.exe analyze_weight_state_validation.py `
  --benchmark-session-id wsval_20260920_s01
```

The final command writes paired means, correlation, Bland-Altman plots, and
TOST equivalence results under `analysis/`.
