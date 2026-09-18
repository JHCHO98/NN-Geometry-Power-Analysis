# Second-round search outputs

This directory preserves the second surrogate-guided search separately from
the original `measurements/search/` results.

Run the following after retraining the surrogate models:

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

`prepare_second_search.py` replaces the predictions for models 0501–0550 with
their measured accuracy, energy, and latency. They remain Pareto anchors but
are automatically excluded from `next_measurement_candidates.csv`.

View the explorer with:

```powershell
.\.venv\Scripts\python.exe serve_pareto_explorer.py `
  --search-dir measurements\search_2nd `
  --predictions measurements\search_2nd\candidate_predictions.csv
```
