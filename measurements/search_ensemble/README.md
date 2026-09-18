# Distance-weighted ensemble Pareto search

This directory keeps ensemble results separate from both `search/` (first-round
surrogate) and `search_2nd/` (second-round surrogate). The ensemble retains the
50 measured candidates as actual Pareto anchors. For all unmeasured candidates,
it blends the first-round global prediction and the second-round local prediction.

The second-round weight is higher for candidates structurally close to the 50 new
measurements. Distance uses depth, pool count, log parameter count, pattern, and
growth pattern. The defaults use a second-round weight from 0.25 (far away) to
0.85 (nearby); they are recorded in `ensemble_summary.json`.

```powershell
.\.venv\Scripts\python.exe ensemble_candidate_predictions.py --overwrite

.\.venv\Scripts\python.exe select_certified_pareto.py `
  --predictions measurements\search_ensemble\candidate_predictions.csv `
  --output-csv measurements\search_ensemble\next_measurement_candidates.csv `
  --summary-json measurements\search_ensemble\selection_summary.json `
  --output-html measurements\search_ensemble\pareto_structures.html `
  --overwrite

.\.venv\Scripts\python.exe serve_pareto_explorer.py `
  --search-dir measurements\search_ensemble `
  --predictions measurements\search_ensemble\candidate_predictions.csv
```

The selector excludes the 50 already measured anchors from the next measurement
list but includes them in every Pareto calculation.
