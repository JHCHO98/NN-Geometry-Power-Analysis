# Paper workspace

This directory is the single workspace for manuscript preparation, figures,
paper-specific analysis, and frozen result snapshots. Keep the measurement and
model-training pipeline at the repository root; do not duplicate raw HWiNFO
logs or ONNX files here.

## Layout

- `manuscripts/abstract/`: two-page abstract draft and official template.
- `manuscripts/full/`: full manuscript working DOCX and its current PDF export.
- `figures/final/`: publication-ready figures approved for insertion. Both PNG
  (Word) and PDF (vector) are kept when available.
- `figures/drafts/`: temporary or exploratory figure exports. Do not cite these.
- `results/`: small, frozen tables and JSON summaries supporting paper claims.
- `scripts/`: paper-specific analysis and figure-generation scripts.
- `notes/`: figure plan, captions, claim wording, and pending-measurement notes.

## Current evidence status

`figures/final/Fig_A_pareto_headline.{png,pdf}` and
`figures/final/Fig_B_pooling_effect.{png,pdf}` are ready for insertion.
Fig. A separates the search-stage Pareto landscape from the trained 0551 versus
0530 headline outcome; do not interpret the Panel A search coordinates as the
Panel B trained measurements. Fig. B reports adjusted associations, not causal
effects: energy uses all 550 measured structures, while accuracy uses the 100
architecture subset with an observed accuracy label. The source summaries are
frozen under `results/pooling_effects/`.

## Reproducible paper commands

Run these from the repository root:

```powershell
.\.venv\Scripts\python.exe paper\scripts\analyze_pooling_energy_effect.py
.\.venv\Scripts\python.exe paper\scripts\analyze_pooling_accuracy_effect.py
.\.venv\Scripts\python.exe paper\scripts\make_pooling_effect_figure.py
```

The first two commands refresh result snapshots; the final command recreates
Fig. B in `figures/final/`. New paper-only scripts, figures, tables, and drafts
belong under `paper/`.
