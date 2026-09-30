# Figure plan

## Full manuscript

1. **Fig. 1 - Study workflow.** Generated CNN geometry, untrained ONNX
   measurement, surrogate fitting, 50,000-candidate search, and final trained
   verification.
2. **Fig. 2 - Surrogate validation.** Held-out actual-versus-predicted energy,
   latency, and accuracy panels with sample size, MAE, and R-squared.
3. **Fig. 3 - Final Energy-Accuracy landscape and headline outcome.** Use
   `figures/final/Fig_A_pareto_headline.{png,pdf}`. Panel A shows the 50,000
   `search_2nd` candidates and C002793 in the search space; Panel B shows the
   trained 0551-versus-0530 measurement outcome.
4. **Fig. 4 - Pooling effects.** `figures/final/Fig_B_pooling_effect`.
5. **Fig. 5 - Untrained-trained validity.** Energy and latency correlation plus
   Bland-Altman panels, and TOST results after paired measurement.
6. **Fig. 6 - Headline comparison.** Trained 0551 vs trained 0530: energy and
   latency with trial variability, plus actual accuracy under the same protocol.

Use the pattern, depth, pool-ratio, latency-energy, zero-cost proxy, and
feature-importance plots as supplementary material unless a result is needed
to answer a specific reviewer question.

## Two-page abstract

Use only two composite figures:

- **Fig. A:** final Energy-Accuracy Pareto landscape plus the trained 0551 vs
  0530 comparison. Ready in `figures/final/Fig_A_pareto_headline`.
- **Fig. B:** adjusted pooling effects on energy and accuracy. Ready now.

## Claim wording currently supported

- Each additional pooling operation was associated with a 38.7% reduction in
  inference energy after adjustment (95% CI: 37.0% to 40.3% reduction; HC3
  p < .001; n = 550).
- Within the 100-architecture accuracy-labelled subset, each additional pool
  was associated with a 1.20 percentage-point higher accuracy (95% CI: 0.76 to
  1.65 pp; HC3 p < .001).

These are adjusted associations. Avoid causal wording and avoid generalizing
the accuracy result beyond the labelled subset.
