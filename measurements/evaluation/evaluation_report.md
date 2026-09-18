# Candidate Models Prediction vs. Ground-Truth Evaluation

## 1. Accuracy Surrogate Model Evaluation
- **Evaluated Models**: 50 / 50
- **Pearson Correlation ($r$)**: 0.9391
- **$R^2$ Score**: 0.2880
- **MAE**: 7.24%
- **RMSE**: 9.67%
- **Interval Coverage [LCB, UCB]**: 26.0%

## 2. Energy Surrogate Model Evaluation
- **Evaluated Models**: 50 / 50
- **Pearson Correlation ($r$)**: 0.9867
- **$R^2$ Score**: 0.8950
- **MAE**: 2.756 mJ
- **MAPE**: 21.56%
- **Interval Coverage [LCB, UCB]**: 36.0%

## 3. Latency Surrogate Model Evaluation
- **Evaluated Models**: 50 / 50
- **Pearson Correlation ($r$)**: 0.9885
- **$R^2$ Score**: 0.9642
- **MAE**: 0.082 ms
- **MAPE**: 16.17%
- **Interval Coverage [LCB, UCB]**: 34.0%

## 4. Empirical Pareto Frontier Analysis
- **Ground-Truth Pareto Models Identified**: 12 out of 50
- **Models Selected via Predicted Pareto**: 20
- **Matched Ground-Truth Pareto**: 8
- **Pareto Hit Rate (Precision)**: 40.0%
- **Pareto Coverage (Recall)**: 66.7%

Detailed model-by-model evaluation saved to: `measurements\evaluation\candidate_evaluation_results.csv`