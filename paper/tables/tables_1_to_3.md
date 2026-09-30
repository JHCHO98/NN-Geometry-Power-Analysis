# Tables 1 to 3

## Table 1. Dataset and measurement protocol

| Item | Value |
|---|---|
| CNN architectures with measured inference energy and latency | 550 |
| Architectures with measured CIFAR-10 accuracy | 100 |
| Candidate architectures evaluated by the surrogate | 50,000 |
| CNN geometry factors | Depth, channel distribution, pooling placement and count |
| Accuracy dataset | CIFAR-10 |
| Inference engine | ONNX Runtime on CPU |
| Energy source | HWiNFO power-log integration |
| Energy outcome | Inference energy (mJ per inference) |
| Latency outcome | Inference latency (ms per inference) |

**Note.** The 100 accuracy-labelled architectures are a targeted subset of the
550 architectures with energy and latency measurements.

## Table 2. Surrogate-model predictive performance

| Target | Split | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Inference energy (mJ) | Train | 3.652 | 9.166 | 0.963 |
|  | Validation | 7.786 | 21.632 | 0.876 |
|  | Test | 7.015 | 16.428 | 0.886 |
| Inference latency (ms) | Train | 0.106 | 0.319 | 0.991 |
|  | Validation | 0.218 | 0.675 | 0.968 |
|  | Test | 0.238 | 0.521 | 0.976 |
| CIFAR-10 accuracy (percentage points) | Train | 0.402 | 0.514 | 0.998 |
|  | Validation | 1.328 | 1.785 | 0.964 |
|  | Test | 2.459 | 3.452 | 0.902 |

**Note.** Energy MAE and RMSE were converted from J to mJ for readability.
Accuracy errors are reported in percentage points (pp).

## Table 3. Adjusted associations of pooling count with inference energy and accuracy

| Outcome | n | Effect per additional pool | 95% CI | HC3 p-value | Adjusted R² |
|---|---:|---:|---:|---:|---:|
| Log inference energy | 550 | −38.7% | −40.3% to −37.0% | < .001 | 0.874 |
| CIFAR-10 accuracy | 100 | +1.20 pp | +0.76 to +1.65 pp | < .001 | 0.909 |

**Note.** Ordinary least-squares models used HC3 robust standard errors and
adjusted for depth, log parameter count, channel mean, channel standard
deviation, last-to-first channel ratio, channel pattern, and growth pattern.
The accuracy analysis is limited to the 100-architecture labelled subset;
both effects are adjusted associations rather than causal estimates.
