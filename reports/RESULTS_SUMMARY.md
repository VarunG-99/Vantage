# Results Summary

## Baseline XGBoost: official test engines, raw supplied RUL

| Subset | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| FD001 | 12.20 | 16.43 | 0.844 |
| FD002 | 20.37 | 28.23 | 0.724 |
| FD003 | 12.66 | 17.62 | 0.819 |
| FD004 | 22.39 | 29.71 | 0.703 |

The training target is capped at 125 cycles. Supplementary capped-target results are in `benchmark_results.csv`.

## FD004 noise robustness: validation, MAE in cycles

| Gaussian noise (% of training sensor SD) | Baseline | 10% noise-augmented |
|---:|---:|---:|
| 0% | 14.01 | 15.65 |
| 5% | 28.37 | 16.73 |
| 10% | 30.69 | 17.23 |
| 20% | 33.56 | 18.40 |

This demonstrates a clean-accuracy versus corruption-robustness trade-off under the tested synthetic Gaussian noise protocol.

## FD004 noise robustness: official test, MAE in cycles

| Gaussian noise (% of training sensor SD) | Baseline | 10% noise-augmented |
|---:|---:|---:|
| 0% | 22.39 | 25.80 |
| 5% | 38.84 | 27.87 |
| 10% | 41.11 | 28.50 |
| 20% | 43.32 | 29.72 |

Noise augmentation is a preliminary candidate contribution. Its novelty is not established; compare it with prior robustness and augmentation work before claiming a research gap.
