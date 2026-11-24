# Final Project Pipeline

```mermaid
flowchart TD
    A[NASA C-MAPSS FD001–FD004] --> B[Dataset audit and EDA]
    B --> C[Cycle-wise RUL label construction]
    C --> D[Causal 30-cycle feature extraction]
    D --> E[Engine-wise 80/20 validation split]
    E --> F1[Ridge Regression baseline]
    E --> F2[XGBoost baseline]
    F1 --> G[Validation and official test evaluation]
    F2 --> G
    G --> H[Sensor-noise sensitivity analysis]
    H --> I[10% Gaussian noise augmentation]
    I --> J[Robustness comparison]
    J --> K[Research discussion and future work]
```

## Implemented now
- Dataset inspection and EDA for all four subsets.
- RUL labels and 30-cycle causal rolling features.
- Ridge and XGBoost training on all four subsets.
- Engine-level validation split.
- Official test evaluation at the final observed cycle.
- Feature importance and per-engine prediction exports.
- Five-seed Gaussian sensor-noise sensitivity on FD001 and FD004.
- 10% noise-augmented XGBoost experiment with clean/noisy validation and test evaluation.

## Not yet implemented
- LSTM/GRU/TCN/Transformer benchmark.
- Operating-condition domain adaptation / cross-domain transfer.
- Predictive uncertainty calibration.
- N-CMAPSS external validation.
- Real-world engine maintenance cost model.
