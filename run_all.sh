#!/usr/bin/env bash
set -e
python src/run_eda.py --data-dir data/raw --output-dir reports/figures
python src/train_models.py --data-dir data/raw --output-dir reports --model-dir models --subset ALL
python src/train_noise_augmented.py --data-dir data/raw --model-dir models --output-dir reports --subsets FD001 FD004 --noise-std 0.10 --seeds 5
python src/evaluate_robustness.py --data-dir data/raw --model-dir models --output reports/robustness_results.json --subsets FD001 FD004 --seeds 5
python src/make_result_figures.py --reports reports
