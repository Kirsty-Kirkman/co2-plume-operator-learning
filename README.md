# 3D Fourier Neural Operator for CO2 Plume Prediction

This repository trains a 3D Fourier Neural Operator (FNO) to predict gas saturation and pressure buildup from reservoir properties and injection conditions.

## Project Goal

Learn an operator that maps static geology + injection setup to full spatiotemporal plume dynamics.

## Current Status

- 3D FNO model implemented (`src/models/fno_model.py`)
- Preprocessing pipeline for heterogeneous grid depths (`src/data/data_processing.py`)
- Training, evaluation, and overfit smoke-test scripts
- Baseline CNN model included for comparisons (`src/models/baseline_cnn.py`)

## Repository Layout

- `src/data` - preprocessing, normalization, dataset assembly
- `src/models` - FNO and baseline CNN architectures
- `train.py` - configurable training entrypoint
- `evaluate.py` - configurable checkpoint evaluation
- `test_run.py` - short overfit smoke test (real or synthetic data)
- `results/` - archived experiment summaries

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Expectations

Scripts expect `.npz` files with fields used by `src/data/data_processing.py` (for example: `porosity`, `perm_r`, `perm_z`, `gas_saturation`, `pressure_buildup`, `perf_interval`).

Default data glob:

```text
dataset/train_data/*.npz
```

You can override this in each script with `--data-path` or `DATA_PATH`.

## Run

Train (default split: 4000 train / 500 val):

```bash
python train.py --data-path "dataset/train_data/*.npz"
```

Evaluate a checkpoint (default checkpoint: `checkpoints/fno_full_best.pt`):

```bash
python evaluate.py --data-path "dataset/train_data/*.npz" --start-index 4500 --num-samples 100
```

Run short smoke test (auto: real data if available, else synthetic):

```bash
python test_run.py --mode auto
```

## Notes

- `train.py` saves:
  - `checkpoints/fno_full_best.pt`
  - `checkpoints/fno_full_last.pt`
  - `checkpoints/normalizer.pkl`
  - `checkpoints/train_config.json`
- `evaluate.py` expects checkpoint architecture arguments to match training settings.

## Planned Improvements

- Stronger experiment tracking (metrics + run metadata aggregation)
- Dedicated benchmark script for CNN vs FNO parity
- More robust generalization splits and uncertainty/error diagnostics
