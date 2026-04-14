# 3D Fourier Neural Operator for CO₂ Plume Prediction

This project implements a 3D Fourier Neural Operator (FNO) to learn the spatiotemporal evolution of CO₂ plumes in subsurface reservoirs.

## Problem
Predict gas saturation and pressure fields over time given geological properties and injection conditions.

## Approach
- 3D FNO (time + spatial dimensions)
- Multi-channel input (rock properties, coordinates, injection source)
- Resampling to unify heterogeneous grid resolutions
- Joint learning of gas saturation and pressure

## Results
Overfit validation on 8 samples:

- Gas R²: ~0.90  
- Pressure R²: ~0.95  

This confirms the model can learn the operator mapping.

## Structure
- `src/data` – preprocessing and normalization
- `src/models` – 3D FNO implementation
- `train.py` – training script
- `evaluate.py` – evaluation script

## Next Steps
- Train on full dataset
- Compare architectures (CNN vs FNO)
- Improve generalization performance