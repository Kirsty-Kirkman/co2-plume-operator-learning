import argparse
import glob
import os

import numpy as np
import torch

from src.data.data_processing import CCSNetDataset, Normalizer
from src.models.fno_model import FNO3d


def r2(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2) + 1e-8
    return 1.0 - (ss_res / ss_tot)


def mae(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return np.mean(np.abs(target - pred))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained 3D FNO model.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", "dataset/train_data/*.npz"),
        help="Glob pattern for evaluation data files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fno_full_best.pt",
        help="Path to model checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--normalizer",
        default="checkpoints/normalizer.pkl",
        help="Path to normalizer file produced by train.py.",
    )
    parser.add_argument("--target-z", type=int, default=51, help="Target Z resolution used in preprocessing.")
    parser.add_argument("--model-width", type=int, default=48, help="FNO channel width.")
    parser.add_argument("--modes-t", type=int, default=8, help="FNO Fourier modes along time axis.")
    parser.add_argument("--modes-z", type=int, default=12, help="FNO Fourier modes along Z axis.")
    parser.add_argument("--modes-r", type=int, default=12, help="FNO Fourier modes along R axis.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=4500,
        help="Starting index in sorted data files for evaluation subset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of files to evaluate. Use <=0 to evaluate all files from start index.",
    )
    return parser.parse_args()


def select_eval_files(all_files, start_index, num_samples):
    start_index = max(0, start_index)
    if start_index >= len(all_files):
        return []
    if num_samples <= 0:
        return all_files[start_index:]
    end_index = min(len(all_files), start_index + num_samples)
    return all_files[start_index:end_index]


def evaluate():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    all_files = sorted(glob.glob(args.data_path))
    if not all_files:
        print(f"Error: no .npz files found at {args.data_path}")
        return 1

    eval_files = select_eval_files(all_files, args.start_index, args.num_samples)
    if not eval_files:
        print(
            f"Error: no evaluation files selected from index {args.start_index}. "
            f"Found only {len(all_files)} files."
        )
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    if not os.path.exists(args.normalizer):
        print(f"Error: normalizer not found: {args.normalizer}")
        return 1

    print(f"Using {len(eval_files)} files for evaluation.")
    normalizer = Normalizer.load(args.normalizer)

    model = FNO3d(
        in_ch=11,
        width=args.model_width,
        modes_t=args.modes_t,
        modes_z=args.modes_z,
        modes_r=args.modes_r,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    gas_r2_scores = []
    pressure_r2_scores = []
    gas_mae_scores = []
    pressure_mae_scores = []

    for file_path in eval_files:
        dataset = CCSNetDataset([file_path], normalizer=normalizer, target_z=args.target_z)
        x, gas_true, pressure_true = dataset[0]

        x = x.unsqueeze(0).to(device)
        gas_true = gas_true.squeeze(0).numpy()
        pressure_true = pressure_true.squeeze(0).numpy()
        pressure_true = normalizer.denormalize("pressure_buildup", pressure_true)

        with torch.no_grad():
            gas_pred, pressure_pred = model(x)

        gas_pred = gas_pred.squeeze(0).squeeze(0).cpu().numpy()
        pressure_pred = pressure_pred.squeeze(0).squeeze(0).cpu().numpy()
        pressure_pred = normalizer.denormalize("pressure_buildup", pressure_pred)

        gas_r2_scores.append(r2(gas_pred, gas_true))
        pressure_r2_scores.append(r2(pressure_pred, pressure_true))
        gas_mae_scores.append(mae(gas_pred, gas_true))
        pressure_mae_scores.append(mae(pressure_pred, pressure_true))

    print("--------------")
    print(f"Average gas saturation R2: {np.mean(gas_r2_scores):.6f}")
    print(f"Average pressure buildup R2: {np.mean(pressure_r2_scores):.6f}")
    print("--------------")
    print(f"Average gas saturation MAE: {np.mean(gas_mae_scores):.6f}")
    print(f"Average pressure buildup MAE: {np.mean(pressure_mae_scores):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
