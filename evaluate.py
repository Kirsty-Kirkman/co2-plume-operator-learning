import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.data_processing import CCSNetDataset, Normalizer
from src.models.fno_model import FNO3d
from src.utils.metrics import mae_score, r2_score


MODEL_DEFAULTS = {
    "target_z": 51,
    "model_width": 48,
    "modes_t": 8,
    "modes_z": 12,
    "modes_r": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained 3D FNO model.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", "dataset/test_data/*.npz"),
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
        default=0,
        help="Starting index in sorted data files for evaluation subset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of files to evaluate. Use <=0 to evaluate all files from start index.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to training config JSON. Defaults to <checkpoint_dir>/train_config.json.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Path to split manifest JSON. Defaults to <checkpoint_dir>/split_manifest.json.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow evaluation files to overlap with train/val split manifest.",
    )
    parser.add_argument(
        "--allow-config-override",
        action="store_true",
        help="Allow CLI model settings to differ from train_config.json values.",
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


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(args):
    if args.config:
        return Path(args.config)
    return Path(args.checkpoint).resolve().parent / "train_config.json"


def resolve_split_manifest_path(args):
    if args.split_manifest:
        return Path(args.split_manifest)
    return Path(args.checkpoint).resolve().parent / "split_manifest.json"


def apply_training_config(args, train_config):
    for key, default_value in MODEL_DEFAULTS.items():
        if key not in train_config:
            continue

        cfg_value = train_config[key]
        current_value = getattr(args, key)

        if current_value == default_value:
            setattr(args, key, cfg_value)
            continue

        if current_value != cfg_value and not args.allow_config_override:
            raise ValueError(
                f"Model setting '{key}' mismatch: eval={current_value}, train={cfg_value}. "
                "Pass --allow-config-override to force evaluation with mismatched settings."
            )


def find_overlap(eval_files, split_manifest):
    eval_names = {Path(f).name for f in eval_files}
    train_names = set(split_manifest.get("train_files", []))
    val_names = set(split_manifest.get("val_files", []))
    return sorted(eval_names & (train_names | val_names))


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

    config_path = resolve_config_path(args)
    if config_path.exists():
        try:
            train_config = load_json(config_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: failed to load training config at {config_path}: {exc}")
            return 1

        try:
            apply_training_config(args, train_config)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        print(f"Loaded training config from {config_path}")
    else:
        print(f"Warning: training config not found at {config_path}; using CLI/default model settings.")

    print(f"Using {len(eval_files)} files for evaluation.")

    split_manifest_path = resolve_split_manifest_path(args)
    if split_manifest_path.exists():
        try:
            split_manifest = load_json(split_manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: failed to load split manifest at {split_manifest_path}: {exc}")
            return 1

        overlap = find_overlap(eval_files, split_manifest)
        if overlap:
            preview = ", ".join(overlap[:10])
            if len(overlap) > 10:
                preview = f"{preview}, ..."
            msg = f"Detected {len(overlap)} eval files overlapping train/val split: {preview}"
            if args.allow_overlap:
                print(f"Warning: {msg}")
            else:
                print(f"Error: {msg}")
                print("Use --allow-overlap to bypass this safety check.")
                return 1
        else:
            print(f"Verified no train/val overlap using {split_manifest_path}")
    elif args.allow_overlap:
        print(f"Warning: split manifest not found at {split_manifest_path}; skipping overlap check.")
    else:
        print(f"Error: split manifest not found at {split_manifest_path}")
        print("Run train.py to generate split_manifest.json, or pass --allow-overlap to bypass.")
        return 1

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

        gas_r2_scores.append(r2_score(gas_pred, gas_true))
        pressure_r2_scores.append(r2_score(pressure_pred, pressure_true))
        gas_mae_scores.append(mae_score(gas_pred, gas_true))
        pressure_mae_scores.append(mae_score(pressure_pred, pressure_true))

    print("--------------")
    print(f"Average gas saturation R2: {np.mean(gas_r2_scores):.6f}")
    print(f"Average pressure buildup R2: {np.mean(pressure_r2_scores):.6f}")
    print("--------------")
    print(f"Average gas saturation MAE: {np.mean(gas_mae_scores):.6f}")
    print(f"Average pressure buildup MAE: {np.mean(pressure_mae_scores):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
