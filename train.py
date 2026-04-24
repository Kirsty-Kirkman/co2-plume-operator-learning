import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.data_processing import CCSNetDataset, collect_stats
from src.models.fno_model import FNO3d
from src.utils.metrics import r2_score


def run_epoch(
    model,
    loader,
    mse_criterion,
    l1_criterion,
    device,
    optimizer=None,
    pressure_weight=0.2,
    pressure_l1_weight=0.1,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_gas_loss = 0.0
    total_pres_loss = 0.0
    total_gas_r2 = 0.0
    total_pres_r2 = 0.0

    for x, gas_true, pres_true in loader:
        x = x.to(device)
        gas_true = gas_true.to(device)
        pres_true = pres_true.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            gas_pred, pres_pred = model(x)

            gas_loss = mse_criterion(gas_pred, gas_true)

            pres_mse = mse_criterion(pres_pred, pres_true)
            pres_l1 = l1_criterion(pres_pred, pres_true)
            pres_loss = pres_mse + pressure_l1_weight * pres_l1

            loss = gas_loss + pressure_weight * pres_loss

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_gas_loss += gas_loss.item()
        total_pres_loss += pres_loss.item()
        total_gas_r2 += r2_score(gas_pred, gas_true)
        total_pres_r2 += r2_score(pres_pred, pres_true)

    n = len(loader)
    if n == 0:
        raise ValueError("Received an empty data loader.")
    return {
        "loss": total_loss / n,
        "gas_loss": total_gas_loss / n,
        "pres_loss": total_pres_loss / n,
        "gas_r2": total_gas_r2 / n,
        "pres_r2": total_pres_r2 / n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D FNO model for CO2 plume prediction.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", "dataset/train_data/*.npz"),
        help="Glob pattern for training .npz files.",
    )
    parser.add_argument("--split-offset", type=int, default=0, help="Start index for train/val split.")
    parser.add_argument("--train-size", type=int, default=4000, help="Number of training samples.")
    parser.add_argument("--val-size", type=int, default=500, help="Number of validation samples.")
    parser.add_argument("--target-z", type=int, default=51, help="Target Z resolution for resampling.")
    parser.add_argument(
        "--normalizer-samples",
        type=int,
        default=500,
        help="Number of training files used to fit normalization stats.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training/validation batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--lr-step-size", type=int, default=10, help="StepLR step size.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR gamma.")
    parser.add_argument("--model-width", type=int, default=48, help="FNO channel width.")
    parser.add_argument("--modes-t", type=int, default=8, help="FNO Fourier modes along time axis.")
    parser.add_argument("--modes-z", type=int, default=12, help="FNO Fourier modes along Z axis.")
    parser.add_argument("--modes-r", type=int, default=12, help="FNO Fourier modes along R axis.")
    parser.add_argument("--pressure-weight", type=float, default=0.2, help="Weight for pressure loss.")
    parser.add_argument(
        "--pressure-l1-weight",
        type=float,
        default=0.1,
        help="L1 regularization term inside pressure loss.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=2, help="Print metrics every N epochs.")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory where model weights and normalizer are saved.",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    print(">>> TRAINING 3D FNO <<<", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    all_files = sorted(glob.glob(args.data_path))
    print(f"Using data path: {args.data_path}", flush=True)
    print(f"Found {len(all_files)} files", flush=True)

    split_start = max(0, args.split_offset)
    train_end = split_start + args.train_size
    val_end = train_end + args.val_size
    if val_end > len(all_files):
        print(f"Error: not enough files for requested split: need {val_end}, found {len(all_files)}.")
        return 1

    train_files = all_files[split_start:train_end]
    val_files = all_files[train_end:val_end]
    train_file_names = [Path(f).name for f in train_files]
    val_file_names = [Path(f).name for f in val_files]

    print(
        f"Train: {len(train_files)} | Val: {len(val_files)} | Target Z: {args.target_z}",
        flush=True,
    )

    normalizer_file_count = min(max(1, args.normalizer_samples), len(train_files))
    print(f"Fitting normalizer on {normalizer_file_count} files...", flush=True)
    normalizer = collect_stats(
        train_files,
        target_z=args.target_z,
        max_files=normalizer_file_count,
    )
    print("Normalizer fitted.", flush=True)

    train_dataset = CCSNetDataset(train_files, normalizer=normalizer, target_z=args.target_z)
    val_dataset = CCSNetDataset(val_files, normalizer=normalizer, target_z=args.target_z)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample_x, sample_gas, sample_pres = train_dataset[0]
    print("Sample x shape:", tuple(sample_x.shape), flush=True)
    print("Sample gas shape:", tuple(sample_gas.shape), flush=True)
    print("Sample pres shape:", tuple(sample_pres.shape), flush=True)
    print("Sample x nan?:", torch.isnan(sample_x).any().item(), flush=True)
    print("Sample gas nan?:", torch.isnan(sample_gas).any().item(), flush=True)
    print("Sample pres nan?:", torch.isnan(sample_pres).any().item(), flush=True)

    model = FNO3d(
        in_ch=11,
        width=args.model_width,
        modes_t=args.modes_t,
        modes_z=args.modes_z,
        modes_r=args.modes_r,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    best_val_loss = float("inf")
    best_ckpt = checkpoint_dir / "fno_full_best.pt"
    last_ckpt = checkpoint_dir / "fno_full_last.pt"
    normalizer_path = checkpoint_dir / "normalizer.pkl"
    config_path = checkpoint_dir / "train_config.json"

    config_payload = vars(args).copy()
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"Saved training config to {config_path}", flush=True)

    split_manifest_path = checkpoint_dir / "split_manifest.json"
    split_manifest = {
        "data_path": args.data_path,
        "split_offset": split_start,
        "train_size": len(train_files),
        "val_size": len(val_files),
        "train_files": train_file_names,
        "val_files": val_file_names,
    }
    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Saved split manifest to {split_manifest_path}", flush=True)

    print("Starting Training Loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            mse_criterion,
            l1_criterion,
            device,
            optimizer=optimizer,
            pressure_weight=args.pressure_weight,
            pressure_l1_weight=args.pressure_l1_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            mse_criterion,
            l1_criterion,
            device,
            optimizer=None,
            pressure_weight=args.pressure_weight,
            pressure_l1_weight=args.pressure_l1_weight,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_ckpt)
            normalizer.save(normalizer_path)
            print(f"New best validation model saved at epoch {epoch:03d}", flush=True)

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"LR: {current_lr:.6e} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Train Gas R2: {train_metrics['gas_r2']:.4f} | "
                f"Train Pres R2: {train_metrics['pres_r2']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val Gas R2: {val_metrics['gas_r2']:.4f} | "
                f"Val Pres R2: {val_metrics['pres_r2']:.4f}",
                flush=True,
            )

    torch.save(model.state_dict(), last_ckpt)
    normalizer.save(normalizer_path)

    print("Training complete.", flush=True)
    print(f"Saved best model to {best_ckpt}", flush=True)
    print(f"Saved last model to {last_ckpt}", flush=True)
    print(f"Saved normalizer to {normalizer_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
