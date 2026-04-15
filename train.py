import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.data_processing import CCSNetDataset, collect_stats
from src.models.fno_model import FNO3d


def calculate_r2(pred, target):
    pred = pred.detach().cpu().numpy().reshape(-1)
    target = target.detach().cpu().numpy().reshape(-1)

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2) + 1e-8
    return 1.0 - (ss_res / ss_tot)


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

    if is_train:
        model.train()
    else:
        model.eval()

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
        total_gas_r2 += calculate_r2(gas_pred, gas_true)
        total_pres_r2 += calculate_r2(pres_pred, pres_true)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "gas_loss": total_gas_loss / n,
        "pres_loss": total_pres_loss / n,
        "gas_r2": total_gas_r2 / n,
        "pres_r2": total_pres_r2 / n,
    }


def train():
    print(">>> DEEPER FNO QUICK OVERFIT SANITY CHECK <<<", flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    data_path = os.getenv("DATA_PATH", "dataset/train_data/*.npz")
    all_files = sorted(glob.glob(data_path))

    print(f"Using data path: {data_path}", flush=True)
    print(f"Found {len(all_files)} files", flush=True)

    if len(all_files) < 12:
        print("Error: Need at least 12 files for 8/2/2 sanity-check split.", flush=True)
        return

    target_z = 51

    # Quick overfit sanity-check split
    train_files = all_files[:8]
    val_files = all_files[8:10]
    test_files = all_files[10:12]

    print(
        f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}",
        flush=True,
    )
    print(f"Resampling all samples to Z = {target_z}", flush=True)

    print("Fitting normalizer on training files...", flush=True)
    normalizer = collect_stats(train_files, target_z=target_z)
    print("Normalizer fitted.", flush=True)

    train_dataset = CCSNetDataset(train_files, normalizer=normalizer, target_z=target_z)
    val_dataset = CCSNetDataset(val_files, normalizer=normalizer, target_z=target_z)
    test_dataset = CCSNetDataset(test_files, normalizer=normalizer, target_z=target_z)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    sample_x, sample_gas, sample_pres = train_dataset[0]
    print("Sample x shape:", sample_x.shape, flush=True)
    print("Sample gas shape:", sample_gas.shape, flush=True)
    print("Sample pres shape:", sample_pres.shape, flush=True)
    print("Sample x nan?:", torch.isnan(sample_x).any().item(), flush=True)
    print("Sample gas nan?:", torch.isnan(sample_gas).any().item(), flush=True)
    print("Sample pres nan?:", torch.isnan(sample_pres).any().item(), flush=True)

    model = FNO3d(in_ch=11, width=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    best_val_loss = float("inf")
    pressure_weight = 0.2
    pressure_l1_weight = 0.1

    print("Starting Training Loop...", flush=True)
    for epoch in range(1, 31):
        train_metrics = run_epoch(
            model,
            train_loader,
            mse_criterion,
            l1_criterion,
            device,
            optimizer=optimizer,
            pressure_weight=pressure_weight,
            pressure_l1_weight=pressure_l1_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            mse_criterion,
            l1_criterion,
            device,
            optimizer=None,
            pressure_weight=pressure_weight,
            pressure_l1_weight=pressure_l1_weight,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "checkpoints/fno3d_deep_sanity_best.pt")
            normalizer.save("checkpoints/normalizer.pkl")
            print(f"New best validation model saved at epoch {epoch:03d}", flush=True)

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

    torch.save(model.state_dict(), "checkpoints/fno3d_deep_sanity_last.pt")
    normalizer.save("checkpoints/normalizer.pkl")

    print("Training complete.", flush=True)
    print("Saved best model to checkpoints/fno3d_deep_sanity_best.pt", flush=True)
    print("Saved last model to checkpoints/fno3d_deep_sanity_last.pt", flush=True)
    print("Saved normalizer to checkpoints/normalizer.pkl", flush=True)

    print("Loading best validation checkpoint for test evaluation...", flush=True)
    model.load_state_dict(torch.load("checkpoints/fno3d_deep_sanity_best.pt", map_location=device))

    test_metrics = run_epoch(
        model,
        test_loader,
        mse_criterion,
        l1_criterion,
        device,
        optimizer=None,
        pressure_weight=pressure_weight,
        pressure_l1_weight=pressure_l1_weight,
    )

    print("--------------", flush=True)
    print("Sanity-Check Test Results", flush=True)
    print("--------------", flush=True)
    print(f"Test Loss:     {test_metrics['loss']:.6f}", flush=True)
    print(f"Test GasLoss:  {test_metrics['gas_loss']:.6f}", flush=True)
    print(f"Test PresLoss: {test_metrics['pres_loss']:.6f}", flush=True)
    print(f"Test Gas R2:   {test_metrics['gas_r2']:.4f}", flush=True)
    print(f"Test Pres R2:  {test_metrics['pres_r2']:.4f}", flush=True)


if __name__ == "__main__":
    train()