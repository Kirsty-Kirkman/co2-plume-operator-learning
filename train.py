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


def train():
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    all_files = sorted(glob.glob("dataset/train_data/*.npz"))
    if len(all_files) == 0:
        print("Error: No .npz files found in dataset/train_data/")
        return

    target_z = 51

    # Overfit sanity check subset
    train_files = all_files[:8]
    print(f"Collecting statistics on {len(train_files)} files...")
    print(f"Resampling all samples to Z = {target_z}")

    normalizer = collect_stats(train_files, target_z=target_z)
    train_dataset = CCSNetDataset(train_files, normalizer=normalizer, target_z=target_z)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    sample_x, sample_gas, sample_pres = train_dataset[0]
    print("Sample x shape:", sample_x.shape)
    print("Sample gas shape:", sample_gas.shape)
    print("Sample pres shape:", sample_pres.shape)
    print("Sample x nan?:", torch.isnan(sample_x).any().item())
    print("Sample gas nan?:", torch.isnan(sample_gas).any().item())
    print("Sample pres nan?:", torch.isnan(sample_pres).any().item())

    model = FNO3d(in_ch=11, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    print("Starting Training Loop...")
    for epoch in range(1, 151):
        model.train()

        total_loss = 0.0
        total_gas_loss = 0.0
        total_pres_loss = 0.0
        g_r2_acc = 0.0
        p_r2_acc = 0.0

        for x, gas_true, pres_true in train_loader:
            x = x.to(device)
            gas_true = gas_true.to(device)
            pres_true = pres_true.to(device)

            optimizer.zero_grad()

            gas_pred, pres_pred = model(x)

            gas_loss = criterion(gas_pred, gas_true)
            pres_loss = criterion(pres_pred, pres_true)
            loss = gas_loss + pres_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_gas_loss += gas_loss.item()
            total_pres_loss += pres_loss.item()
            g_r2_acc += calculate_r2(gas_pred, gas_true)
            p_r2_acc += calculate_r2(pres_pred, pres_true)

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_gas_loss = total_gas_loss / len(train_loader)
        avg_pres_loss = total_pres_loss / len(train_loader)
        avg_g = g_r2_acc / len(train_loader)
        avg_p = p_r2_acc / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/fno3d_overfit_best.pt")
            normalizer.save("checkpoints/normalizer.pkl")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"LR: {current_lr:.6e} | "
                f"Loss: {avg_loss:.6f} | "
                f"GasLoss: {avg_gas_loss:.6f} | "
                f"PresLoss: {avg_pres_loss:.6f} | "
                f"Gas R2: {avg_g:.4f} | "
                f"Pres R2: {avg_p:.4f}"
            )

    torch.save(model.state_dict(), "checkpoints/fno3d_overfit_last.pt")
    normalizer.save("checkpoints/normalizer.pkl")
    print("Training complete.")
    print("Saved best model to checkpoints/fno3d_overfit_best.pt")
    print("Saved last model to checkpoints/fno3d_overfit_last.pt")
    print("Saved normalizer to checkpoints/normalizer.pkl")


if __name__ == "__main__":
    train()