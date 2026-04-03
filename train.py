import os
import torch
from torch.utils.data import DataLoader

from src.data.data_processing import CCSNetDataset, get_train_val_files, collect_stats
from src.models.model_fno import BaselineCNN


def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0

    for x, gas, pressure in loader:
        x = x.to(device)
        gas = gas.to(device)
        pressure = pressure.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            gas_pred, pressure_pred = model(x)

            loss_gas = torch.mean(torch.abs(gas_pred - gas))
            loss_pressure = torch.mean(torch.abs(pressure_pred - pressure))
            loss = loss_gas + loss_pressure

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    all_train_files, all_val_files = get_train_val_files()

    # Phase 1C: small-scale baseline run
    train_files = all_train_files[:10]
    val_files = all_val_files[:10]

    stats = collect_stats(train_files, max_files=200)

    train_dataset = CCSNetDataset(train_files, stats=stats)
    val_dataset = CCSNetDataset(val_files, stats=stats)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = BaselineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    best_val_loss = float("inf")

    for epoch in range(1, 101):
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                "checkpoints/best_baseline_phase1c.pt"
            )
            print("Saved new best validation model")

if __name__ == "__main__":
    main()