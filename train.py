import torch
from torch.utils.data import DataLoader

from data_processing import CCSNetDataset, get_train_val_files, collect_stats
from model_fno import DummyModel


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

            # weight them equally for now
            loss = loss_gas + loss_pressure

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    train_files, _ = get_train_val_files()

    # overfit test: 10 samples only
    overfit_files = train_files[:10]

    # compute normalization stats from these 10 files for the overfit test
    stats = collect_stats(overfit_files, max_files=10)

    train_dataset = CCSNetDataset(overfit_files, stats=stats)

    # must stay 1 because vertical size varies across samples
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = DummyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Overfit samples:", len(train_dataset))

    best_loss = float("inf")

        for epoch in range(1, 51):

        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss

            torch.save(
                model.state_dict(),
                "best_dummy_overfit.pt"
            )

            print("Saved new best model")


if __name__ == "__main__":
    main()