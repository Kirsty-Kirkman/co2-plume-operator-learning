import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.data_processing import CCSNetDataset, Normalizer
from src.models.fno_model import FNO3d


def calculate_r2(pred, target):
    pred = pred.detach().cpu().numpy().reshape(-1)
    target = target.detach().cpu().numpy().reshape(-1)

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2) + 1e-8
    return 1.0 - (ss_res / ss_tot)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    test_files = sorted(glob.glob("dataset/test_data/*.npz"))
    if len(test_files) == 0:
        print("Error: No .npz files found in dataset/test_data/")
        return

    normalizer = Normalizer.load("checkpoints/normalizer.pkl")
    test_dataset = CCSNetDataset(test_files, normalizer=normalizer, target_z=51)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = FNO3d(in_ch=11, width=32).to(device)
    model.load_state_dict(torch.load("checkpoints/fno3d_overfit_best.pt", map_location=device))
    model.eval()

    criterion = nn.MSELoss()

    total_gas_loss = 0.0
    total_pres_loss = 0.0
    total_gas_r2 = 0.0
    total_pres_r2 = 0.0

    with torch.no_grad():
        for x, gas_true, pres_true in test_loader:
            x = x.to(device)
            gas_true = gas_true.to(device)
            pres_true = pres_true.to(device)

            gas_pred, pres_pred = model(x)

            gas_loss = criterion(gas_pred, gas_true)
            pres_loss = criterion(pres_pred, pres_true)

            total_gas_loss += gas_loss.item()
            total_pres_loss += pres_loss.item()
            total_gas_r2 += calculate_r2(gas_pred, gas_true)
            total_pres_r2 += calculate_r2(pres_pred, pres_true)

    n = len(test_loader)
    print(f"Gas MSE:  {total_gas_loss / n:.6f}")
    print(f"Pres MSE: {total_pres_loss / n:.6f}")
    print(f"Gas R2:   {total_gas_r2 / n:.4f}")
    print(f"Pres R2:  {total_pres_r2 / n:.4f}")


if __name__ == "__main__":
    evaluate()