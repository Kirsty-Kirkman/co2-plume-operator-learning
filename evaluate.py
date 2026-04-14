import os
import glob
import numpy as np
import torch

from src.data.data_processing import CCSNetDataset, Normalizer
from src.models.fno_model import FNO3d


def r2(x, y):
    x = x.flatten()
    y = y.flatten()

    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0

    zx = (x - np.mean(x)) / x_std
    zy = (y - np.mean(y)) / y_std
    r = np.sum(zx * zy) / (len(x) - 1)
    return r ** 2


def MAE(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.mean(np.abs(x - y))


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    data_path = os.getenv("DATA_PATH", "dataset/train_data/*.npz")
    all_files = sorted(glob.glob(data_path))
    if len(all_files) == 0:
        print(f"Error: No .npz files found at {data_path}")
        return

    eval_files = all_files[100:150]
    if len(eval_files) == 0:
        print("Error: No evaluation files selected.")
        return

    print(f"Using {len(eval_files)} unseen files for evaluation.")

    normalizer = Normalizer.load("checkpoints/normalizer.pkl")

    ckpt = "checkpoints/fno3d_generalisation_best.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/fno3d_overfit_best.pt"

    model = FNO3d(in_ch=11, width=32).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    gas_saturation_r2 = []
    pressure_buildup_r2 = []
    gas_saturation_MAE = []
    pressure_buildup_MAE = []

    for file_path in eval_files:
        with np.load(file_path) as raw_data:
            gas_true_raw = raw_data["gas_saturation"].astype(np.float32)
            pres_true_raw = raw_data["pressure_buildup"].astype(np.float32)

        dataset = CCSNetDataset([file_path], normalizer=normalizer, target_z=51)
        x, _, pres_true_proc = dataset[0]

        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            gas_pred, pres_pred = model(x)

        gas_pred = gas_pred.squeeze(0).squeeze(0).cpu().numpy()   # [T, Z, R]
        pres_pred = pres_pred.squeeze(0).squeeze(0).cpu().numpy()

        pres_pred = normalizer.denormalize("pressure_buildup", pres_pred)

        gas_pred = np.transpose(gas_pred, (1, 2, 0))   # [Z, R, T]
        pres_pred = np.transpose(pres_pred, (1, 2, 0))

        if gas_true_raw.shape != gas_pred.shape or pres_true_raw.shape != pres_pred.shape:
            dataset = CCSNetDataset([file_path], normalizer=normalizer, target_z=51)
            _, gas_true_proc, pres_true_proc = dataset[0]

            gas_true_raw = gas_true_proc.squeeze(0).numpy()
            pres_true_proc = pres_true_proc.squeeze(0).numpy()

            pres_true_proc = normalizer.denormalize("pressure_buildup", pres_true_proc)

            gas_true_raw = np.transpose(gas_true_raw, (1, 2, 0))
            pres_true_raw = np.transpose(pres_true_proc, (1, 2, 0))

        gas_saturation_r2.append(r2(gas_true_raw, gas_pred))
        pressure_buildup_r2.append(r2(pres_true_raw, pres_pred))

        gas_saturation_MAE.append(MAE(gas_true_raw, gas_pred))
        pressure_buildup_MAE.append(MAE(pres_true_raw, pres_pred))

    print("--------------")
    print(f"Average gas saturation R2: {np.mean(gas_saturation_r2):.6f}")
    print(f"Average pressure buildup R2: {np.mean(pressure_buildup_r2):.6f}")
    print("--------------")
    print(f"Average gas saturation MAE: {np.mean(gas_saturation_MAE):.6f}")
    print(f"Average pressure buildup MAE: {np.mean(pressure_buildup_MAE):.6f}")


if __name__ == "__main__":
    evaluate()