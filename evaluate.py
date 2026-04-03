import numpy as np
import torch

from src.models.fno_model import FNO2d
from src.data.data_processing import (
    collect_stats,
    get_train_val_files,
    normalize,
    build_coordinate_channels,
)


def r2(x, y):
    x = x.flatten()
    y = y.flatten()
    zx = (x - np.mean(x)) / np.std(x, ddof=1)
    zy = (y - np.mean(y)) / np.std(y, ddof=1)
    r = np.sum(zx * zy) / (len(x) - 1)
    return r ** 2


def MAE(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.mean(np.abs(x - y))


def build_input(data, stats):
    porosity = data["porosity"].astype(np.float32)
    perm_r = np.log10(data["perm_r"].astype(np.float32) + 1e-12)
    perm_z = np.log10(data["perm_z"].astype(np.float32) + 1e-12)

    nz, nx = porosity.shape

    inj_rate = np.full((nz, nx), data["inj_rate"], dtype=np.float32)
    temperature = np.full((nz, nx), data["temperature"], dtype=np.float32)
    depth = np.full((nz, nx), data["depth"], dtype=np.float32)
    swi = np.full((nz, nx), data["Swi"], dtype=np.float32)
    lam = np.full((nz, nx), data["lam"], dtype=np.float32)

    perf_mask = np.zeros((nz, nx), dtype=np.float32)
    z0, z1 = int(data["perf_interval"][0]), int(data["perf_interval"][1])
    perf_mask[z0:z1 + 1, 0] = 1.0

    z_channel, r_channel = build_coordinate_channels(nz, nx)

    porosity = normalize(porosity, stats["porosity"]["mean"], stats["porosity"]["std"])
    perm_r = normalize(perm_r, stats["perm_r"]["mean"], stats["perm_r"]["std"])
    perm_z = normalize(perm_z, stats["perm_z"]["mean"], stats["perm_z"]["std"])
    inj_rate = normalize(inj_rate, stats["inj_rate"]["mean"], stats["inj_rate"]["std"])
    temperature = normalize(temperature, stats["temperature"]["mean"], stats["temperature"]["std"])
    depth = normalize(depth, stats["depth"]["mean"], stats["depth"]["std"])
    swi = normalize(swi, stats["Swi"]["mean"], stats["Swi"]["std"])
    lam = normalize(lam, stats["lam"]["mean"], stats["lam"]["std"])

    x = np.stack(
        [
            porosity,
            perm_r,
            perm_z,
            inj_rate,
            temperature,
            depth,
            swi,
            lam,
            perf_mask,
            z_channel,
            r_channel,
        ],
        axis=0,
    )

    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_train_files, all_val_files = get_train_val_files()

    # Keep these aligned with the training run that produced fno_coords.pt
    train_files = all_train_files[:10]
    val_files = all_val_files[:10]

    stats = collect_stats(train_files, max_files=200)

    model = FNO2d(in_ch=11).to(device)
    model.load_state_dict(
        torch.load("checkpoints/fno_coords.pt", map_location=device)
    )
    model.eval()

    gas_saturation_r2, pressure_buildup_r2 = [], []
    gas_saturation_MAE, pressure_buildup_MAE = [], []

    for file_path in val_files:
        with np.load(file_path) as data:
            gas_saturation = data["gas_saturation"]
            pressure_buildup = data["pressure_buildup"]

            x = build_input(data, stats)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                gas_pred, pressure_pred = model(x_tensor)

            gas_pred = gas_pred.squeeze(0).cpu().numpy()
            pressure_pred = pressure_pred.squeeze(0).cpu().numpy()

            pressure_pred = (
                pressure_pred * stats["pressure"]["std"]
                + stats["pressure"]["mean"]
            )

            gas_saturation_pred = np.transpose(gas_pred, (1, 2, 0))
            pressure_buildup_pred = np.transpose(pressure_pred, (1, 2, 0))

            gas_saturation_r2.append(r2(gas_saturation, gas_saturation_pred))
            pressure_buildup_r2.append(r2(pressure_buildup, pressure_buildup_pred))

            gas_saturation_MAE.append(MAE(gas_saturation, gas_saturation_pred))
            pressure_buildup_MAE.append(MAE(pressure_buildup, pressure_buildup_pred))

    print("--------------")
    print(f"Average validation set gas saturation R2 score is: {np.mean(gas_saturation_r2)}")
    print(f"Average validation set pressure buildup R2 score is: {np.mean(pressure_buildup_r2)}")
    print("--------------")
    print(f"Average validation set gas saturation MAE is: {np.mean(gas_saturation_MAE)}")
    print(f"Average validation set pressure buildup MAE is: {np.mean(pressure_buildup_MAE)}")


if __name__ == "__main__":
    main()