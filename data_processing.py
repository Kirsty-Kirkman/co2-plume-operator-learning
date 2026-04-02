import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def collect_stats(files, max_files=200):
    """
    Compute simple global mean/std stats from a subset of training files.
    """
    files = files[:max_files]

    sums = {
        "porosity": 0.0,
        "perm_r": 0.0,
        "perm_z": 0.0,
        "inj_rate": 0.0,
        "temperature": 0.0,
        "depth": 0.0,
        "Swi": 0.0,
        "lam": 0.0,
        "pressure": 0.0,
    }
    sq_sums = {k: 0.0 for k in sums}
    counts = {k: 0 for k in sums}

    for file_path in files:
        with np.load(file_path) as data:
            porosity = data["porosity"].astype(np.float32)
            perm_r = np.log10(data["perm_r"].astype(np.float32) + 1e-12)
            perm_z = np.log10(data["perm_z"].astype(np.float32) + 1e-12)

            nz, nx = porosity.shape

            scalar_fields = {
                "inj_rate": np.full((nz, nx), data["inj_rate"], dtype=np.float32),
                "temperature": np.full((nz, nx), data["temperature"], dtype=np.float32),
                "depth": np.full((nz, nx), data["depth"], dtype=np.float32),
                "Swi": np.full((nz, nx), data["Swi"], dtype=np.float32),
                "lam": np.full((nz, nx), data["lam"], dtype=np.float32),
            }

            fields = {
                "porosity": porosity,
                "perm_r": perm_r,
                "perm_z": perm_z,
                **scalar_fields,
                "pressure": data["pressure_buildup"].astype(np.float32),
            }

            for key, arr in fields.items():
                sums[key] += arr.sum()
                sq_sums[key] += (arr ** 2).sum()
                counts[key] += arr.size

    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        var = sq_sums[key] / counts[key] - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        stats[key] = {"mean": float(mean), "std": float(std)}

    return stats


def normalize(arr, mean, std):
    return (arr - mean) / (std + 1e-8)


def load_sample(file_path, stats=None):
    with np.load(file_path) as data:
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

        if stats is not None:
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
            ],
            axis=0,
        )

        gas = data["gas_saturation"].astype(np.float32)
        pressure = data["pressure_buildup"].astype(np.float32)

        # channel-first for pytorch
        gas = np.transpose(gas, (2, 0, 1))         # (24, nz, nx)
        pressure = np.transpose(pressure, (2, 0, 1))

        if stats is not None:
            pressure = normalize(
                pressure,
                stats["pressure"]["mean"],
                stats["pressure"]["std"]
            )

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(gas, dtype=torch.float32),
            torch.tensor(pressure, dtype=torch.float32),
        )


class CCSNetDataset(Dataset):
    def __init__(self, files, stats=None):
        self.files = files
        self.stats = stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_sample(self.files[idx], stats=self.stats)


def get_train_val_files(data_dir="dataset/train_data", seed=42):
    all_files = sorted(glob.glob(f"{data_dir}/*.npz"))
    rng = random.Random(seed)
    rng.shuffle(all_files)

    train_files = all_files[:4000]
    val_files = all_files[4000:]

    return train_files, val_files