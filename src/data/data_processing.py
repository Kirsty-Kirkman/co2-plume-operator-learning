import numpy as np
import torch
from torch.utils.data import Dataset


def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)


def resample_2d_z(field, target_z):
    """
    Resample a 2D field from [Z, R] to [target_z, R] along the Z axis.
    """
    old_z, r = field.shape
    if old_z == target_z:
        return field.astype(np.float32)

    old_coords = np.linspace(0.0, 1.0, old_z, dtype=np.float32)
    new_coords = np.linspace(0.0, 1.0, target_z, dtype=np.float32)

    out = np.empty((target_z, r), dtype=np.float32)
    for j in range(r):
        out[:, j] = np.interp(new_coords, old_coords, field[:, j])

    return out


def resample_3d_z(field, target_z):
    """
    Resample a 3D field from [Z, R, T] to [target_z, R, T] along the Z axis.
    """
    old_z, r, t = field.shape
    if old_z == target_z:
        return field.astype(np.float32)

    old_coords = np.linspace(0.0, 1.0, old_z, dtype=np.float32)
    new_coords = np.linspace(0.0, 1.0, target_z, dtype=np.float32)

    out = np.empty((target_z, r, t), dtype=np.float32)
    for j in range(r):
        for k in range(t):
            out[:, j, k] = np.interp(new_coords, old_coords, field[:, j, k])

    return out


def rescale_perf_interval(perf_interval, old_z, target_z):
    """
    Rescale perf_interval from old Z resolution to target Z resolution.
    """
    start, end = int(perf_interval[0]), int(perf_interval[1])

    start_scaled = int(round(start * target_z / old_z))
    end_scaled = int(round(end * target_z / old_z))

    start_scaled = max(0, min(target_z - 1, start_scaled))
    end_scaled = max(start_scaled + 1, min(target_z, end_scaled))

    return start_scaled, end_scaled


def collect_stats(files, target_z=51):
    """
    Compute mean/std for each required field after resampling to a common grid.
    """
    stats = {}
    keys = [
        "porosity",
        "perm_r",
        "perm_z",
        "inj_rate",
        "temperature",
        "depth",
        "Swi",
        "lam",
        "pressure_buildup",
    ]
    data_map = {k: [] for k in keys}

    for f in files[:200]:
        with np.load(f) as d:
            # resample spatial fields
            porosity = resample_2d_z(d["porosity"], target_z)
            perm_r = resample_2d_z(d["perm_r"], target_z)
            perm_z = resample_2d_z(d["perm_z"], target_z)
            pressure = resample_3d_z(d["pressure_buildup"], target_z)

            data_map["porosity"].append(porosity.flatten())
            data_map["perm_r"].append(perm_r.flatten())
            data_map["perm_z"].append(perm_z.flatten())
            data_map["pressure_buildup"].append(pressure.flatten())

            # scalars
            data_map["inj_rate"].append(np.asarray(d["inj_rate"]).reshape(-1))
            data_map["temperature"].append(np.asarray(d["temperature"]).reshape(-1))
            data_map["depth"].append(np.asarray(d["depth"]).reshape(-1))
            data_map["Swi"].append(np.asarray(d["Swi"]).reshape(-1))
            data_map["lam"].append(np.asarray(d["lam"]).reshape(-1))

    for k in keys:
        all_vals = np.concatenate(data_map[k])
        stats[k] = {
            "mean": np.mean(all_vals),
            "std": np.std(all_vals),
        }

    return stats


def load_sample(file_path, stats, target_z=51):
    with np.load(file_path) as data:
        old_z, r = data["porosity"].shape

        # Resample spatial fields to common Z resolution
        porosity_raw = resample_2d_z(data["porosity"], target_z)
        perm_r_raw = resample_2d_z(data["perm_r"], target_z)
        perm_z_raw = resample_2d_z(data["perm_z"], target_z)

        gas_raw = resample_3d_z(data["gas_saturation"], target_z)
        pressure_raw = resample_3d_z(data["pressure_buildup"], target_z)

        # Normalize static fields
        porosity = normalize(
            porosity_raw, stats["porosity"]["mean"], stats["porosity"]["std"]
        )
        perm_r = normalize(
            perm_r_raw, stats["perm_r"]["mean"], stats["perm_r"]["std"]
        )
        perm_z = normalize(
            perm_z_raw, stats["perm_z"]["mean"], stats["perm_z"]["std"]
        )

        # Normalize scalars
        inj_rate = normalize(
            data["inj_rate"], stats["inj_rate"]["mean"], stats["inj_rate"]["std"]
        )
        temperature = normalize(
            data["temperature"], stats["temperature"]["mean"], stats["temperature"]["std"]
        )
        depth = normalize(
            data["depth"], stats["depth"]["mean"], stats["depth"]["std"]
        )
        swi = normalize(
            data["Swi"], stats["Swi"]["mean"], stats["Swi"]["std"]
        )
        lam = normalize(
            data["lam"], stats["lam"]["mean"], stats["lam"]["std"]
        )

        # Targets are [Z, R, T] -> [T, Z, R]
        gas = np.transpose(gas_raw, (2, 0, 1))
        pressure = np.transpose(pressure_raw, (2, 0, 1))
        pressure = normalize(
            pressure,
            stats["pressure_buildup"]["mean"],
            stats["pressure_buildup"]["std"],
        )

        T, Z, R = gas.shape

        # Coordinate channels
        z_coords = np.linspace(-1.0, 1.0, Z, dtype=np.float32)[:, None]
        r_coords = np.linspace(-1.0, 1.0, R, dtype=np.float32)[None, :]
        z_grid = np.broadcast_to(z_coords, (Z, R))
        r_grid = np.broadcast_to(r_coords, (Z, R))

        # Rescale perforation interval to new Z resolution
        z_start, z_end = rescale_perf_interval(data["perf_interval"], old_z, target_z)

        source = np.zeros((Z, R), dtype=np.float32)
        source[z_start:z_end, 0] = 1.0

        # Combine location and rate
        source_strength = source * np.float32(inj_rate)

        # Broadcast scalar fields
        temperature_grid = np.full((Z, R), temperature, dtype=np.float32)
        depth_grid = np.full((Z, R), depth, dtype=np.float32)
        swi_grid = np.full((Z, R), swi, dtype=np.float32)
        lam_grid = np.full((Z, R), lam, dtype=np.float32)

        time_steps = np.linspace(0.0, 1.0, T, dtype=np.float32)

        # [C, T, Z, R]
        x = np.zeros((11, T, Z, R), dtype=np.float32)

        for t in range(T):
            x[0, t] = porosity
            x[1, t] = perm_r
            x[2, t] = perm_z
            x[3, t] = temperature_grid
            x[4, t] = depth_grid
            x[5, t] = swi_grid
            x[6, t] = lam_grid
            x[7, t] = r_grid
            x[8, t] = z_grid
            x[9, t] = source_strength
            x[10, t] = time_steps[t]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(gas, dtype=torch.float32).unsqueeze(0),
            torch.tensor(pressure, dtype=torch.float32).unsqueeze(0),
        )


class CCSNetDataset(Dataset):
    def __init__(self, files, stats, target_z=51):
        self.files = files
        self.stats = stats
        self.target_z = target_z

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_sample(self.files[idx], self.stats, target_z=self.target_z)