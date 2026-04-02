import os
import numpy as np


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


def predict_sample(data):
    """
    Replace this with your actual model inference code.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        One loaded sample from np.load(...)

    Returns
    -------
    gas_saturation_pred : np.ndarray
        Must have the same shape as data["gas_saturation"]
    pressure_buildup_pred : np.ndarray
        Must have the same shape as data["pressure_buildup"]
    """

    # Placeholder predictions
    gas_saturation_pred = np.zeros_like(data["gas_saturation"])
    pressure_buildup_pred = np.zeros_like(data["pressure_buildup"])

    return gas_saturation_pred, pressure_buildup_pred


def evaluate(data_dir="dataset/test_data", n_samples=None):
    gas_saturation_r2 = []
    pressure_buildup_r2 = []
    gas_saturation_MAE = []
    pressure_buildup_MAE = []

    all_files = sorted(
        [
            f for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ]
    )

    if n_samples is None:
        n_samples = len(all_files)

    for idx in range(n_samples):
        file_path = os.path.join(data_dir, f"data_{str(idx).zfill(4)}.npz")

        with np.load(file_path) as data:
            gas_saturation = data["gas_saturation"]
            pressure_buildup = data["pressure_buildup"]

            gas_saturation_pred, pressure_buildup_pred = predict_sample(data)

            if gas_saturation_pred.shape != gas_saturation.shape:
                raise ValueError(
                    f"gas_saturation_pred has shape {gas_saturation_pred.shape}, "
                    f"but expected {gas_saturation.shape}"
                )

            if pressure_buildup_pred.shape != pressure_buildup.shape:
                raise ValueError(
                    f"pressure_buildup_pred has shape {pressure_buildup_pred.shape}, "
                    f"but expected {pressure_buildup.shape}"
                )

            gas_saturation_r2.append(r2(gas_saturation, gas_saturation_pred))
            pressure_buildup_r2.append(r2(pressure_buildup, pressure_buildup_pred))

            gas_saturation_MAE.append(MAE(gas_saturation, gas_saturation_pred))
            pressure_buildup_MAE.append(MAE(pressure_buildup, pressure_buildup_pred))

    print("--------------")
    print(f"Average gas saturation R2 score is: {np.mean(gas_saturation_r2):.6f}")
    print(f"Average pressure buildup R2 score is: {np.mean(pressure_buildup_r2):.6f}")
    print("--------------")
    print(f"Average gas saturation MAE is: {np.mean(gas_saturation_MAE):.6f}")
    print(f"Average pressure buildup MAE is: {np.mean(pressure_buildup_MAE):.6f}")

    return {
        "gas_saturation_r2": np.mean(gas_saturation_r2),
        "pressure_buildup_r2": np.mean(pressure_buildup_r2),
        "gas_saturation_MAE": np.mean(gas_saturation_MAE),
        "pressure_buildup_MAE": np.mean(pressure_buildup_MAE),
    }


if __name__ == "__main__":
    evaluate()