import numpy as np
import torch


def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.reshape(-1)


def r2_score(pred, target):
    pred = _to_numpy_1d(pred)
    target = _to_numpy_1d(target)

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2) + 1e-8
    return float(1.0 - (ss_res / ss_tot))


def mae_score(pred, target):
    pred = _to_numpy_1d(pred)
    target = _to_numpy_1d(target)
    return float(np.mean(np.abs(target - pred)))