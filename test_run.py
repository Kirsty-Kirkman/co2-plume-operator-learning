import argparse
import glob

import torch
import torch.nn.functional as F

from src.data.data_processing import CCSNetDataset, collect_stats
from src.models.fno_model import FNO3d


def build_synthetic_batch(time_steps=24, target_z=51, radial_cells=32):
    x = torch.randn(1, 11, time_steps, target_z, radial_cells)
    gas = torch.sigmoid(0.5 * x[:, 0:1] + 0.25 * x[:, 9:10] + 0.1 * x[:, 10:11])
    pressure = 10.0 * x[:, 1:2] + 2.0 * x[:, 4:5] + x[:, 10:11]
    return x, gas, pressure


def load_real_batch(data_path, target_z):
    all_files = sorted(glob.glob(data_path))
    if not all_files:
        return None

    subset = all_files[:1]
    normalizer = collect_stats(subset, target_z=target_z, max_files=1)
    dataset = CCSNetDataset(subset, normalizer=normalizer, target_z=target_z)

    x, gas, pressure = dataset[0]
    return x.unsqueeze(0), gas.unsqueeze(0), pressure.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short overfit smoke test for FNO3d.")
    parser.add_argument(
        "--mode",
        choices=["auto", "synthetic", "real"],
        default="auto",
        help="Data source for overfit test.",
    )
    parser.add_argument(
        "--data-path",
        default="dataset/train_data/*.npz",
        help="Glob pattern for real dataset files used when mode=real or auto.",
    )
    parser.add_argument("--target-z", type=int, default=51, help="Target Z resolution for preprocessing.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--model-width", type=int, default=16, help="FNO channel width for smoke test.")
    return parser.parse_args()


def test_overfit():
    args = parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    batch = None
    data_source = args.mode

    if args.mode in {"auto", "real"}:
        batch = load_real_batch(args.data_path, args.target_z)
        if batch is not None:
            data_source = "real"
        elif args.mode == "real":
            print(f"Error: no .npz files found at {args.data_path}")
            return 1

    if batch is None:
        batch = build_synthetic_batch(target_z=args.target_z)
        data_source = "synthetic"

    x, gas, pressure = batch
    x = x.to(device)
    gas = gas.to(device)
    pressure = pressure.to(device)

    model = FNO3d(in_ch=11, width=args.model_width, modes_t=4, modes_z=4, modes_r=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Starting short overfit test using {data_source} data...")

    initial_loss = None
    final_loss = None

    for i in range(args.iterations):
        optimizer.zero_grad()

        gas_pred, pressure_pred = model(x)
        loss = F.mse_loss(gas_pred, gas) + F.mse_loss(pressure_pred, pressure)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        if i % 5 == 0 or i == args.iterations - 1:
            print(f"Iteration {i:03d} | Loss: {loss.item():.6f}")

    if final_loss is not None and initial_loss is not None and final_loss < initial_loss:
        print("SUCCESS")
        print(f"Input shape: {tuple(x.shape)}")
        print(f"Gas output shape: {tuple(gas_pred.shape)}")
        print(f"Pressure output shape: {tuple(pressure_pred.shape)}")
        print("The 3D pipeline is functioning and the model is learning.")
        return 0

    print("FAILURE: model ran, but loss did not decrease.")
    return 1


if __name__ == "__main__":
    raise SystemExit(test_overfit())
