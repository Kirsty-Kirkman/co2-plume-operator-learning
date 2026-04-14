import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.data_processing import CCSNetDataset, collect_stats
from src.models.fno_model import FNO3d


def test_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Load a single sample
    all_files = sorted(glob.glob("dataset/train_data/*.npz"))
    if not all_files:
        print("❌ Error: No .npz files found in dataset/train_data!")
        return

    subset = all_files[:1]
    stats = collect_stats(subset, max_files=1)
    dataset = CCSNetDataset(subset, stats=stats)
    loader = DataLoader(dataset, batch_size=1)

    # Initialize 3D model
    model = FNO3d(in_ch=11, width=16, modes_t=4, modes_z=4, modes_r=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting short overfit test...")

    x, gas, pressure = next(iter(loader))
    x, gas, pressure = x.to(device), gas.to(device), pressure.to(device)

    initial_loss = None

    for i in range(20):
        optimizer.zero_grad()

        gas_pred, pressure_pred = model(x)

        loss = F.mse_loss(gas_pred, gas) + F.mse_loss(pressure_pred, pressure)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()

        if i % 5 == 0:
            print(f"Iteration {i} | Loss: {loss.item():.6f}")

    if loss.item() < initial_loss:
        print("\n✅ SUCCESS!")
        print(f"Input Shape: {x.shape}")
        print(f"Gas Output Shape: {gas_pred.shape}")
        print(f"Pressure Output Shape: {pressure_pred.shape}")
        print("The 3D pipeline is functioning and the model is learning.")
    else:
        print("\n❌ Model is running but loss did not decrease much.")


if __name__ == "__main__":
    test_overfit()