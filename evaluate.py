import numpy as np
import torch
import glob
import torch.nn.functional as F
from fno_model import FNO3d
from data_processing import collect_stats, load_sample, normalize

def r2_score(pred, target):
    """
    Standard R2 score for physical evaluation.
    """
    pred = pred.flatten()
    target = target.flatten()
    target_mean = np.mean(target)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target_mean) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def mae_score(pred, target):
    return np.mean(np.abs(pred - target))

def evaluate_model(model_path, data_dir, device="cpu"):
    # 1. Setup Data and Stats
    test_files = sorted(glob.glob(f"{data_dir}/*.npz"))
    # Use the same stats logic used during training
    stats = collect_stats(test_files) 
    
    # 2. Load Model
    model = FNO3d(in_ch=11, width=32, modes_t=8, modes_z=12, modes_r=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gas_r2_list, pressure_r2_list = [], []
    gas_mae_list, pressure_mae_list = [], []

    print(f"Starting evaluation on {len(test_files)} files...")

    with torch.no_grad():
        for file_path in test_files:
            # Load and normalize sample using the 3D logic
            x, gas_true, pressure_true = load_sample(file_path, stats=stats)
            
            # Prepare for model (add batch dimension)
            x = x.unsqueeze(0).to(device)
            
            # Forward Pass
            gas_pred, pressure_pred = model(x)
            
            # Convert back to CPU/Numpy and remove dimensions
            gas_pred = gas_pred.squeeze().cpu().numpy()         # (nt, nz, nx)
            gas_true = gas_true.squeeze().cpu().numpy()
            pressure_pred = pressure_pred.squeeze().cpu().numpy()
            pressure_true = pressure_true.squeeze().cpu().numpy()

            # 3. De-normalize Pressure
            # Gas is usually 0-1, but pressure needs to return to physical units
            pressure_pred_phys = (pressure_pred * stats["pressure"]["std"]) + stats["pressure"]["mean"]
            pressure_true_phys = (pressure_true * stats["pressure"]["std"]) + stats["pressure"]["mean"]

            # 4. Calculate Metrics
            g_r2 = r2_score(gas_pred, gas_true)
            p_r2 = r2_score(pressure_pred_phys, pressure_true_phys)
            
            gas_r2_list.append(g_r2)
            pressure_r2_list.append(p_r2)
            gas_mae_list.append(mae_score(gas_pred, gas_true))
            pressure_mae_list.append(mae_score(pressure_pred_phys, pressure_true_phys))

    # 5. Final Report
    print("\n--- Evaluation Results ---")
    print(f"Gas Saturation R2:    {np.mean(gas_r2_list):.4f}")
    print(f"Pressure Buildup R2:  {np.mean(pressure_r2_list):.4f}")
    print(f"Gas Saturation MAE:   {np.mean(gas_mae_list):.6f}")
    print(f"Pressure Buildup MAE: {np.mean(pressure_mae_list):.4f}")
    
    # Lecturer tip: R2 > 0.95 is generally considered a strong performance
    if np.mean(gas_r2_list) > 0.95:
        print("Performance Meeting Requirements: Excellent correlation detected.")

if __name__ == "__main__":
    # Example usage:
    # evaluate_model("checkpoints/fno3d_best.pt", "data/test_set", device="cuda")
    pass