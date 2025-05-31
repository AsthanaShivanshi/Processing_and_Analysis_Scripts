import yaml
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def gridded_R_squared(pred_path, truth_path, var1, var2, chunk_size={'time': 50}):
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    ss_res = ((var1_data - var2_data) ** 2).where(valid_mask).sum(dim='time')
    mean_true = var2_data.where(valid_mask).mean(dim='time')
    ss_tot = ((var2_data - mean_true) ** 2).where(valid_mask).sum(dim='time')
    r2 = 1 - (ss_res / ss_tot)
    return r2.astype(np.float32)

def gridded_RMSE(pred_path, truth_path, var1, var2, chunk_size={'time': 100}):
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    diff_squared = (var1_data - var2_data) ** 2
    diff_squared = diff_squared.where(valid_mask)
    mse = diff_squared.sum(dim='time') / valid_mask.sum(dim='time')
    return np.sqrt(mse).astype(np.float32)

def plot_gridded_metric(dataarray, title, cmap="viridis", save_dir="Outputs/RMSE_Rsquared_gridded", filename=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    dataarray.plot(cmap=cmap)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    if filename:
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"Saved: {full_path}")

    plt.close()  

with open("config_files/evaluation_metrics.yaml", "r") as file:
    config = yaml.safe_load(file)

# Only process train and test
for split in ["train", "test"]:
    print(f"Split: {split}")

    pred_paths = config["datasets"][split]["pred"]
    truth_paths = config["datasets"][split]["truth"]
    pred_vars = config["variables"]["pred"]
    truth_vars = config["variables"]["truth"]

    for pred_path, truth_path, var_pred, var_truth in zip(pred_paths, truth_paths, pred_vars, truth_vars):
        print(f"📂 {var_pred} | File: {os.path.basename(pred_path)}")

        try:
            r2 = gridded_R_squared(pred_path, truth_path, var_pred, var_truth)
            filename_r2 = f"{split}_R2_{var_pred}.png"
            plot_gridded_metric(r2, f"{split.capitalize()} Gridded R² - {var_pred}",
                                cmap="viridis", filename=filename_r2)
        except Exception as e:
            print(f"Failed R² for {var_pred}: {e}")

        try:
            rmse = gridded_RMSE(pred_path, truth_path, var_pred, var_truth)
            filename_rmse = f"{split}_RMSE_{var_pred}.png"
            plot_gridded_metric(rmse, f"{split.capitalize()} Gridded RMSE - {var_pred}",
                                cmap="magma", filename=filename_rmse)
        except Exception as e:
            print(f"Failed RMSE for {var_pred}: {e}")
