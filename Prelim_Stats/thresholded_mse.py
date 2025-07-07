import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config

unet_pretrain_path = f"{config.UNET_1771_DIR}/Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc" #This has precip, temp, tmin and tmax
unet_train_path = f"{config.UNET_1971_DIR}/Training_Dataset_Downscaled_Predictions_2011_2020.nc"#this has RhiresD, TminD, TmaxD and TabsD as vars
target_files = {#this has RhiresD, TminD, TmaxD and TabsD as vars
    "RhiresD": f"{config.TARGET_DIR}/RhiresD_1971_2023.nc",
    "TabsD": f"{config.TARGET_DIR}/TabsD_1971_2023.nc",
    "TminD": f"{config.TARGET_DIR}/TminD_1971_2023.nc",
    "TmaxD": f"{config.TARGET_DIR}/TmaxD_1971_2023.nc",
} 
bicubic_files = { #this has RhiresD, TminD, TmaxD and TabsD as vars
    "RhiresD": f"{config.DATASETS_TRAINING_DIR}/RhiresD_step3_interp.nc",
    "TabsD":   f"{config.DATASETS_TRAINING_DIR}/TabsD_step3_interp.nc",
    "TminD":   f"{config.DATASETS_TRAINING_DIR}/TminD_step3_interp.nc",
    "TmaxD":   f"{config.DATASETS_TRAINING_DIR}/TmaxD_step3_interp.nc",
}

bicubic_varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_names = ["precip", "temp", "tmin", "tmax"]

bicubic_ds = xr.open_mfdataset(bicubic_files) #Bicubic baseline
unet_pretrain_ds = xr.open_dataset(unet_pretrain_path) #downscaled ts from 1771 model
unet_train_ds = xr.open_dataset(unet_train_path) #downscaled ts from 1971 model
target_ds = xr.open_mfdataset(target_files)

quantiles = list(range(5, 100, 5))

for var in var_names:
    target = target_ds[var].values
    bicubic = bicubic_ds[var].values
    unet_pretrain = unet_pretrain_ds[var].values
    unet_train = unet_train_ds[var].values

    target_flat = target.flatten()
    bicubic_flat = bicubic.flatten()
    unet_pretrain_flat = unet_pretrain.flatten()
    unet_train_flat = unet_train.flatten()

    thresholds = [np.quantile(target_flat, q/100) for q in quantiles]

    mse_bicubic = []
    mse_pretrain = []
    mse_train = []

    for thresh in thresholds:
        mask = target_flat >= thresh
        if np.sum(mask) == 0:
            mse_bicubic.append(np.nan)
            mse_pretrain.append(np.nan)
            mse_train.append(np.nan)
            continue
        mse_bicubic.append(np.mean((bicubic_flat[mask] - target_flat[mask])**2))
        mse_pretrain.append(np.mean((unet_pretrain_flat[mask] - target_flat[mask])**2))
        mse_train.append(np.mean((unet_train_flat[mask] - target_flat[mask])**2))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(quantiles, mse_bicubic, marker='o', label="Bicubic")
    plt.plot(quantiles, mse_pretrain, marker='o', label="UNet 1771 time series")
    plt.plot(quantiles, mse_train, marker='o', label="UNet 1971 time series")
    plt.xlabel("Quantile (%)")
    plt.ylabel("Thresholded MSE")
    plt.title(f"Thresholded MSE by Quantile for {var}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUTS_DIR}/thresholded_mse_comparison_multiple_baselines_{var}.png", dpi=1000)
    plt.close()