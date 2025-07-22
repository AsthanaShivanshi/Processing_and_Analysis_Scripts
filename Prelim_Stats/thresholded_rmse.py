
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxSCRIPT UNDER CONSTRUCTIONxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse

parser = argparse.ArgumentParser(description="Spatially Pooled MSE vs Quantile Threshold")
parser.add_argument("--var", type=int, required=True, help="Variable index (0-3)")
args = parser.parse_args()

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())
var = var_list[args.var]
file_var = varnames[var]

unet_train_path = f"{config.UNET_1971_DIR}/Training_Dataset_Downscaled_Predictions_2011_2020.nc"
unet_combined_path= f"{config.UNET_COMBINED_DIR}/Combined_Dataset_Downscaled_Predictions_2011_2020.nc"
target_files = {
    "RhiresD": f"{config.TARGET_DIR}/RhiresD_1971_2023.nc",
    "TabsD": f"{config.TARGET_DIR}/TabsD_1971_2023.nc",
    "TminD": f"{config.TARGET_DIR}/TminD_1971_2023.nc",
    "TmaxD": f"{config.TARGET_DIR}/TmaxD_1971_2023.nc",
}
bicubic_files = {
    "RhiresD": f"{config.DATASETS_TRAINING_DIR}/RhiresD_step3_interp.nc",
    "TabsD":   f"{config.DATASETS_TRAINING_DIR}/TabsD_step3_interp.nc",
    "TminD":   f"{config.DATASETS_TRAINING_DIR}/TminD_step3_interp.nc",
    "TmaxD":   f"{config.DATASETS_TRAINING_DIR}/TmaxD_step3_interp.nc",
}

unet_train_ds = xr.open_dataset(unet_train_path)
unet_combined_ds = xr.open_dataset(unet_combined_path)

bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
bicubic = bicubic_ds[file_var].values

target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target = target_ds_var[file_var].values

unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_combined = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values

valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_train)
target = np.where(valid_mask, target, np.nan)
bicubic = np.where(valid_mask, bicubic, np.nan)
unet_train = np.where(valid_mask, unet_train, np.nan)
unet_combined = np.where(valid_mask, unet_combined, np.nan)

# Quantile thresholds for pooling
quantiles_to_plot = np.arange(0, 101, 10) # 0 to 100 percentiles
thresholds = [np.nanquantile(target, q/100) for q in quantiles_to_plot]

pooled_mse = {
    "Bicubic": [],
    "UNet 1771": [],
    "UNet 1971": [],
    "UNet Combined": []
}

for thresh in thresholds:
    mask = (target <= thresh)
    def pooled(pred):
        squared_error = (pred - target) ** 2
        squared_error_masked = np.where(mask, squared_error, np.nan)
        return np.nanmean(squared_error_masked)
    pooled_mse["Bicubic"].append(pooled(bicubic))
    pooled_mse["UNet 1971"].append(pooled(unet_train))
    pooled_mse["UNet Combined"].append(pooled(unet_combined))

plt.figure(figsize=(8,6))
for method, color in zip(pooled_mse.keys(), ["orange", "red", "blue", "green"]):
    plt.plot(quantiles_to_plot, pooled_mse[method], label=method, color=color, linewidth=2)
plt.xlabel("Quantile threshold (%)")
plt.ylabel("Spatially pooled MSE")
plt.title(f"Spatially pooled MSE vs Quantile threshold for {var} ({file_var})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/spatially_pooled_mse_vs_quantile_{var}.png", dpi=1000)
plt.close()