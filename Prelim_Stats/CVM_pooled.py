import xarray as xr
import numpy as np
import pandas as pd
import config

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())

unet_train_path = f"{config.UNET_1971_DIR}/Optim_Training_Downscaled_Predictions_2011_2020.nc"
unet_combined_path = f"{config.UNET_COMBINED_DIR}/Combined_Downscaled_Predictions_2011_2020.nc"
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

results = {}

for var in var_list:
    file_var = varnames[var]
    unet_train_ds = xr.open_dataset(unet_train_path)
    unet_combined_ds = xr.open_dataset(unet_combined_path)
    bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
    target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))

    target = target_ds_var[file_var].values
    bicubic = bicubic_ds[file_var].values
    unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
    unet_combined = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values

    valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_train) & ~np.isnan(unet_combined)
    target = np.where(valid_mask, target, np.nan)
    bicubic = np.where(valid_mask, bicubic, np.nan)
    unet_train = np.where(valid_mask, unet_train, np.nan)
    unet_combined = np.where(valid_mask, unet_combined, np.nan)

    # each baseline
    rmse_bicubic = np.sqrt(np.nanmean((bicubic - target) ** 2))
    rmse_unet_train = np.sqrt(np.nanmean((unet_train - target) ** 2))
    rmse_unet_combined = np.sqrt(np.nanmean((unet_combined - target) ** 2))

    results[var] = {
        "Bicubic": rmse_bicubic,
        "UNet 1971": rmse_unet_train,
        "UNet Combined": rmse_unet_combined
    }

columns = pd.MultiIndex.from_product(
    [var_list, ["Bicubic", "UNet 1971", "UNet Combined"]],
    names=["Variable", "Baseline"]
)
rmse_row = []
for var in var_list:
    rmse_row.extend([results[var]["Bicubic"], results[var]["UNet 1971"], results[var]["UNet Combined"]])

df = pd.DataFrame([rmse_row], columns=columns, index=["Spatiotemporal RMSE"])

output_csv = f"{config.OUTPUTS_DIR}/pooled_spatiotemporal_rmse_table.csv"
df.to_csv(output_csv)
print(f"Saved spatiotemporal pooled RMSE table to {output_csv}")