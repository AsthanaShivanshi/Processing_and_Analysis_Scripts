import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse

parser = argparse.ArgumentParser(description="City-specific RMSE vs Quantile")
parser.add_argument("--var", type=int, required=True, help="Variable index (0-3)")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--lat", type=float, required=True, help="Latitude of city")
parser.add_argument("--lon", type=float, required=True, help="Longitude of city")
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

# Load datasets
unet_train_ds = xr.open_dataset(unet_train_path)
unet_combined_ds = xr.open_dataset(unet_combined_path)
bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))

lats = target_ds_var['lat'].values
lons = target_ds_var['lon'].values

if lats.ndim == 1 and lons.ndim == 1:
    lat_idx = np.argmin(np.abs(lats - args.lat))
    lon_idx = np.argmin(np.abs(lons - args.lon))
elif lats.ndim == 2 and lons.ndim == 2:
    dist = np.sqrt((lats - args.lat)**2 + (lons - args.lon)**2)
    lat_idx, lon_idx = np.unravel_index(np.argmin(dist), lats.shape)
else:
    raise ValueError("Latitude/Longitude arrays have unexpected dimensions.")

# Extract time series for the city grid cell
target_city = target_ds_var[file_var].values[:, lat_idx, lon_idx]
bicubic_city = bicubic_ds[file_var].values[:, lat_idx, lon_idx]
unet_train_city = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values[:, lat_idx, lon_idx]
unet_combined_city = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values[:, lat_idx, lon_idx]

# Mask invalid points
valid_mask = ~np.isnan(target_city) & ~np.isnan(bicubic_city) & ~np.isnan(unet_train_city) & ~np.isnan(unet_combined_city)
target_city = np.where(valid_mask, target_city, np.nan)
bicubic_city = np.where(valid_mask, bicubic_city, np.nan)
unet_train_city = np.where(valid_mask, unet_train_city, np.nan)
unet_combined_city = np.where(valid_mask, unet_combined_city, np.nan)

# Only consider non-zero precipitation for RMSE calculation
if var == "precip":
    nonzero_mask = target_city > 0
    target_city = np.where(nonzero_mask, target_city, np.nan)
    bicubic_city = np.where(nonzero_mask, bicubic_city, np.nan)
    unet_train_city = np.where(nonzero_mask, unet_train_city, np.nan)
    unet_combined_city = np.where(nonzero_mask, unet_combined_city, np.nan)

# Quantile thresholds for pooling
quantiles_to_plot = np.arange(0, 101, 10)  # 0 to 100 percentiles
thresholds = [np.nanquantile(target_city, q/100) for q in quantiles_to_plot]

pooled_rmse_dict = {
    "Bicubic": [],
    "UNet 1971": [],
    "UNet Combined": []
}

def pooled_rmse(pred, mask):
    squared_error = (pred - target_city) ** 2
    squared_error_masked = np.where(mask, squared_error, np.nan)
    return np.sqrt(np.nanmean(squared_error_masked))

for thresh in thresholds:
    mask = (target_city <= thresh)
    pooled_rmse_dict["Bicubic"].append(pooled_rmse(bicubic_city, mask))
    pooled_rmse_dict["UNet 1971"].append(pooled_rmse(unet_train_city, mask))
    pooled_rmse_dict["UNet Combined"].append(pooled_rmse(unet_combined_city, mask))

plt.figure(figsize=(8,6))
for method, color in zip(pooled_rmse_dict.keys(), ["orange", "red", "blue"]):
    plt.plot(quantiles_to_plot, pooled_rmse_dict[method], label=method, color=color, linewidth=2)
plt.xlabel("Quantile threshold")
plt.ylabel(f"RMSE for {args.city} ({args.lat}, {args.lon})")
plt.title(f"RMSE vs Quantile threshold for {args.city} ({var})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/rmse_vs_quantile_{var}_{args.city}.png", dpi=1000)