import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

downscaled_test_dataset = xr.open_dataset(
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/Training_Dataset_Downscaled_Predictions_2011_2020.nc"
)
input_test_temp = xr.open_dataset(
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TabsD_step3_interp.nc"
).sel(time=slice("2021-01-01", "2023-12-31"))
input_test_precip = xr.open_dataset(
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/RhiresD_step3_interp.nc"
).sel(time=slice("2021-01-01", "2023-12-31"))
input_test_tmin = xr.open_dataset(
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TminD_step3_interp.nc"
).sel(time=slice("2021-01-01", "2023-12-31"))
input_test_tmax = xr.open_dataset(
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Training_Chronological_Dataset/TmaxD_step3_interp.nc"
).sel(time=slice("2021-01-01", "2023-12-31"))

# Variable info: (name in dataset, plot title)
variables = [
    ("TabsD", "Daily Temperature", input_test_temp),
    ("TminD", "Daily Minimum Temperature", input_test_tmin),
    ("TmaxD", "Daily Maximum Temperature", input_test_tmax),
    ("RhiresD", "Daily Precipitation", input_test_precip),
]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
for ax, (var, title, bicubic_ds) in zip(axs.flat, variables):
    # Mean over time for bicubic and downscaled
    bicubic = bicubic_ds[var].mean(dim="time")
    downscaled = downscaled_test_dataset[var].sel(time=slice("2021-01-01", "2023-12-31")).mean(dim="time")
    # Calculate improvement
    improvement = ((bicubic - downscaled) / bicubic) * 100
    improvement = improvement.where(~np.isnan(bicubic) & ~np.isnan(downscaled))
    # Plot
    lon = improvement["lon"].values
    lat = improvement["lat"].values
    Lon, Lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(
    Lon, Lat, improvement.values,
    cmap="BrBG", vmin=-100, vmax=100, shading="auto"
)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

fig.colorbar(im, ax=axs, orientation="vertical", label="% Improvement over Bicubically Interpolated Test Set")
plt.tight_layout()
plt.savefig("all_variables_improvement_map_RMSE_allvars.png", dpi=1000)
plt.show()