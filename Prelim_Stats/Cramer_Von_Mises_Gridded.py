import xarray as xr
import numpy as np
from scipy.stats import cramervonmises_2samp
import matplotlib.pyplot as plt
from pathlib import Path
import config

VAR_MAP = {
    "precip": {"hr": "RhiresD", "model": "precip"},
    "temp":   {"hr": "TabsD",   "model": "temp"},
    "tmin":   {"hr": "TminD",   "model": "tmin"},
    "tmax":   {"hr": "TmaxD",   "model": "tmax"},
}

BASE_DIR = Path(config.BASE_DIR)
OUTPUTS_DIR = Path(config.OUTPUTS_DIR)

bicubic_paths = {
    "RhiresD": BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "RhiresD_step3_interp.nc",
    "TabsD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TabsD_step3_interp.nc",
    "TminD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TminD_step3_interp.nc",
    "TmaxD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TmaxD_step3_interp.nc",
}

def compute_gridwise_cvm(obs, baselines):
    #[time, latlon] or [time, N, E]
    dims = [d for d in obs.dims if d != "time"]
    shape = obs.isel(time=0).shape
    coords = {d: obs.coords[d] for d in dims}
    cvm_maps = {name: np.full(shape, np.nan) for name in baselines}
    for idx in np.ndindex(shape):
        obs_series = obs.isel({dims[0]: idx[0], dims[1]: idx[1]}).values
        if np.all(np.isnan(obs_series)):
            continue
        for name, baseline in baselines.items():
            base_series = baseline.isel({dims[0]: idx[0], dims[1]: idx[1]}).values
            mask = ~np.isnan(obs_series) & ~np.isnan(base_series)
            if np.sum(mask) < 10:
                continue
            try:
                stat = cramervonmises_2samp(obs_series[mask], base_series[mask]).statistic
            except Exception:
                stat = np.nan
            cvm_maps[name][idx] = stat
    return {name: xr.DataArray(arr, coords=coords, dims=dims, name=f"cvm_{name}") for name, arr in cvm_maps.items()}


def plot_cvm_maps(cvm_maps, var, file_var, save_path):
    method_names = list(cvm_maps.keys())
    ncols = len(method_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    vmin, vmax = 0, np.nanmax([np.nanmax(cvm_maps[m].values) for m in method_names])
    for j, method in enumerate(method_names):
        ax = axes[j]
        im = ax.imshow(cvm_maps[method].values, origin='lower', aspect='auto', cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label("Cramer-von Mises Statistic")
    fig.suptitle(f"Spatial CvM Maps for {var} ({file_var})", fontsize=18)
    plt.savefig(save_path, dpi=1000)
    plt.close()

if __name__ == "__main__":
    for var_key in VAR_MAP:
        hr_var = VAR_MAP[var_key]["hr"]
        model_var = VAR_MAP[var_key]["model"]
        print(f"Processing {var_key}")

        obs_ds = xr.open_dataset(
            str(BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full" / f"{hr_var}_1971_2023.nc"),
            chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))
        unet_ds_1971 = xr.open_dataset(
            str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Training_Dataset" / "Training_Dataset_Downscaled_Predictions_2011_2020.nc"),
            chunks={"time": 100})
        unet_ds_1771 = xr.open_dataset(
            str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Pretraining_Dataset" / "Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc"),
            chunks={"time": 100})
        unet_combined = xr.open_dataset(
            str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Combined_Dataset" / "Combined_Dataset_Downscaled_Predictions_2011_2020.nc"),
            chunks={"time": 100}
        )
        bicubic_path = bicubic_paths[hr_var]
        bicubic_ds = xr.open_dataset(str(bicubic_path), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

        obs = obs_ds[hr_var]
        unet_1971 = unet_ds_1971[hr_var]
        unet_1771 = unet_ds_1771[model_var]
        unet_comb = unet_combined[model_var]
        bicubic = bicubic_ds[hr_var]

        baselines = {
            "UNet 1971": unet_1971,
            "UNet 1771": unet_1771,
            "UNet Combined": unet_comb,
            "Bicubic": bicubic,
        }
        cvm_maps = compute_gridwise_cvm(obs, baselines)
        plot_cvm_maps(cvm_maps, var_key, hr_var, OUTPUTS_DIR / f"spatial_cvm_maps_{var_key}.png")
        print(f"Done {var_key}")