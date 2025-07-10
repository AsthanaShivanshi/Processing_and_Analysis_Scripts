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

def compute_gridwise_cvm(obs, baselines, alpha=0.05):
    dims = [d for d in obs.dims if d != "time"]
    shape = obs.isel(time=0).shape
    coords = {d: obs.coords[d] for d in dims}
    cvm_maps = {name: np.full(shape, np.nan) for name in baselines}
    pval_maps = {name: np.full(shape, np.nan) for name in baselines}
    print("Spatial dims used:", dims)
    print("Shape:", np.shape)
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
                res = cramervonmises_2samp(obs_series[mask], base_series[mask])
                stat = res.statistic
                pval = res.pvalue
            except Exception:
                stat = np.nan
                pval = np.nan
            cvm_maps[name][idx] = stat
            pval_maps[name][idx] = pval
    return (
        {name: xr.DataArray(arr, coords=coords, dims=dims, name=f"cvm_{name}") for name, arr in cvm_maps.items()},
        {name: xr.DataArray(arr, coords=coords, dims=dims, name=f"pval_{name}") for name, arr in pval_maps.items()}
    )

def ensure_latlon(da):
    # Only rename if needed and avoid conflicts
    dims = list(da.dims)
    rename_dict = {}
    if 'N' in dims and 'lat' not in dims:
        rename_dict['N'] = 'lat'
    if 'E' in dims and 'lon' not in dims:
        rename_dict['E'] = 'lon'
    if rename_dict:
        da = da.rename(rename_dict)
    return da

def plot_cvm_maps_with_pval(cvm_maps, pval_maps, var, file_var, save_path, alpha=0.05):
    method_names = list(cvm_maps.keys())
    ncols = len(method_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    vmax = np.nanmax([np.nanmax(cvm_maps[m].where(pval_maps[m] >= alpha).values) for m in method_names])
    vmin = 0
    for j, method in enumerate(method_names):
        ax = axes[j]
        stat = cvm_maps[method].values
        pval = pval_maps[method].values
        # red where p < 0.05 ( less than 95 percent confidence)
        cmap = plt.get_cmap("viridis")
        normed_stat = (stat - vmin) / (vmax - vmin) if vmax > vmin else stat
        rgb = cmap(normed_stat)
        # Set red where p < 0.05
        red_mask = (pval < 0.05) & ~np.isnan(pval)
        rgb[red_mask] = [1, 0, 0, 1]  
        nan_mask = np.isnan(stat)
        rgb[nan_mask] = [1, 1, 1, 0]
        ax.imshow(rgb, origin='lower', aspect='auto')
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label("Cramer-von Mises Statistic")
    fig.suptitle(f"Spatial CvM Maps for {var} ({file_var})\nRed: p < {alpha}", fontsize=18)
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
        
        #For debugging
        print("obs_ds dims:", obs_ds[hr_var].dims)
        print("unet_1971 dims:", unet_ds_1971[hr_var].dims)
        print("unet_1771 dims:", unet_ds_1771[model_var].dims)
        print("unet_comb dims:", unet_combined[model_var].dims)
        print("bicubic dims:", bicubic_ds[hr_var].dims)


        obs = ensure_latlon(obs_ds[hr_var])
        bicubic = ensure_latlon(bicubic_ds[hr_var])
        unet_1971 = unet_ds_1971[hr_var]
        unet_1771 = unet_ds_1771[model_var]
        unet_comb = unet_combined[model_var]

        print("obs dims after ensure_latlon:", obs.dims)
        print("unet_1971 dims after ensure_latlon:", unet_1971.dims)
        print("unet_1771 dims after ensure_latlon:", unet_1771.dims)
        print("unet_comb dims after ensure_latlon:", unet_comb.dims)
        print("bicubic dims after ensure_latlon:", bicubic.dims)

        baselines = {
            "UNet 1971": unet_1971,
            "UNet 1771": unet_1771,
            "UNet Combined": unet_comb,
            "Bicubic": bicubic,
        }

        #alpha rejection level : 95 percent confidence
        cvm_maps, pval_maps = compute_gridwise_cvm(obs, baselines)
        plot_cvm_maps_with_pval(cvm_maps, pval_maps, var_key, hr_var, OUTPUTS_DIR / f"spatial_cvm_maps_{var_key}.png")
        print(f"Done {var_key}")