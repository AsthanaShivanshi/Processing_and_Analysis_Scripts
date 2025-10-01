import numpy as np
import pandas as pd
import sys
import os
import xarray as xr
from major_return_levels_bm import get_extreme_return_levels_bm
sys.path.append('../Prelim_Stats')
import config
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def gridcell(ds):
    lats = ds['lat'].values
    lons = ds['lon'].values
    grid_cells = []
    for lat in lats:
        for lon in lons:
            grid_cells.append((float(lat), float(lon)))
    return grid_cells

def rmse(errors):
    errors = np.array(errors).flatten()
    return np.sqrt(np.mean(errors ** 2))

def pooled_rmse(
    obs_file, obs_var, baseline_files, baseline_vars, return_periods, block_size, time_slice
):
    ds_obs = xr.open_dataset(obs_file).sel(time=slice(*time_slice))
    grid_cells = gridcell(ds_obs)
    results = {name: [] for name in baseline_files.keys()}

    for lat, lon in grid_cells:
        try:
            obs_rl = get_extreme_return_levels_bm(
                nc_file=obs_file,
                variable_name=obs_var,
                lat=lat,
                lon=lon,
                return_periods=return_periods,
                block_size=block_size,
                time_slice=time_slice,
                return_all_periods=False
            )["return value"].values
        except Exception:
            continue 

        for name, file in baseline_files.items():
            try:
                model_rl = get_extreme_return_levels_bm(
                    nc_file=file,
                    variable_name=baseline_vars[name],
                    lat=lat,
                    lon=lon,
                    return_periods=return_periods,
                    block_size=block_size,
                    time_slice=time_slice,
                    return_all_periods=False
                )["return value"].values
                results[name].append(model_rl - obs_rl)
            except Exception:
                continue

    rmse_table = {}
    for name, diffs in results.items():
        rmse_table[name] = rmse(diffs)

    return rmse_table

if __name__ == "__main__":
    return_periods = [10, 20, 50, 100]
    block_size = '365D'
    time_slice = ('1981-01-01', '2010-12-31') #Cal period

    temp_baseline_files = {
        "COARSE": f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc",
        "EQM": f"{config.BIAS_CORRECTED_DIR}/EQM/tmax_BC_bicubic_r01.nc",
        "EQM_UNET": f"{config.BIAS_CORRECTED_DIR}/EQM/DOWNSCALED_TRAINING_QM_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "DOTC": f"{config.BIAS_CORRECTED_DIR}/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
        "DOTC_UNET": f"{config.BIAS_CORRECTED_DIR}/dOTC/DOWNSCALED_TRAINING_DOTC_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "QDM": f"{config.BIAS_CORRECTED_DIR}/QDM/tmax_BC_bicubic_r01.nc",
        "QDM_UNET": f"{config.BIAS_CORRECTED_DIR}/QDM/DOWNSCALED_TRAINING_QM_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc"
    }
    temp_baseline_vars = {
        "COARSE": "tmax",
        "EQM": "tmax",
        "EQM_UNET": "tmax",
        "DOTC": "tmax",
        "DOTC_UNET": "tmax",
        "QDM": "tmax",
        "QDM_UNET": "tmax"
    }
    temp_obs_file = f"{config.TARGET_DIR}/TmaxD_1971_2023.nc"
    temp_obs_var = "TmaxD"

    temp_rmse = pooled_rmse(
        temp_obs_file, temp_obs_var, temp_baseline_files, temp_baseline_vars,
        return_periods, block_size, time_slice
    )
    print("tmax RMSE table (pooled across all grid cells):")
    rmse_table = pd.DataFrame([temp_rmse])
    print(rmse_table)
    rmse_table.to_csv("tmax_return_level_rmse_table.csv", index=False)



def get_gridwise_return_levels(nc_file, variable_name, return_period, block_size, time_slice):
    ds = xr.open_dataset(nc_file).sel(time=slice(*time_slice))
    lats = ds['lat'].values
    lons = ds['lon'].values
    rl_grid = np.full((len(lats), len(lons)), np.nan)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            try:
                rl = get_extreme_return_levels_bm(
                    nc_file=nc_file,
                    variable_name=variable_name,
                    lat=lat,
                    lon=lon,
                    return_periods=[return_period],
                    block_size=block_size,
                    time_slice=time_slice,
                    return_all_periods=False
                )["return value"].values[0]
                rl_grid[i, j] = rl
            except Exception:
                continue
    return lats, lons, rl_grid

titles = ['OBS'] + list(temp_baseline_files.keys())
all_files = [temp_obs_file] + [temp_baseline_files[name] for name in temp_baseline_files.keys()]
all_vars = [temp_obs_var] + [temp_baseline_vars[name] for name in temp_baseline_files.keys()]

for rp in return_periods:
    grids = []
    for file, var in zip(all_files, all_vars):
        lats, lons, grid = get_gridwise_return_levels(
            file, var, rp, block_size, time_slice
        )
        grids.append(grid)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    for ax, title, grid in zip(axes, titles, grids):
        im = ax.pcolormesh(lons, lats, grid, cmap='coolwarm', shading='auto')
        ax.set_title(title)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent([5.5, 10.5, 45.5, 47.9])
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
    if len(axes) > len(grids):
        for ax in axes[len(grids):]:
            ax.axis('off')
    plt.suptitle(f"{rp}-year Return Level (Tmax),1981-2010",fontsize=16)
    plt.tight_layout()
    fig.savefig(f"tmax_return_levels_{rp}yr.png")
    plt.close(fig)