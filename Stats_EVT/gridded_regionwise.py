import xarray as xr
import pandas as pd
import numpy as np
import argparse
from major_return_levels_bm import get_extreme_return_levels_bm
import sys
import config

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=int, required=True, help='Region out of 5 climate regions of CH')
args = parser.parse_args()
region_num = args.region

#Masks for regions
mask_da_coarse = xr.open_dataset("swiss_mask_on_coarse_grid_latlon.nc")["swiss_mask_on_coarse_grid"]
region_mask_coarse = (mask_da_coarse == region_num)
mask_da_hr = xr.open_dataset("swiss_mask_on_hr_grid_latlon.nc")["swiss_mask_on_hr_grid"]
region_mask_hr = (mask_da_hr == region_num)

ds_coarse = xr.open_dataset(f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc")
lat_vals_coarse = ds_coarse['lat'].values
lon_vals_coarse = ds_coarse['lon'].values

ds_hr_example = xr.open_dataset(f"{config.TARGET_DIR}/TmaxD_1971_2023.nc")
lat_vals_hr = ds_hr_example['lat'].values
lon_vals_hr = ds_hr_example['lon'].values

return_periods = [20, 50, 100]
rmse_table = pd.DataFrame(index=return_periods, columns=["COARSE", "EQM", "EQM_UNET", "DOTC", "DOTC_UNET", "QDM", "QDM_UNET"])
baseline_info_hr = {
    "OBS": {
        "nc_file": f"{config.TARGET_DIR}/TmaxD_1971_2023.nc",
        "variable_name": "TmaxD"
    },
    "EQM": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/EQM/tmax_BC_bicubic_r01.nc",
        "variable_name": "tmax"
    },
    "EQM_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/EQM/DOWNSCALED_TRAINING_QM_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "tmax"
    },
    "DOTC": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
        "variable_name": "tmax"
    },
    "DOTC_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/dOTC/DOWNSCALED_TRAINING_DOTC_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "tmax"
    },
    "QDM": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/QDM/tmax_BC_bicubic_r01.nc",
        "variable_name": "tmax"
    },
    "QDM_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/QDM/DOWNSCALED_TRAINING_QDM_BC_tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "tmax"
    }
}

ds_hr_baselines = {name: xr.open_dataset(info["nc_file"]) for name, info in baseline_info_hr.items()}

for rp in return_periods:
    # COARSE : single file only for model run
    coarse_rl = []
    for i, lat in enumerate(lat_vals_coarse):
        for j, lon in enumerate(lon_vals_coarse):
            ts = ds_coarse['tmax'][:, i, j].values
            if region_mask_coarse.values[i, j] and ts.size > 0 and not np.isnan(ts).all():
                rl = get_extreme_return_levels_bm(
                    nc_file=f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc",
                    variable_name="tmax",
                    lat=lat,
                    lon=lon,
                    block_size='365D',
                    return_periods=[rp],
                    return_all_periods=False
                )["return value"].values[0]
                coarse_rl.append(rl)
            else:
                coarse_rl.append(np.nan)

    # COARSE df
    lat_grid_coarse, lon_grid_coarse = np.meshgrid(lat_vals_coarse, lon_vals_coarse, indexing='ij')
    df_coarse = pd.DataFrame({
        "lat": lat_grid_coarse.flatten(),
        "lon": lon_grid_coarse.flatten(),
        "COARSE": coarse_rl
    })
    df_coarse_masked = df_coarse[region_mask_coarse.values.flatten()]

    # HR 
    results_hr = {name: [] for name in baseline_info_hr}
    for i, lat in enumerate(lat_vals_hr):
        for j, lon in enumerate(lon_vals_hr):
            for name, info in baseline_info_hr.items():
                ts_hr = ds_hr_baselines[name][info["variable_name"]][:, i, j].values
                if region_mask_hr.values[i, j] and ts_hr.size > 0 and not np.isnan(ts_hr).all():
                    rl = get_extreme_return_levels_bm(
                        nc_file=info["nc_file"],
                        variable_name=info["variable_name"],
                        lat=lat,
                        lon=lon,
                        block_size='365D',
                        return_periods=[rp],
                        return_all_periods=False
                    )["return value"].values[0]
                    results_hr[name].append(rl)
                else:
                    results_hr[name].append(np.nan)

    # HR df
    lat_grid_hr, lon_grid_hr = np.meshgrid(lat_vals_hr, lon_vals_hr, indexing='ij')
    df_hr = pd.DataFrame({
        "lat": lat_grid_hr.flatten(),
        "lon": lon_grid_hr.flatten(),
        **results_hr
    })
    df_hr_masked = df_hr[region_mask_hr.values.flatten()]

    # RMSE
    for baseline in rmse_table.columns:
        valid_mask = (~df_hr_masked["OBS"].isna()) & (~df_hr_masked[baseline].isna())
        rmse = ((df_hr_masked.loc[valid_mask, "OBS"] - df_hr_masked.loc[valid_mask, baseline]) ** 2).mean() ** 0.5
        rmse_table.loc[rp, baseline] = rmse

    # RMSE for coarse field : pooled
    valid_mask_coarse = (~df_hr_masked["OBS"].isna()) & (~df_coarse_masked["COARSE"].isna())
    rmse_table.loc[rp, "COARSE"] = ((df_hr_masked.loc[valid_mask_coarse, "OBS"] - df_coarse_masked.loc[valid_mask_coarse, "COARSE"]) ** 2).mean() ** 0.5
print(f"Pooled RMSE table for Swiss region {region_num}:")
rmse_table.to_csv(f"swiss_region{region_num}_rmse_table_tmax.csv")

for ds in ds_hr_baselines.values():
    ds.close()