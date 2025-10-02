import xarray as xr
import pandas as pd
import numpy as np
import argparse
from major_return_levels_bm import get_extreme_return_levels_bm
import sys
sys.path.append('../Prelim_Stats')
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

ds_coarse = xr.open_dataset(f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc")
lat_vals_coarse = ds_coarse['lat'].values
lon_vals_coarse = ds_coarse['lon'].values

ds_hr_example = xr.open_dataset(f"{config.TARGET_DIR}/RhiresD_1971_2023.nc")
lat_vals_hr = ds_hr_example['lat'].values
lon_vals_hr = ds_hr_example['lon'].values

return_periods = [20, 50, 100]
rmse_table = pd.DataFrame(index=return_periods, columns=["COARSE", "EQM", "EQM_UNET", "DOTC", "DOTC_UNET", "QDM", "QDM_UNET"])
baseline_info_hr = {
    "OBS": {
        "nc_file": f"{config.TARGET_DIR}/RhiresD_1971_2023.nc",
        "variable_name": "RhiresD"
    },
    "EQM": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/EQM/precip_BC_bicubic_r01.nc",
        "variable_name": "precip"
    },
    "EQM_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/EQM/DOWNSCALED_TRAINING_QM_BC_precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "precip"
    },
    "DOTC": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
        "variable_name": "precip"
    },
    "DOTC_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/dOTC/DOWNSCALED_TRAINING_DOTC_BC_precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "precip"
    },
    "QDM": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/QDM/precip_BC_bicubic_r01.nc",
        "variable_name": "precip"
    },
    "QDM_UNET": {
        "nc_file": f"{config.BIAS_CORRECTED_DIR}/QDM/DOWNSCALED_TRAINING_QDM_BC_precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_gridset_r01.nc",
        "variable_name": "precip"
    }
}

ds_hr_baselines = {name: xr.open_dataset(info["nc_file"]) for name, info in baseline_info_hr.items()}

for rp in return_periods:
    coarse_rl = []
    for i, lat in enumerate(lat_vals_coarse):
        for j, lon in enumerate(lon_vals_coarse):
            if region_mask_coarse.values[i, j]:
                ts = ds_coarse['precip'][:, i, j].values
                if np.isnan(ts).all():
                    coarse_rl.append(np.nan)
                    continue
                rl = get_extreme_return_levels_bm(
                    nc_file=f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc",
                    variable_name="precip",
                    lat=lat,
                    lon=lon,
                    block_size='365D',
                    return_periods=[rp],
                    return_all_periods=False
                )["return value"].values[0]
                coarse_rl.append(rl)
            else:
                coarse_rl.append(np.nan)
    df_coarse = pd.DataFrame({
        "lat": [lat for lat in lat_vals_coarse for _ in lon_vals_coarse],
        "lon": [lon for _ in lat_vals_coarse for lon in lon_vals_coarse],
        "COARSE": coarse_rl
    })
    df_coarse_masked = df_coarse[region_mask_coarse.values.flatten()]

    # HR grid , except coarse
    results_hr = {name: [] for name in baseline_info_hr}
    for i, lat in enumerate(lat_vals_hr):
        for j, lon in enumerate(lon_vals_hr):
            if region_mask_hr.values[i, j]:
                for name, info in baseline_info_hr.items():
                    ds_hr_baseline = ds_hr_baselines[name]  # Use already opened dataset
                    ts_hr = ds_hr_baseline[info["variable_name"]][:, i, j].values
                    if np.isnan(ts_hr).all():
                        results_hr[name].append(np.nan)
                        continue
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
                for name in results_hr:
                    results_hr[name].append(np.nan)
    df_hr = pd.DataFrame({
        "lat": [lat for lat in lat_vals_hr for _ in lon_vals_hr],
        "lon": [lon for _ in lat_vals_hr for lon in lon_vals_hr],
        **results_hr
    })
    df_hr_masked = df_hr[region_mask_hr.values.flatten()]

    # RMSE
    for baseline in rmse_table.columns:
        valid_mask = (~df_hr_masked["OBS"].isna()) & (~df_hr_masked[baseline].isna())
        rmse = ((df_hr_masked.loc[valid_mask, "OBS"] - df_hr_masked.loc[valid_mask, baseline]) ** 2).mean() ** 0.5
        rmse_table.loc[rp, baseline] = rmse

    # RMSE
    valid_mask_coarse = (~df_hr_masked["OBS"].isna()) & (~df_coarse_masked["COARSE"].isna())
    rmse_table.loc[rp, "COARSE"] = ((df_hr_masked.loc[valid_mask_coarse, "OBS"] - df_coarse_masked.loc[valid_mask_coarse, "COARSE"]) ** 2).mean() ** 0.5

print(f"Pooled RMSE table for Swiss region {region_num}:")
rmse_table.to_csv(f"swiss_region{region_num}_rmse_table.csv")

for ds in ds_hr_baselines.values():
    ds.close()