import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from closest_grid_cell import select_nearest_grid_cell
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor

np.Inf = np.inf

cities = {
    "Locarno": (46.1670, 8.7943),
    "Bern": (46.9480, 7.4474),
    "Chur": (46.8508, 9.5320),
    "Sion": (46.2331, 7.3606),
    "Interlaken": (46.6863, 7.8632),
    "Neuchatel": (46.9901, 6.9246)
}

obs_ds = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')
coarse_ds = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc')
bicubic_ds = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc')
unet_ds = xr.open_dataset("../../Downscaling_Models/LDM_conditional/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc")
ddim_median_ds = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")

def clean_and_slice(arr):
    arr = arr.sel(time=slice("2011-01-01", "2023-12-31"))
    arr = arr.values
    arr = np.squeeze(arr)
    return arr[~np.isnan(arr)]

def process_city(city, lat, lon):
    obs = clean_and_slice(select_nearest_grid_cell(obs_ds, lat, lon, var_name="RhiresD")['data'])
    coarse = clean_and_slice(select_nearest_grid_cell(coarse_ds, lat, lon, var_name="RhiresD")['data'])
    bicubic = clean_and_slice(select_nearest_grid_cell(bicubic_ds, lat, lon, var_name="RhiresD")['data'])
    unet = clean_and_slice(select_nearest_grid_cell(unet_ds, lat, lon, var_name="precip")['data'])
    ddim_median = clean_and_slice(select_nearest_grid_cell(ddim_median_ds, lat, lon, var_name="precip")['data'])

    quantiles = np.linspace(0, 1, 101)
    obs_q = np.quantile(obs, quantiles)
    coarse_q = np.quantile(coarse, quantiles)
    bicubic_q = np.quantile(bicubic, quantiles)
    unet_q = np.quantile(unet, quantiles)
    ddim_median_q = np.quantile(ddim_median, quantiles)

    plt.figure(figsize=(8, 5))
    plt.plot(quantiles, coarse_q - obs_q, label='Coarse', color='#E69F00')      
    plt.plot(quantiles, bicubic_q - obs_q, label='Bicubic', color='#56B4E9')   
    plt.plot(quantiles, unet_q - obs_q, label='UNet', color='#009E73')         
    plt.plot(quantiles, ddim_median_q - obs_q, label='DDIM 4 sample mean', color='#D55E00') 

    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Quantile')
    plt.ylabel('Bias (Model - Obs) (mm/day)')
    plt.title(f'Bias of Quantiles: {city} (2011-2023)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/quantile_bias_{city}.png')
    plt.close()

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_city, city, lat, lon) for city, (lat, lon) in cities.items()]
    for f in futures:
        f.result()