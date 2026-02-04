import xarray as xr
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor

np.Inf = np.inf  

def compute_metrics(obs, pred):
    psnr_list, ssim_list, rmse_list = [], [], []
    for t in range(obs.shape[0]):
        hr_img = obs.isel(time=t).values.squeeze()
        pred_img = pred.isel(time=t).values.squeeze()
        mask = ~np.isnan(hr_img)
        hr_img = hr_img[mask]
        pred_img = pred_img[mask]
        if not np.any(mask):
            psnr_list.append(np.nan)
            ssim_list.append(np.nan)
            rmse_list.append(np.nan)
            continue
        if hr_img.max() == hr_img.min():
            psnr_list.append(np.nan)
            ssim_list.append(np.nan)
            rmse_list.append(np.nan)
            continue
        psnr = peak_signal_noise_ratio(hr_img, pred_img, data_range=hr_img.max() - hr_img.min())
        try:
            ssim = structural_similarity(hr_img, pred_img, data_range=hr_img.max() - hr_img.min())
        except:
            ssim = np.nan
        rmse = np.sqrt(mean_squared_error(hr_img, pred_img))
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        rmse_list.append(rmse)
    return np.nanmean(psnr_list), np.nanmean(ssim_list), np.nanmean(rmse_list)

obs_temp = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_temp = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_temp = xr.open_dataset('../../Downscaling_Models/LDM_conditional/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc')["temp"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_temp = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_temp = ddim_median_temp.rename({'y': 'N', 'x': 'E'})

obs_precip = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_precip = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_precip = xr.open_dataset('../../Downscaling_Models/LDM_conditional/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc')["precip"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_precip = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_precip = ddim_median_precip.rename({'y': 'N', 'x': 'E'})

print("obs_temp shape:", obs_temp.shape)
print("bicubic_temp shape:", bicubic_temp.shape)
print("unet_temp shape:", unet_temp.shape)
print("ddim_median_temp shape:", ddim_median_temp.shape)
print("obs_precip shape:", obs_precip.shape)
print("bicubic_precip shape:", bicubic_precip.shape)
print("unet_precip shape:", unet_precip.shape)
print("ddim_median_precip shape:", ddim_median_precip.shape)



bicubic_temp_hr = bicubic_temp
unet_temp_hr = unet_temp
ddim_median_temp_hr = ddim_median_temp

bicubic_precip_hr = bicubic_precip
unet_precip_hr = unet_precip
ddim_median_precip_hr = ddim_median_precip




temp_metrics = {
    'Bicubic': compute_metrics(obs_temp, bicubic_temp_hr),
    'UNet': compute_metrics(obs_temp, unet_temp_hr),
    'DDIM': compute_metrics(obs_temp, ddim_median_temp_hr)
}

temp_df = pd.DataFrame(temp_metrics, index=['PSNR', 'SSIM', 'RMSE']).T
temp_df.to_csv('temp_metrics.csv')
print("Temperature metrics:")
print(temp_df)

unet_precip = unet_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)
ddim_median_precip = ddim_median_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)

def mask_like(obs, pred):
    return pred.where(~np.isnan(obs))

bicubic_precip_masked = mask_like(obs_precip, bicubic_precip)
unet_precip_masked = mask_like(obs_precip, unet_precip)
ddim_median_precip_masked = mask_like(obs_precip, ddim_median_precip)

precip_metrics = {
    'Bicubic': compute_metrics(obs_precip, bicubic_precip_masked),
    'UNet': compute_metrics(obs_precip, unet_precip_masked),
    'DDIM': compute_metrics(obs_precip, ddim_median_precip_masked)
}
precip_df = pd.DataFrame(precip_metrics, index=['PSNR', 'SSIM', 'RMSE']).T
precip_df.to_csv('precip_metrics.csv')
print("Precipitation metrics:")
print(precip_df)
