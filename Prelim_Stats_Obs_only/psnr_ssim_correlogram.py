import xarray as xr
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor

np.Inf = np.inf  

def compute_metrics(obs, pred):
    def single_metric(t):
        hr_img = obs.isel(time=t).values
        pred_img = pred.isel(time=t).values
        mask = ~np.isnan(hr_img) & ~np.isnan(pred_img)
        if not np.any(mask):
            return np.nan, np.nan
        hr_img = hr_img[mask]
        pred_img = pred_img[mask]
        if hr_img.size == 0 or hr_img.max() == hr_img.min():
            return np.nan, np.nan
        psnr = peak_signal_noise_ratio(hr_img, pred_img, data_range=hr_img.max() - hr_img.min())
        try:
            ssim = structural_similarity(hr_img, pred_img, data_range=hr_img.max() - hr_img.min())
        except:
            ssim = np.nan
        return psnr, ssim
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(single_metric, range(obs.shape[0])))
    psnr_list, ssim_list = zip(*results)
    return np.nanmean(psnr_list), np.nanmean(ssim_list)


def spatial_correlogram(field, max_lag=10):
    field = field - np.nanmean(field)
    corrs = []
    for lag in range(1, max_lag + 1):
        shifted_x = np.roll(field, lag, axis=1)
        shifted_y = np.roll(field, lag, axis=0)
        shifted_x[:, :lag] = np.nan
        shifted_y[:lag, :] = np.nan
        valid_x = ~np.isnan(field) & ~np.isnan(shifted_x)
        valid_y = ~np.isnan(field) & ~np.isnan(shifted_y)
        corr_x = np.corrcoef(field[valid_x], shifted_x[valid_x])[0,1] if np.any(valid_x) else np.nan
        corr_y = np.corrcoef(field[valid_y], shifted_y[valid_y])[0,1] if np.any(valid_y) else np.nan
        corrs.append(np.nanmean([corr_x, corr_y]))
    return np.array(corrs)

# Load datasets
obs_temp = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_temp = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_temp = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp.nc')["TabsD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_temp = xr.open_dataset('../../Downscaling_Models/DDIM_conditional_derived/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc')["temp"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_temp = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")["temp"].sel(time=slice("2011-01-01","2023-12-31"))

obs_precip = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
coarse_precip = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_precip = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_precip = xr.open_dataset('../../Downscaling_Models/LDM_conditional/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc')["precip"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_precip = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))

# Interpolate coarse and bicubic fields to obs grid for fair comparison
coarse_temp_hr = coarse_temp.interp(N=obs_temp.N, E=obs_temp.E)
bicubic_temp_hr = bicubic_temp.interp(N=obs_temp.N, E=obs_temp.E)
unet_temp_hr = unet_temp.assign_coords(N=obs_temp.N, E=obs_temp.E)
ddim_median_temp_hr = ddim_median_temp.assign_coords(N=obs_temp.N, E=obs_temp.E)

coarse_precip_hr = coarse_precip.interp(N=obs_precip.N, E=obs_precip.E)
bicubic_precip_hr = bicubic_precip.interp(N=obs_precip.N, E=obs_precip.E)
unet_precip_hr = unet_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)
ddim_median_precip_hr = ddim_median_precip.assign_coords(N=obs_precip.N, E=obs_precip.E)

# Metrics tables
temp_metrics = {
    'Coarse': compute_metrics(obs_temp, coarse_temp_hr),
    'Bicubic': compute_metrics(obs_temp, bicubic_temp_hr),
    'UNet': compute_metrics(obs_temp, unet_temp_hr),
    'DDIM 5 four sample mean': compute_metrics(obs_temp, ddim_median_temp_hr)
}
temp_df = pd.DataFrame(temp_metrics, index=['PSNR', 'SSIM']).T
temp_df.to_csv('temp_metrics.csv')
print("Temperature metrics:")
print(temp_df)

precip_metrics = {
    'Coarse': compute_metrics(obs_precip, coarse_precip_hr),
    'Bicubic': compute_metrics(obs_precip, bicubic_precip_hr),
    'UNet': compute_metrics(obs_precip, unet_precip_hr),
    'DDIM 5 four sample mean': compute_metrics(obs_precip, ddim_median_precip_hr)
}
precip_df = pd.DataFrame(precip_metrics, index=['PSNR', 'SSIM']).T
precip_df.to_csv('precip_metrics.csv')
print("Precipitation metrics:")
print(precip_df)

# Spatial correlogram
max_lag = 15
n_time = obs_temp.shape[0]
corrs_obs, corrs_coarse, corrs_bicubic, corrs_unet, corrs_ddim_median = [], [], [], [], []

for t in range(n_time):
    corrs_obs.append(spatial_correlogram(obs_temp.isel(time=t).values, max_lag=max_lag))
    corrs_coarse.append(spatial_correlogram(coarse_temp_hr.isel(time=t).values, max_lag=max_lag))
    corrs_bicubic.append(spatial_correlogram(bicubic_temp_hr.isel(time=t).values, max_lag=max_lag))
    corrs_unet.append(spatial_correlogram(unet_temp_hr.isel(time=t).values, max_lag=max_lag))
    corrs_ddim_median.append(spatial_correlogram(ddim_median_temp_hr.isel(time=t).values, max_lag=max_lag))

corrs_obs = np.nanmean(corrs_obs, axis=0)
corrs_coarse = np.nanmean(corrs_coarse, axis=0)
corrs_bicubic = np.nanmean(corrs_bicubic, axis=0)
corrs_unet = np.nanmean(corrs_unet, axis=0)
corrs_ddim_median = np.nanmean(corrs_ddim_median, axis=0)
lags = np.arange(1, max_lag+1)
plt.figure(figsize=(8,5))
plt.plot(lags, corrs_obs, label='Obs', color='black')
plt.plot(lags, corrs_coarse, label='Coarse', color='orange')
plt.plot(lags, corrs_bicubic, label='Bicubic', color='blue')
plt.plot(lags, corrs_unet, label='UNet', color='green')
plt.plot(lags, corrs_ddim_median, label='DDIM Median', color='red')
plt.xlabel('Spatial lag (grid cells)')
plt.ylabel('Mean spatial autocorrelation')
plt.title('Mean Spatial Correlogram for Temp (time mean test set)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/spatial_correlogram_temp.png')
plt.close()