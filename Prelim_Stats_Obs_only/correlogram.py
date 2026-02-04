import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def spatial_correlogram(field, max_lag=30):
    field = field - np.nanmean(field)
    corrs = []
    for lag in range(1, max_lag+1):
        shifted_x = np.roll(field, lag, axis=1)
        shifted_y = np.roll(field, lag, axis=0)
        shifted_x[:, :lag] = np.nan
        shifted_y[:lag, :] = np.nan
        valid_x = ~np.isnan(field) & ~np.isnan(shifted_x)
        valid_y = ~np.isnan(field) & ~np.isnan(shifted_y)
        if np.any(valid_x):
            corr_x = spearmanr(field[valid_x], shifted_x[valid_x])[0]
        else:
            corr_x = np.nan
        if np.any(valid_y):
            corr_y = spearmanr(field[valid_y], shifted_y[valid_y])[0]
        else:
            corr_y = np.nan
        corrs.append(np.nanmean([corr_x, corr_y]))
    return np.array(corrs)

def mask_like(obs, pred):
    return np.where(np.isnan(obs), np.nan, pred)

obs_ds = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
bicubic_ds = xr.open_dataset('../../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc')["RhiresD"].sel(time=slice("2011-01-01","2023-12-31"))
unet_ds = xr.open_dataset("../../Downscaling_Models/LDM_conditional/outputs/test_UNet_baseline_CRPS__UNet_constrained_withoutReLU.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_ds = xr.open_dataset("../../Downscaling_Models/DDIM_conditional_derived/outputs/ddim_downscaled_test_set_second_sample_run.nc")["precip"].sel(time=slice("2011-01-01","2023-12-31"))
ddim_median_ds = ddim_median_ds.rename({'y': 'N', 'x': 'E'})

max_lag = 30
n_time = obs_ds.shape[0]

corrs_obs = []
corrs_bicubic = []
corrs_unet = []
corrs_ddim = []

for t in range(n_time):
    obs_field = obs_ds.isel(time=t).values
    bicubic_field = mask_like(obs_field, bicubic_ds.isel(time=t).values)
    unet_field = mask_like(obs_field, unet_ds.isel(time=t).values)
    ddim_field = mask_like(obs_field, ddim_median_ds.isel(time=t).values)

    corrs_obs.append(spatial_correlogram(obs_field, max_lag=max_lag))
    corrs_bicubic.append(spatial_correlogram(bicubic_field, max_lag=max_lag))
    corrs_unet.append(spatial_correlogram(unet_field, max_lag=max_lag))
    corrs_ddim.append(spatial_correlogram(ddim_field, max_lag=max_lag))

corrs_obs = np.nanmean(corrs_obs, axis=0)
corrs_bicubic = np.nanmean(corrs_bicubic, axis=0)
corrs_unet = np.nanmean(corrs_unet, axis=0)
corrs_ddim = np.nanmean(corrs_ddim, axis=0)

lags = np.arange(1, max_lag+1)
plt.figure(figsize=(8,5))
plt.plot(lags, corrs_obs, label='Obs', color='black')
plt.plot(lags, corrs_bicubic, label='Bicubic', color='blue')
plt.plot(lags, corrs_unet, label='UNet', color='green')
plt.plot(lags, corrs_ddim, label='DDIM single sample', color='red')
plt.xlabel('Grid cell lag')
plt.ylabel('Mean spatial autocorrelation (Spearman)')
plt.title('Mean Spatial Correlogram for Test Set Precipitation (2011-2023)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/mean_spatial_correlogram_precip.png', dpi=1000)
plt.close()