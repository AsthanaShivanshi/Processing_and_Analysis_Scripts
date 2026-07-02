import pandas as pd
import xarray as xr
import numpy as np
from skimage.metrics import structural_similarity

def framewise_ssim(obs, pred):
    ssim_frames = []
    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values
        mask = ~np.isnan(obs_frame) & ~np.isnan(pred_frame)
        if not np.any(mask):
            ssim_frames.append(np.nan)
            continue
        obs_filled = np.where(mask, obs_frame, np.nanmean(obs_frame[mask]))
        pred_filled = np.where(mask, pred_frame, np.nanmean(pred_frame[mask]))
        data_range = obs_filled[mask].max() - obs_filled[mask].min()
        if data_range == 0:
            ssim_frames.append(np.nan)
            continue
        try:
            ssim = structural_similarity(obs_filled, pred_filled, data_range=data_range)
        except Exception:
            ssim = np.nan
        ssim_frames.append(ssim)
    return np.nanmean(ssim_frames)



