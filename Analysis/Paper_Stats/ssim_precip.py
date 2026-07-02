import pandas as pd
import xarray as xr
import numpy as np
from skimage.metrics import structural_similarity

def framewise_ssim(obs, pred, mask2d=None):
    ssim_frames = []
    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values
        if mask2d is not None:
            valid_mask = mask2d.values
        else:
            valid_mask = ~np.isnan(obs_frame) & ~np.isnan(pred_frame)
        if not np.any(valid_mask):
            ssim_frames.append(np.nan)
            continue
        obs_filled = np.where(valid_mask, obs_frame, np.nanmean(obs_frame[valid_mask]))
        pred_filled = np.where(valid_mask, pred_frame, np.nanmean(pred_frame[valid_mask]))
        data_range = obs_filled[valid_mask].max() - obs_filled[valid_mask].min()
        if data_range == 0:
            ssim_frames.append(np.nan)
            continue
        try:
            ssim = structural_similarity(obs_filled, pred_filled, data_range=data_range)
        except Exception:
            ssim = np.nan
        ssim_frames.append(ssim)
    return np.nanmean(ssim_frames)

