import numpy as np
import xarray as xr
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd


def psnr(predictions, targets):
    mse = float(np.mean((predictions - targets) ** 2))
    max_pixel_value = np.max(targets)
    return float(20 * np.log10(max_pixel_value / np.sqrt(mse)))