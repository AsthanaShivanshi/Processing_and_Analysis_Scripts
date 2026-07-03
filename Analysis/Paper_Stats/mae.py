import numpy as np
import pandas as pd


def spatial_mean_ts(da):
    spatial_dims = [d for d in da.dims if d != "time"]
    return da.mean(dim=spatial_dims, skipna=True)

def mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))