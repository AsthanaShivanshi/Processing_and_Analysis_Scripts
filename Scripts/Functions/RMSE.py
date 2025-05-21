import xarray as xr
import numpy as np

def gridded_RMSE(pred_path, truth_path, var1, var2, chunk_size={'time': 100}):
    """
    Calculate grid-wise RMSE between two gridded variables, ignoring NaNs.
    Loads data with manual chunking and outputs float32 dtype.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    # Align the variables
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    # Mask valid data (non-NaN)
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    
    diff_squared = (var1_data - var2_data) ** 2
    diff_squared = diff_squared.where(valid_mask)
    
    mse = diff_squared.sum(dim='time') / valid_mask.sum(dim='time')
    
    rmse = np.sqrt(mse)
    
    rmse = rmse.astype(np.float32)
    
    return rmse

def pooled_RMSE(pred_path, truth_path, var1, var2, chunk_size={'time': 100}):
    """
    Calculate pooled RMSE across all time steps and grid cells.
    Outputs a single float32 scalar.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    
    diff_squared = (var1_data - var2_data) ** 2
    diff_squared = diff_squared.where(valid_mask)
    
    mse_total = diff_squared.sum() / valid_mask.sum()
    
    rmse_total = np.sqrt(mse_total)
    
    rmse_total = rmse_total.astype(np.float32)
    
    return rmse_total


