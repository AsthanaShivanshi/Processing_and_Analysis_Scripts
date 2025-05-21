import xarray as xr
import numpy as np

def gridded_R_squared(pred_path, truth_path, var1, var2, chunk_size={'time': 50}):
    """
    Calculate grid-wise coefficient of determination (R^2) between two gridded variables, ignoring NaNs.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    
    # Residual sum of squares
    ss_res = ((var1_data - var2_data) ** 2).where(valid_mask).sum(dim='time')
    
    # Total sum of squares
    mean_true = var2_data.where(valid_mask).mean(dim='time')
    ss_tot = ((var2_data - mean_true) ** 2).where(valid_mask).sum(dim='time')
    
    r2 = 1 - (ss_res / ss_tot)
    r2 = r2.astype(np.float32)
    
    return r2



def pooled_R_squared(pred_path, truth_path, var1, var2, chunk_size={'time': 100}):
    """
    Renders pooled R squared by pooling all values across all time steps and grid cells.
    """
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    
    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2])
    
    valid_mask = (~np.isnan(var1_data)) & (~np.isnan(var2_data))
    valid_mask=valid_mask.compute()
    diff_squared = (var1_data - var2_data) ** 2
    diff_squared = diff_squared.where(valid_mask)
    ss_res = diff_squared.sum()
    
    truth_valid = var2_data.where(valid_mask)
    truth_mean = truth_valid.mean()
    
    ss_tot = ((truth_valid - truth_mean) ** 2).sum()
    
    r2_total = 1.0 - (ss_res / ss_tot) #1-ratio(residual sum of squares/total sum of squares)
    
    r2_total = r2_total.astype(np.float32)
    
    return r2_total





