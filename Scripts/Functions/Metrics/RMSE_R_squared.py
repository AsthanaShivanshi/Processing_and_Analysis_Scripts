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
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)
    print(f"🔍 Variable: {var1} vs {var2}")

    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2], join='inner')
    var1_data = var1_data.load()
    var2_data = var2_data.load()

    diff = var1_data - var2_data
    truth = var2_data

    # Stacking and dropping NaNs globally
    diff_sq_flat = (diff ** 2).stack(points=diff.dims).dropna(dim='points')
    truth_flat = truth.stack(points=truth.dims).dropna(dim='points')

    if diff_sq_flat.size == 0 or truth_flat.size == 0:
        return xr.DataArray(np.nan)

    ss_res = diff_sq_flat.sum()
    mean_truth = truth_flat.mean()
    ss_tot = ((truth_flat - mean_truth) ** 2).sum()

    r2 = 1.0 - (ss_res / ss_tot)
    return r2.astype(np.float32)



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
    ds_pred = xr.open_dataset(pred_path, chunks=chunk_size)
    ds_true = xr.open_dataset(truth_path, chunks=chunk_size)

    print(f"🔍 Variable: {var1} vs {var2}")

    var1_data, var2_data = xr.align(ds_pred[var1], ds_true[var2], join='inner')
    var1_data = var1_data.load()
    var2_data = var2_data.load()

    # Compute squared differences and flatten
    diff = var1_data - var2_data
    diff_squared = diff ** 2

    # Stack across all dims (e.g., time, lat, lon)
    flat_diff = diff_squared.stack(points=diff_squared.dims).dropna(dim='points')

    if flat_diff.size == 0:
        return xr.DataArray(np.nan)

    mse = flat_diff.mean()
    rmse = np.sqrt(mse)

    return rmse.astype(np.float32)









