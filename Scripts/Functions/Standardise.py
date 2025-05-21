
import xarray as xr
import numpy as np

def norm_params(ds, var_name):
    """Calculating the gridwise mean and standard deviation for a given variable assuming a normal distribtuion (to be used for Tmeperature"""
    var= ds[var_name]
    mean= var.mean(dim="time", skipna=True).compute()
    std= var.std(dim="time", skipna=True).compute()

    return mean, std

def min_max_calculator(ds, var_name):
    "Calcualting the gridwise min and max for a given variable(to be used for non normally distributed variales such as precipitation)"""
    var= ds[var_name]
    min= var.min(dim="time", skipna=True).compute()
    max= var.max(dim="time", skipna=True).compute()

    return min,max

#Depending on the distribution of the var (precip or temperature) , now standardisation will be performed

def normalise(var, mean, std):
    return (var-mean)/std

def min_max_scaler(var, min, max):
    return (var-min)/(max-min)


def standardise(input_path, output_path, var, mean=None, std=None, min=None, max=None):
    ds = xr.open_dataset(input_path, chunks={"time": 100})
    
    if var in ["pr", "RhiresD"]: 
        if min is None or max is None:
            min, max = min_max_calculator(ds, var)
        scaled_var = min_max_scaler(ds[var], min, max)

    elif var in ["tas", "TabsD","TminD","TmaxD"]: 
        if mean is None or std is None:
            mean, std = norm_params(ds, var)
        scaled_var = normalise(ds[var], mean, std)
    
    else:
        raise ValueError("Unknown name of the variable.")
    
    scaled_ds = scaled_var.to_dataset(name=var)
    scaled_ds.to_netcdf(output_path)
    print(f"Normalised/scaled variable saved to {output_path}")
    
    if var in ["pr", "RhiresD"]:
        return min, max
    else:
        return mean, std

