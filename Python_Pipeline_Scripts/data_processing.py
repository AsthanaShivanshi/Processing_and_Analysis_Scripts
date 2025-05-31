import os
import yaml
import subprocess
import xarray as xr
import numpy as np

def run(cmd):
    print(f"\nRunning command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def norm_params(ds, var_name):
    var = ds[var_name]
    mean = var.mean(dim="time", skipna=True).compute()
    std = var.std(dim="time", skipna=True).compute()
    return mean, std

def min_max_calculator(ds, var_name):
    var = ds[var_name]
    min_ = var.min(dim="time", skipna=True).compute()
    max_ = var.max(dim="time", skipna=True).compute()
    return min_, max_

def normalise(var, mean, std):
    return (var - mean) / std

def min_max_scaler(var, min_, max_):
    return (var - min_) / (max_ - min_)

def standardise(input_path, output_path, var, mean=None, std=None, min_=None, max_=None):
    ds = xr.open_dataset(input_path, chunks={"time": 100})

    if var in ["pr", "RhiresD"]:
        if min_ is None or max_ is None:
            min_, max_ = min_max_calculator(ds, var)
        scaled_var = min_max_scaler(ds[var], min_, max_)
        params = {"min": min_.values, "max": max_.values}
    elif var in ["tas", "TabsD", "TminD", "TmaxD"]:
        if mean is None or std is None:
            mean, std = norm_params(ds, var)
        scaled_var = normalise(ds[var], mean, std)
        params = {"mean": mean.values, "std": std.values}
    else:
        raise ValueError(f"Unknown variable: {var}")

    scaled_ds = scaled_var.to_dataset(name=var)
    scaled_ds.to_netcdf(output_path)
    print(f"Standardized {output_path}")
    return params

# processing data
with open("config_files/baselines_and_split.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

masked_dir = os.path.join(output_dir, "masked")
baseline_dir = os.path.join(output_dir, "baseline")
standardised_dir = os.path.join(output_dir, "standardised")

for d in [masked_dir, baseline_dir, standardised_dir]:
    os.makedirs(d, exist_ok=True)

#Creating wet-day mask (from coarse RhiresD only)
wet_mask_file = os.path.join(masked_dir, "coarse_wet_mask.nc")
if config["use_wet_mask"]:
    varname = config["wet_mask_variable"]
    threshold = config["wet_mask_threshold"]
    expr = f'{varname}=({varname}>={threshold})'
    cmd = f'cdo -expr,"{expr}" {config["coarse_precip_file"]} {wet_mask_file}'
else:
    cmd = f"cdo setmisstoc,0 {config['coarse_precip_file']} {wet_mask_file}"
run(cmd)

# Masking coarse files using ifthen (NaNs for dry days)
masked_coarse = {}
for var, file in config["coarse_vars"].items():
    out_file = os.path.join(masked_dir, f"coarse_masked_{var}.nc")
    run(f"cdo ifthen {wet_mask_file} {file} {out_file}")
    masked_coarse[var] = out_file

# Remapping mask to HR  (Using NN)
hr_mask_file = os.path.join(masked_dir, "highres_mask.nc")
run(f"cdo remapnn,{config['highres_template']} {wet_mask_file} {hr_mask_file}")

# Masking HR  using remapped mask
masked_hr = {}
for var, file in config["highres_vars"].items():
    out_file = os.path.join(masked_dir, f"hr_masked_{var}.nc")
    run(f"cdo ifthen {hr_mask_file} {file} {out_file}")
    masked_hr[var] = out_file

# Step 5: Temporal split (train, val, test)
splits = {
    "train": config["train_period"],
    "val": config["val_period"],
    "test": config["test_period"]
}
split_files = {"coarse": {}, "hr": {}}

for split, years in splits.items():
    for var, file in masked_coarse.items():
        out_file = os.path.join(masked_dir, f"{split}_coarse_{var}.nc")
        run(f"cdo selyear,{years[0]}/{years[1]} {file} {out_file}")
        split_files["coarse"].setdefault(split, {})[var] = out_file
    for var, file in masked_hr.items():
        out_file = os.path.join(masked_dir, f"{split}_hr_{var}.nc")
        run(f"cdo selyear,{years[0]}/{years[1]} {file} {out_file}")
        split_files["hr"].setdefault(split, {})[var] = out_file

# Bicubic interpolation 
baseline_files = {}
if config.get("generate_bicubic_baseline", True):
    for var, file in masked_coarse.items():
        out_file = os.path.join(baseline_dir, f"baseline_hr_{var}.nc")
        run(f"cdo remapbic,{config['highres_template']} {file} {out_file}")
        baseline_files[var] = out_file

    # Splitting baseline files 
    for split, years in splits.items():
        for var, file in baseline_files.items():
            out_file = os.path.join(baseline_dir, f"{split}_baseline_hr_{var}.nc")
            run(f"cdo selyear,{years[0]}/{years[1]} {file} {out_file}")
            split_files.setdefault("baseline", {}).setdefault(split, {})[var] = out_file

# Standardising using training stats
norm_params_dict = {}
for var in ["RhiresD", "TabsD", "TminD", "TmaxD"]:
    # HR Standardization
    train_file = split_files["hr"]["train"][var]
    std_train_out = os.path.join(standardised_dir, f"std_train_hr_{var}.nc")
    params = standardise(train_file, std_train_out, var)
    norm_params_dict[f"{var}_target"] = params

    for split in ["val", "test"]:
        in_file = split_files["hr"][split][var]
        out_file = os.path.join(standardised_dir, f"std_{split}_hr_{var}.nc")
        standardise(in_file, out_file, var, **params)

    # Baseline standardization
    if config.get("generate_bicubic_baseline", True):
        train_file = split_files["baseline"]["train"][var]
        std_train_base = os.path.join(standardised_dir, f"std_train_baseline_hr_{var}.nc")
        base_params = standardise(train_file, std_train_base, var)
        norm_params_dict[f"{var}_baseline"] = base_params

        for split in ["val", "test"]:
            in_file = split_files["baseline"][split][var]
            out_file = os.path.join(standardised_dir, f"std_{split}_baseline_hr_{var}.nc")
            standardise(in_file, out_file, var, **base_params)

# Saving normalization parameters
np.savez(os.path.join(output_dir, "norm_params.npz"), **norm_params_dict)
print("\n Normalization parameters saved.")
