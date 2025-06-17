import os
import yaml
import subprocess
import xarray as xr
import numpy as np
import json

def run(cmd):
    print(f"Running command:\n{cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

def print_shape(path, var):
    ds = xr.open_dataset(path)
    print(f"Saved: {path} | Shape: {ds[var].shape} | Dims: {ds[var].dims}")
    ds.close()

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
        params = {"min_": min_, "max_": max_}
    elif var in ["tas", "TabsD", "TminD", "TmaxD"]:
        if mean is None or std is None:
            mean, std = norm_params(ds, var)
        scaled_var = normalise(ds[var], mean, std)
        params = {"mean": mean, "std": std}
    else:
        raise ValueError(f"Unknown variable: {var}")

    scaled_ds = scaled_var.to_dataset(name=var)
    scaled_ds.to_netcdf(output_path)
    print_shape(output_path, var)
    return params

def make_dirs(base_dir):
    full_path = os.path.join(base_dir, "Full_files")
    split_path = os.path.join(base_dir, "Split_files")
    for folder in [full_path] + [os.path.join(split_path, s) for s in ["Train", "Val", "Test"]]:
        os.makedirs(folder, exist_ok=True)
    return full_path, split_path

def get_out_path(base, kind, split=None, var=None):
    if split:
        return os.path.join(base, "Split_files", split.capitalize(), f"{var}.nc")
    else:
        return os.path.join(base, "Full_files", f"{var}.nc")

# Loading  configuration from the baselines YAML file
with open("config_files/baselines_and_split.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
masked_dir = os.path.join(output_dir, "masked")
baseline_dir = os.path.join(output_dir, "baseline")
standardised_dir = os.path.join(output_dir, "standardised")

for base in [masked_dir, baseline_dir, standardised_dir]:
    make_dirs(base)

# Creating wet day mask from coarse rhiresD dataset
wet_mask_file = os.path.join(masked_dir, "Full_files", "coarse_wet_mask.nc")
if config["use_wet_mask"]:
    expr = f'{config["wet_mask_variable"]}=({config["wet_mask_variable"]}>={config["wet_mask_threshold"]})'
    cmd = f'cdo -expr,"{expr}" {config["coarse_precip_file"]} {wet_mask_file}'
else:
    cmd = f"cdo setmisstoc,0 {config['coarse_precip_file']} {wet_mask_file}"
run(cmd)
print_shape(wet_mask_file, config["wet_mask_variable"])

# Applying mask to coarse files: all four
masked_coarse = {}
for var, path in config["coarse_vars"].items():
    out_path = get_out_path(masked_dir, "masked", None, f"coarse_masked_{var}")
    run(f"cdo ifthen {wet_mask_file} {path} {out_path}")
    print_shape(out_path, var)
    masked_coarse[var] = out_path

# Upscaling mask to HR grid using remapnn
hr_mask_file = os.path.join(masked_dir, "Full_files", "highres_mask.nc")
run(f"cdo remapnn,{config['highres_template']} {wet_mask_file} {hr_mask_file}")
print_shape(hr_mask_file, config["wet_mask_variable"])

# Applying HR mask to HR data: all four files
masked_hr = {}
for var, path in config["highres_vars"].items():
    out_path = get_out_path(masked_dir, "masked", None, f"hr_masked_{var}")
    run(f"cdo ifthen {hr_mask_file} {path} {out_path}")
    print_shape(out_path, var)
    masked_hr[var] = out_path

# Splitting masked HR data
splits = {"train": config["train_period"], "val": config["val_period"], "test": config["test_period"]}
split_files = {"hr": {}, "baseline": {}}

for split, years in splits.items():
    for var, path in masked_hr.items():
        out_path = get_out_path(masked_dir, "masked", split, f"{var}")
        run(f"cdo selyear,{years[0]}/{years[1]} {path} {out_path}")
        print_shape(out_path, var)
        split_files["hr"].setdefault(split, {})[var] = out_path

# Bicubic baseline and split
baseline_files = {}
if config.get("generate_bicubic_baseline", True):
    for var, path in masked_coarse.items():
        out_path = get_out_path(baseline_dir, "baseline", None, f"baseline_hr_{var}")
        run(f"cdo remapbic,{config['highres_template']} {path} {out_path}")
        print_shape(out_path, var)
        baseline_files[var] = out_path

    for split, years in splits.items():
        for var, path in baseline_files.items():
            out_path = get_out_path(baseline_dir, "baseline", split, f"{var}")
            run(f"cdo selyear,{years[0]}/{years[1]} {path} {out_path}")
            print_shape(out_path, var)
            split_files["baseline"].setdefault(split, {})[var] = out_path

# Standardize with train stats only
norm_params_dict = {}
for var in ["RhiresD", "TabsD", "TminD", "TmaxD"]:
    # Standardising HR
    train_path = split_files["hr"]["train"][var]
    train_std_out = get_out_path(standardised_dir, "standardised", "train", f"hr_{var}")
    params = standardise(train_path, train_std_out, var)
    norm_params_dict[f"{var}_target"] = params

    for split in ["val", "test"]:
        in_path = split_files["hr"][split][var]
        out_path = get_out_path(standardised_dir, "standardised", split, f"hr_{var}")
        standardise(in_path, out_path, var, **params)

    # Standardising bicubic baseline
    if config.get("generate_bicubic_baseline", True):
        train_base_path = split_files["baseline"]["train"][var]
        train_base_std_out = get_out_path(standardised_dir, "standardised", "train", f"baseline_hr_{var}")
        base_params = standardise(train_base_path, train_base_std_out, var)
        norm_params_dict[f"{var}_baseline"] = base_params

        for split in ["val", "test"]:
            in_path = split_files["baseline"][split][var]
            out_path = get_out_path(standardised_dir, "standardised", split, f"baseline_hr_{var}")
            standardise(in_path, out_path, var, **base_params)

#Saving standardization parameters for later use

norm_params_serializable = {
    k: {kk: float(vv.values) for kk, vv in v.items()}
    for k, v in norm_params_dict.items()
}

with open(os.path.join(output_dir, "norm_params.yaml"), "w") as f:
    yaml.dump(norm_params_serializable, f)

print(f"\n Normalization parameters saved to: {output_dir}/norm_params.yaml")

