import xarray as xr
import numpy as np
import os
from pathlib import Path
import subprocess
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Data_pretraining"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
SEED = 42

def conservative_coarsening(infile, varname, block_size, outfile, latname='lat', lonname='lon'):
    ds = xr.open_dataset(infile)
    da = ds[varname]
    dims = da.dims
    has_time = 'time' in dims

    if 'E' in dims and 'N' in dims:
        da_coarse = da.coarsen(N=block_size, E=block_size, boundary='pad').mean()
    elif latname in ds and lonname in ds:
        lat = ds[latname].values
        lon = ds[lonname].values
        var = da

        if lat.shape != var.shape[-2:] or lon.shape != var.shape[-2:]:
            raise ValueError("lat/lon shape mismatch")

        ny, nx = lat.shape
        ny_pad = (block_size - ny % block_size) % block_size
        nx_pad = (block_size - nx % block_size) % block_size

        var = var.pad({var.dims[-2]: (0, ny_pad), var.dims[-1]: (0, nx_pad)}, mode='edge')
        lat = np.pad(lat, ((0, ny_pad), (0, nx_pad)), mode='edge')
        lon = np.pad(lon, ((0, ny_pad), (0, nx_pad)), mode='edge')

        R = 6371000
        dlat = np.deg2rad(np.diff(lat[:, 0]).mean())
        dlon = np.deg2rad(np.diff(lon[0, :]).mean())
        area = (R ** 2) * dlat * dlon * np.cos(np.deg2rad(lat))

        if not has_time:
            var = var.expand_dims('time')

        data = var.values
        area_blocks = area.reshape(ny_pad // block_size + ny // block_size, block_size,
                                   nx_pad // block_size + nx // block_size, block_size)
        var_blocks = data.reshape(
            data.shape[0],
            ny_pad // block_size + ny // block_size, block_size,
            nx_pad // block_size + nx // block_size, block_size
        )

        weighted = (var_blocks * area_blocks).sum(axis=(2, 4))
        total_area = area_blocks.sum(axis=(1, 3))
        data_coarse = weighted / total_area

        lat_coarse = lat.reshape(ny_pad // block_size + ny // block_size, block_size,
                                 nx_pad // block_size + nx // block_size, block_size).mean(axis=(1, 3))
        lon_coarse = lon.reshape(ny_pad // block_size + ny // block_size, block_size,
                                 nx_pad // block_size + nx // block_size, block_size).mean(axis=(1, 3))

        coords = {'lat': (['y', 'x'], lat_coarse), 'lon': (['y', 'x'], lon_coarse)}
        if has_time:
            coords['time'] = var['time']
            dims = ('time', 'y', 'x')
        else:
            data_coarse = data_coarse.squeeze()
            dims = ('y', 'x')

        da_coarse = xr.DataArray(data_coarse, coords=coords, dims=dims, name=varname)
    else:
        raise ValueError("Unknown grid type")

    da_coarse.to_netcdf(outfile)
    return outfile



def interpolate_bicubic(coarse_file, target_file, output_file):
    print(f"Running CDO bicubic interpolation: {coarse_file} → {output_file}")
    cmd = [
        "cdo", f"remapbic,{target_file}",
        str(coarse_file), str(output_file)
    ]
    subprocess.run(cmd, check=True)



def split(ds, seed, train_ratio):
    np.random.seed(seed)
    indices = np.arange(ds.sizes['time'])
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(indices))
    return ds.isel(time=indices[:split_idx]), ds.isel(time=indices[split_idx:])



def scale(train, val, method):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    train_flat = train.values.reshape(-1, 1)
    val_flat = val.values.reshape(-1, 1)

    scaler.fit(train_flat)
    train_scaled = scaler.transform(train_flat).reshape(train.shape)
    val_scaled = scaler.transform(val_flat).reshape(val.shape)

    train_da = xr.DataArray(train_scaled, coords=train.coords, dims=train.dims)
    val_da = xr.DataArray(val_scaled, coords=val.coords, dims=val.dims)
    return train_da, val_da, scaler



def save_params(scaler, varname, dataset_type, out_dir):
    params = {}
    if hasattr(scaler, 'mean_'):
        params['mean'] = float(scaler.mean_[0])
    if hasattr(scaler, 'scale_'):
        params['std'] = float(scaler.scale_[0])
    if hasattr(scaler, 'data_min_'):
        params['min'] = float(scaler.data_min_[0])
    if hasattr(scaler, 'data_max_'):
        params['max'] = float(scaler.data_max_[0])
    
    out_path = out_dir / f"{varname}_{dataset_type}_scaling_params.json"
    with open(out_path, 'w') as f:
        json.dump(params, f, indent=2)

datasets = [
    ("precip_1771_2010.nc", "precip", "minmax"),
    ("temp_1771_2010.nc", "temp", "standard"),
    ("tmin_1771_2010.nc", "tmin", "standard"),
    ("tmax_1771_2010.nc", "tmax", "standard"),
]

for infile, varname, scale_type in datasets:
    infile_path = INPUT_DIR / infile
    coarse_path = OUTPUT_DIR / f"{varname}_coarse.nc"
    interp_path = OUTPUT_DIR / f"{varname}_interp.nc"


    # Coarsening high-res 
    conservative_coarsening(infile_path, varname, block_size=11, outfile=coarse_path)

    # Interpolating back using bicubic
    interpolate_bicubic(coarse_path, infile_path, interp_path)

    # Loading
    highres = xr.open_dataset(infile_path)[varname]
    upsampled = xr.open_dataset(interp_path)[varname]

    # Splitting 80-20
    x_train, x_val = split(upsampled, SEED, TRAIN_RATIO)
    y_train, y_val = split(highres, SEED, TRAIN_RATIO)

    # Scaling
    x_train_scaled, x_val_scaled, scaler_x = scale(x_train, x_val, scale_type)
    y_train_scaled, y_val_scaled, scaler_y = scale(y_train, y_val, scale_type)

    # Saving
    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")

    #Saving scaling params
    save_params(scaler_x, varname, "input", OUTPUT_DIR)
    save_params(scaler_y, varname, "target", OUTPUT_DIR)

print("datasets processed and saved.")
