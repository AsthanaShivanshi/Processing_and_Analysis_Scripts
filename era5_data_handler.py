import numpy as np
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import os
import cdsapi
import calendar
import netCDF4

# --- Configuration for CDS API ---
CDS_CLIENT = cdsapi.Client()
REFERENCE_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/"
OUTPUT_ROOT_DIR = os.path.join(REFERENCE_DIR, "ERA5_daily_EUR")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
os.chdir(REFERENCE_DIR)

# European domain
EUROPE_BBOX = [70,-30,37,50]  # [N, W, S, E] - European continental domain

# Single-level variables
ERA5_SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "total_precipitation",
    "surface_pressure",
]

# Pressure-level variables
ERA5_PRESSURE_LEVEL_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
PRESSURE_LEVELS = ["850", "700", "500"]

def download_era5_daily_data_europe(start_date, end_date):
    print(
        f"Requesting hourly ERA5 data for European domain from {start_date.strftime('%Y-%m-%d')} "
        f"to {end_date.strftime('%Y-%m-%d')}..."
    )

    file_paths = []
    current_date = start_date

    while current_date.year < end_date.year or (
        current_date.year == end_date.year and current_date.month <= end_date.month
    ):
        year = current_date.year
        month = current_date.month

        output_filename = os.path.join(
            OUTPUT_ROOT_DIR, f"era5_daily_europe_{year}_{month:02d}.nc"
        )
        file_paths.append(output_filename)

        if not os.path.exists(output_filename):
            print(f"Downloading hourly data for {current_date.strftime('%Y-%m')}")

            num_days = calendar.monthrange(year, month)[1]

            # Download single-level data
            temp_single_file = os.path.join(
                OUTPUT_ROOT_DIR, f"temp_hourly_single_europe_{year}_{month:02d}.nc"
            )
            single_params = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ERA5_SINGLE_LEVEL_VARS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, num_days + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": EUROPE_BBOX,
            }

            # Download pressure-level data
            temp_pressure_file = os.path.join(
                OUTPUT_ROOT_DIR, f"temp_hourly_pressure_europe_{year}_{month:02d}.nc"
            )
            pressure_params = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ERA5_PRESSURE_LEVEL_VARS,
                "pressure_level": PRESSURE_LEVELS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, num_days + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": EUROPE_BBOX,
            }

            try:
                CDS_CLIENT.retrieve(
                    "reanalysis-era5-single-levels", single_params, temp_single_file
                )
                CDS_CLIENT.retrieve(
                    "reanalysis-era5-pressure-levels", pressure_params, temp_pressure_file
                )
                print(f"Downloaded hourly data for {year}-{month:02d}")

                # Load and merge
                ds_single = xr.open_dataset(temp_single_file)
                ds_pressure = xr.open_dataset(temp_pressure_file)
                hourly_ds = xr.merge([ds_single, ds_pressure])

                daily_ds = convert_hourly_to_daily(hourly_ds)
                daily_ds.to_netcdf(output_filename)
                print(f"Converted to daily averages and saved to {output_filename}")

                # Clean up temporary files
                for f in [temp_single_file, temp_pressure_file]:
                    if os.path.exists(f):
                        os.remove(f)
            except Exception as e:
                print(f"Error downloading data for {year}-{month:02d}: {e}")
                raise
        else:
            print(f"Daily file for {current_date.strftime('%Y-%m')} already exists")

        current_date += relativedelta(months=1)

    # Load all daily files    
    if not file_paths:
        raise ValueError("No datasets were downloaded or loaded.")

    # Combine all monthly files
    ds = xr.open_mfdataset(
        file_paths, combine="by_coords", engine="netcdf4"
    ).load()

    # Standardize variable and dimension names
    ds = standardize_dataset(ds)

    # Filter to exact date range
    ds = ds.sel(time=slice(start_date, end_date))

    # Validate that we have data for the requested period
    if ds.time.size == 0:
        raise ValueError(f"No data found for the period {start_date} to {end_date}")

    return ds

def convert_hourly_to_daily(hourly_ds):
    ds = standardize_dataset(hourly_ds)

    aggregation_methods = {
        't2m': ['mean', 'min', 'max'],  # 2m temperature
        'tp': 'sum',                    # Precipitation
        'sp': 'mean',                   # Surface pressure
        'u': 'mean',                    # Zonal wind at pressure levels
        'v': 'mean',                    # Meridional wind at pressure levels
        'q': 'mean',                    # Specific humidity at pressure levels
        'z': 'mean',                    # Geopotential at pressure levels
    }

    daily_vars = {}

    for var_name in ds.data_vars:
        if var_name in aggregation_methods:
            methods = aggregation_methods[var_name]
            if isinstance(methods, list):
                for method in methods:
                    if method == 'mean':
                        daily_vars[var_name] = ds[var_name].resample(time='1D').mean()
                    elif method == 'min':
                        daily_vars['tmin'] = ds[var_name].resample(time='1D').min()
                    elif method == 'max':
                        daily_vars['tmax'] = ds[var_name].resample(time='1D').max()
            else:
                method = methods
                if method == 'mean':
                    daily_vars[var_name] = ds[var_name].resample(time='1D').mean()
                elif method == 'sum':
                    daily_vars[var_name] = ds[var_name].resample(time='1D').sum()
        else:
            daily_vars[var_name] = ds[var_name].resample(time='1D').mean()

    # Optionally, rename pressure-level variables for clarity (e.g., u850, v850, q850, z500)
    if 'u' in daily_vars and 'level' in ds['u'].dims:
        for lev in ds['u'].level.values:
            daily_vars[f'u{lev}'] = daily_vars['u'].sel(level=lev)
        del daily_vars['u']
    if 'v' in daily_vars and 'level' in ds['v'].dims:
        for lev in ds['v'].level.values:
            daily_vars[f'v{lev}'] = daily_vars['v'].sel(level=lev)
        del daily_vars['v']
    if 'q' in daily_vars and 'level' in ds['q'].dims:
        for lev in ds['q'].level.values:
            daily_vars[f'q{lev}'] = daily_vars['q'].sel(level=lev)
        del daily_vars['q']
    if 'z' in daily_vars and 'level' in ds['z'].dims:
        for lev in ds['z'].level.values:
            daily_vars[f'z{lev}'] = daily_vars['z'].sel(level=lev)
        del daily_vars['z']

    daily_ds = xr.Dataset(daily_vars, coords={k: v for k, v in ds.coords.items() if k in ['time', 'lat', 'lon']})

    return daily_ds

def standardize_dataset(ds):
    var_map_long_to_short = {
        "2m_temperature": "t2m",
        "total_precipitation": "tp",
        "surface_pressure": "sp",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "specific_humidity": "q",
        "geopotential": "z",
    }
    rename_vars = {
        old: new for old, new in var_map_long_to_short.items() if old in ds.data_vars
    }
    if rename_vars:
        ds = ds.rename(rename_vars)

    # Standardize dimension names
    dim_map = {}
    if "latitude" in ds.dims:
        dim_map["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_map["longitude"] = "lon"
    if "valid_time" in ds.dims:
        dim_map["valid_time"] = "time"
    if dim_map:
        ds = ds.rename(dim_map)

    # Also fix coordinate names
    coord_map = {}
    if "latitude" in ds.coords:
        coord_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        coord_map["longitude"] = "lon"
    if "valid_time" in ds.coords:
        coord_map["valid_time"] = "time"
    if coord_map:
        ds = ds.rename(coord_map)

    return ds

if __name__ == "__main__":
    for year in range(1981, 2011): #Validation period
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        print(f"Processing year {year}...")
        ds = download_era5_daily_data_europe(start_date, end_date)
        yearly_file = os.path.join(OUTPUT_ROOT_DIR, f"era5_daily_europe_{year}.nc")
        ds.to_netcdf(yearly_file)
        print(f"Saved {yearly_file}")