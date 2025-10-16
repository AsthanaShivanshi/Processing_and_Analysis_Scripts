import numpy as np
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import os
import cdsapi
import calendar
import netCDF4
import zipfile
import requests
from requests.auth import HTTPBasicAuth

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

def _read_cdsapirc(path=os.path.expanduser("~/.cdsapirc")):
    """Read ~/.cdsapirc and return dict with 'url', 'key' (uid:key)."""
    creds = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    creds[k.strip()] = v.strip()
    except Exception:
        return None
    return creds


def get_remote_dataset_info(dataset_name):
    """
    Try to fetch dataset metadata from the CDS REST API.
    Returns the parsed JSON (if successful) or None on failure.
    """
    creds = _read_cdsapirc()
    if not creds or "url" not in creds or "key" not in creds:
        print("Could not read ~/.cdsapirc; skipping remote catalogue lookup.")
        return None

    # build resource URL from the 'url' field in .cdsapirc
    base_url = creds["url"]
    # ensure base like "https://cds.climate.copernicus.eu/api/v2"
    if base_url.endswith("/api/v2"):
        base = base_url[:-len("/api/v2")]
    else:
        base = base_url.rstrip("/")

    resource_url = f"{base}/api/v2/resources/{dataset_name}"

    try:
        keyval = creds["key"]
        if ":" not in keyval:
            print("Malformed CDS API key in ~/.cdsapirc. Should be 'uid:apikey'.")
            return None
        uid, apikey = keyval.split(":", 1)
        resp = requests.get(resource_url, auth=HTTPBasicAuth(uid, apikey), timeout=30)
        if resp.status_code != 200:
            print(f"Remote catalogue lookup for {dataset_name} failed: HTTP {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Error contacting CDS REST API for {dataset_name}: {e}")
        return None
    

def display_catalogue():
    """
    Print a local + remote catalogue summary before starting downloads.
    Shows configured variable lists and, when possible, variables available on the CDS server.
    """
    print("=== Local requested variable catalogue ===")
    print("Single-level variables:", ERA5_SINGLE_LEVEL_VARS)
    print("Pressure-level variables:", ERA5_PRESSURE_LEVEL_VARS)
    print("Pressure levels:", PRESSURE_LEVELS)
    print("Request area (latN, lonW, latS, lonE):", EUROPE_BBOX)
    print("")

    # Try remote lookup for both datasets
    for ds_name in ("reanalysis-era5-single-levels", "reanalysis-era5-pressure-levels"):
        info = get_remote_dataset_info(ds_name)
        if info is None:
            print(f"Could not retrieve remote info for {ds_name}.")
            continue
        print(f"=== Remote catalogue for {ds_name} ===")
        # Common keys: 'title', 'id', maybe 'variables' or 'fields'
        if "title" in info:
            print("Title:", info.get("title"))
        if "id" in info:
            print("ID:", info.get("id"))
        # try common places for variable listings
        variables = None
        if "variables" in info:
            variables = info["variables"]
        elif "fields" in info:
            variables = info["fields"]
        if variables:
            # variables may be dict mapping name->desc
            if isinstance(variables, dict):
                print("Available variables (sample):", list(variables.keys())[:50])
            elif isinstance(variables, list):
                print("Available variables (sample):", variables[:50])
        else:
            print("No variable listing found in remote metadata.")
        print("")



def file_has_all_vars(path):
    """Open an existing file, standardize names and check presence of required variables."""
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return False
    try:
        ds = standardize_dataset(ds)
    except Exception:
        return False

    # Required base vars
    required = {"t2m", "tp", "sp"}
    present = set(ds.data_vars)

    # For pressure-levels, check for level-specific names (e.g., u850) or presence of u/v/q/z with level dim
    for base in ["u", "v", "q", "z"]:
        # check u850 etc
        has_levels = any(f"{base}{lev}" in present for lev in PRESSURE_LEVELS)
        # or original var with level dim
        has_level_dim = (base in ds.data_vars) and ("level" in ds[base].dims)
        if not (has_levels or has_level_dim):
            return False

    if not required.issubset(present):
        return False

    return True


def download_era5_daily_data_europe(start_date, end_date):
    print(
        f"Requesting hourly ERA5 data for European domain from {start_date.strftime('%Y-%m-%d')} "
        f"to {end_date.strftime('%Y-%m-%d')}..."
    )

    display_catalogue() #Dispalying what is available. 

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

        # If file exists, check it contains all required variables; otherwise re-download
        need_download = True
        if os.path.exists(output_filename):
            try:
                if file_has_all_vars(output_filename):
                    print(f"Daily file for {current_date.strftime('%Y-%m')} already exists and is complete")
                    need_download = False
                else:
                    print(f"Daily file for {current_date.strftime('%Y-%m')} exists but is incomplete; re-downloading")
            except Exception as e:
                print(f"Could not validate existing file {output_filename}: {e}. Will re-download.")
                need_download = True

        if not need_download:
            file_paths.append(output_filename)
            current_date += relativedelta(months=1)
            continue

        print(f"Downloading hourly data for {current_date.strftime('%Y-%m')}")

        num_days = calendar.monthrange(year, month)[1]

        temp_single_file = os.path.join(
            OUTPUT_ROOT_DIR, f"temp_hourly_single_europe_{year}_{month:02d}.nc"
        )
        temp_pressure_file = os.path.join(
            OUTPUT_ROOT_DIR, f"temp_hourly_pressure_europe_{year}_{month:02d}.nc"
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
            # Download single-level data
            CDS_CLIENT.retrieve(
                "reanalysis-era5-single-levels", single_params, temp_single_file
            )
            # Download pressure-level data
            CDS_CLIENT.retrieve(
                "reanalysis-era5-pressure-levels", pressure_params, temp_pressure_file
            )
            print(f"Downloaded hourly data for {year}-{month:02d}")

            # --- Handle ZIP files if present ---
            def extract_nc_from_zip(zip_path):
                if zipfile.is_zipfile(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                        if not nc_files:
                            raise RuntimeError(f"No NetCDF file found in {zip_path}")
                        # extract to same folder as zip_path
                        extract_dir = os.path.dirname(zip_path)
                        zip_ref.extract(nc_files[0], extract_dir)
                        return os.path.join(extract_dir, nc_files[0])
                return zip_path

            temp_single_file_extracted = extract_nc_from_zip(temp_single_file)
            temp_pressure_file_extracted = extract_nc_from_zip(temp_pressure_file)

            # Load and merge
            ds_single = xr.open_dataset(temp_single_file_extracted, engine="netcdf4")
            ds_pressure = xr.open_dataset(temp_pressure_file_extracted, engine="netcdf4")
            hourly_ds = xr.merge([ds_single, ds_pressure])

            # Convert to daily (ensuring the output contains only daily timesteps)
            daily_ds = convert_hourly_to_daily(hourly_ds)

            # Write daily monthly file
            daily_ds.to_netcdf(output_filename)
            print(f"Converted to daily averages and saved to {output_filename}")
            file_paths.append(output_filename)

            # Clean up temporary files
            for f in [temp_single_file, temp_pressure_file, temp_single_file_extracted, temp_pressure_file_extracted]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Skipping {year}-{month:02d} due to error: {e}")
            continue

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

    return ds, file_paths


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
                        # mean will be used as the main var_name
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

    # Determine proper coordinates from the daily variables (avoid using original hourly time coordinates)
    time_coord = None
    lat_coord = None
    lon_coord = None
    for v in daily_vars.values():
        if time_coord is None and 'time' in v.dims:
            time_coord = v.coords['time']
        if lat_coord is None and 'lat' in v.coords:
            lat_coord = v.coords['lat']
        if lon_coord is None and 'lon' in v.coords:
            lon_coord = v.coords['lon']
        if time_coord is not None and lat_coord is not None and lon_coord is not None:
            break

    coords = {}
    if time_coord is not None:
        coords['time'] = time_coord
    if lat_coord is not None:
        coords['lat'] = lat_coord
    if lon_coord is not None:
        coords['lon'] = lon_coord

    daily_ds = xr.Dataset(daily_vars, coords=coords)

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
    for year in range(1971, 2021): #Entire period for training, testing and validation. 
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        print(f"Processing year {year}...")
        ds, monthly_files = download_era5_daily_data_europe(start_date, end_date)
        yearly_file = os.path.join(OUTPUT_ROOT_DIR, f"era5_daily_europe_{year}.nc")
        ds.to_netcdf(yearly_file)
        print(f"Saved {yearly_file}")

        # Remove monthly files after successful aggregation
        for mf in monthly_files:
            try:
                if os.path.exists(mf):
                    os.remove(mf)
            except Exception:
                pass
        print(f"Removed monthly files for {year}")