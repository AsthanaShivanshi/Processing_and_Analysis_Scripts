import numpy as np
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import os
import cdsapi


# --- Configuration for CDS API ---
CDS_CLIENT = cdsapi.Client()
REFERENCE_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/"
OUTPUT_ROOT_DIR = os.path.join(REFERENCE_DIR, "ERA5_daily_EUR")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
os.chdir(REFERENCE_DIR)

# European domain
EUROPE_BBOX = [70,-30,37,50]  # [N, W, S, E] - European continental domain

# List of variables to download
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure", 
    "relative_humidity",
]

def download_era5_daily_data_europe(start_date, end_date):
    """
    Downloads hourly ERA5-Land data for European domain and converts to daily averages.
    Returns an xarray.Dataset with daily data covering the European domain.
    
    Args:
        start_date (datetime): Start date for data download
        end_date (datetime): End date for data download
        
    Returns:
        xarray.Dataset: Daily averaged ERA5 data for European domain
    """
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
            
            # First download hourly data
            temp_hourly_file = os.path.join(
                OUTPUT_ROOT_DIR, f"temp_hourly_europe_{year}_{month:02d}.nc"
            )
            
            request_params = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ERA5_VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": EUROPE_BBOX,  # [N, W, S, E] - Europe
            }

            try:
                CDS_CLIENT.retrieve(
                    "reanalysis-era5-land", request_params, temp_hourly_file
                )
                print(f"Downloaded hourly data for {year}-{month:02d}")
                
                # Load and convert to daily averages
                hourly_ds = xr.open_dataset(temp_hourly_file)
                daily_ds = convert_hourly_to_daily(hourly_ds)
                
                # Save daily data
                daily_ds.to_netcdf(output_filename)
                print(f"Converted to daily averages and saved to {output_filename}")
                
                # Clean up temporary hourly file
                if os.path.exists(temp_hourly_file):
                    os.remove(temp_hourly_file)
                    
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
    """
    Converts hourly ERA5 data to daily averages/sums as appropriate.
    
    Args:
        hourly_ds (xarray.Dataset): Hourly ERA5 dataset
        
    Returns:
        xarray.Dataset: Daily averaged/summed dataset with tmin, tmax included
    """
    # Standardize the dataset first
    ds = standardize_dataset(hourly_ds)
    
    # Define aggregation methods for each variable
    aggregation_methods = {
        't2m': ['mean', 'min', 'max'],  # Temperature: daily mean, min, max
        'tp': 'sum',                    # Precipitation: daily sum
        'u10': 'mean',                  # Wind components: daily mean
        'v10': 'mean',
        'sp': 'mean',                   # Surface pressure: daily mean
        'relative_humidity': 'mean',    # Relative humidity: daily mean
    }
    
    daily_vars = {}
    
    for var_name in ds.data_vars:
        if var_name in aggregation_methods:
            methods = aggregation_methods[var_name]
            
            if isinstance(methods, list):
                # Handle multiple aggregations for temperature
                for method in methods:
                    if method == 'mean':
                        daily_vars[var_name] = ds[var_name].resample(time='1D').mean()
                    elif method == 'min':
                        daily_vars['tmin'] = ds[var_name].resample(time='1D').min()
                    elif method == 'max':
                        daily_vars['tmax'] = ds[var_name].resample(time='1D').max()
            else:
                # Handle single aggregation
                method = methods
                if method == 'mean':
                    daily_vars[var_name] = ds[var_name].resample(time='1D').mean()
                elif method == 'sum':
                    daily_vars[var_name] = ds[var_name].resample(time='1D').sum()
        else:
            # Default to mean for any other variables
            daily_vars[var_name] = ds[var_name].resample(time='1D').mean()
    
    # Create new dataset with daily data
    daily_ds = xr.Dataset(daily_vars, coords=ds.coords)
    
    # Recalculate wind speed from daily averaged components
    if 'u10' in daily_ds and 'v10' in daily_ds:
        daily_ds['wind_speed'] = np.sqrt(daily_ds['u10']**2 + daily_ds['v10']**2)
    
    return daily_ds


def standardize_dataset(ds):
    """
    Standardizes variable and dimension names in the dataset.
    
    Args:
        ds (xarray.Dataset): Input dataset
        
    Returns:
        xarray.Dataset: Standardized dataset
    """
    # Standardize variable names
    var_map_long_to_short = {
        "2m_temperature": "t2m",
        "total_precipitation": "tp",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "surface_pressure": "sp",
        "relative_humidity": "relative_humidity",
        
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