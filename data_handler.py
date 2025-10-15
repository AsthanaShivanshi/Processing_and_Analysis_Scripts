import numpy as np
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import os
import cdsapi
import math
import tempfile
import shutil
import zipfile

# --- Configuration for CDS API ---
# In a real-world scenario, you would need to have your .cdsapirc file configured
# with your API key and URL. This client object is used to make the requests.
CDS_CLIENT = cdsapi.Client()
OUTPUT_ROOT_DIR = "era5_daily_data"
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# Constants for the 3x3 pixel area (approximated for simplicity)
TARGET_SIDE_KM = 30  # A 3x3 pixel area is approximately 30km x 30km
HALF_SIDE_KM = TARGET_SIDE_KM / 2
KM_PER_DEG_LAT = 111.0  # Approximate km per degree of latitude

# List of variables to download, based on the original request
ERA5_VARIABLES = [
    "2m_temperature",
    "Maximum_2m_temperature_since_previous_post_processing",
    "Minimum_2m_temperature_since_previous_post_processing",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",  
]


def calculate_bounding_box(lat, lon, half_side_km=HALF_SIDE_KM):
    """
    Calculates a dynamic bounding box for a given center point and side length
    in kilometers, approximating a square area.

    Args:
        lat (float): Center latitude.
        lon (float): Center longitude.
        half_side_km (float): Half the side length of the square in km.

    Returns:
        list: A bounding box in the format [N, W, S, E].
    """
    delta_lat_deg = half_side_km / KM_PER_DEG_LAT

    # Calculate longitude delta, adjusting for the curvature of the Earth
    cos_lat = np.cos(np.radians(lat))
    if math.isclose(cos_lat, 0.0, abs_tol=1e-9):
        delta_lon_deg = 1.0
    else:
        delta_lon_deg = half_side_km / (KM_PER_DEG_LAT * cos_lat)

    bbox = [
        round(lat + delta_lat_deg, 2),  # Northern Lat
        round(lon - delta_lon_deg, 2),  # Western Lon
        round(lat - delta_lon_deg, 2),  # Southern Lat
        round(lon + delta_lon_deg, 2),  # Eastern Lon
    ]

    # Ensure the bounding box is within valid geographical limits
    bbox[0] = min(90.0, max(-90.0, bbox[0]))
    bbox[2] = min(90.0, max(-90.0, bbox[2]))
    bbox[1] = min(180.0, max(-180.0, bbox[1]))
    bbox[3] = min(180.0, max(-180.0, bbox[3]))

    return bbox


def parse_row(row):
    """
    Parses a single row from the CSV table to extract relevant details.

    Args:
        row (list): A list of strings representing a single row.

    Returns:
        dict: A dictionary containing parsed ascent details.
    """
    try:
        # Extract data from the fixed-format row
        year = int(row[0])
        lat = float(row[5])
        lon = float(row[6])
        start_date_str = row[7]
        end_date_str = row[8]

        # Parse dates and calculate the duration of the ascent
        start_date = datetime.datetime.strptime(f"{start_date_str}/{year}", "%d/%m/%Y")
        end_date = datetime.datetime.strptime(f"{end_date_str}/{year}", "%d/%m/%Y")

        # Add 1 day to the duration to be inclusive of the end date
        ascent_duration = (end_date - start_date).days + 1

        # Define the start date for the 30-year historical analysis
        analysis_start_date = start_date - relativedelta(years=30)

        return {
            "year": year,
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "ascent_duration": ascent_duration,
            "analysis_start_date": analysis_start_date,
        }
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Error parsing input row. Please check the format. Details: {e}"
        )


# --- Helper function to unzip if file is zipped ---
def ensure_unzipped(nc_file_path):
    try:
        if zipfile.is_zipfile(nc_file_path):
            temp_dir = tempfile.mkdtemp(prefix="unzipped_nc_")
            with zipfile.ZipFile(nc_file_path, "r") as z:
                extracted_files = [f for f in z.namelist() if f.endswith(".nc")]
                if extracted_files:
                    z.extract(
                        extracted_files[0], temp_dir
                    )  # Extract only the first .nc file
                    return os.path.join(temp_dir, extracted_files[0])
                else:
                    print(f"  Warning: Zip file {nc_file_path} contained no .nc file.")
                    shutil.rmtree(temp_dir)  # clean up
                    return nc_file_path
        else:
            return nc_file_path
    except Exception as e:
        print(f"  Warning: Could not unzip {nc_file_path} due to error: {e}")
        return nc_file_path


def download_era5_data(lat, lon, start_date, end_date):
    """
    Downloads hourly ERA5-Land data for a given location/time range, month by month,
    returns an xarray.Dataset with standardized variable/dimension names:
      variables: t2m, tp, u10, v10, sp, wind_speed
      dims: time, lat, lon
    """
    print(
        f"Requesting hourly ERA5 data from {start_date.strftime('%Y-%m-%d')} "
        f"to {end_date.strftime('%Y-%m-%d')}..."
    )

    # --- bounding box (N, W, S, E) ---
    bbox = calculate_bounding_box(lat, lon)
    bbox_str = "_".join([str(x).replace(".", "p").replace("-", "m") for x in bbox])

    file_paths = []
    current_date = start_date

    while current_date.year < end_date.year or (
        current_date.year == end_date.year and current_date.month <= end_date.month
    ):
        year = current_date.year
        month = current_date.month

        output_filename = os.path.join(
            OUTPUT_ROOT_DIR, f"era5_hourly_data_{bbox_str}_{year}_{month:02d}.nc"
        )
        file_paths.append(output_filename)

        if not os.path.exists(output_filename):
            print(f"Downloading data for {current_date.strftime('%Y-%m')}")
            request_params = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ERA5_VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                # NOTE: CDS uses 'time' for hours, not 'valid_time'
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,  # [N, W, S, E]
            }

            try:
                CDS_CLIENT.retrieve(
                    "reanalysis-era5-land", request_params, output_filename
                )
                print(
                    f"Successfully downloaded data for {year}-{month:02d} to {output_filename}"
                )
            except Exception as e:
                print(
                    f"Error downloading data for {year}-{month:02d}: {e}. "
                    "Please ensure your CDS API credentials are set up."
                )
                raise
        else:
            print(
                f"File for {current_date.strftime('%Y-%m')} already exists, loading from disk."
            )

        current_date += relativedelta(months=1)

    # Unzip if necessary
    unzipped_file_paths = [ensure_unzipped(p) for p in file_paths]

    if not unzipped_file_paths:
        raise ValueError("No datasets were downloaded or loaded.")

    # Open and force-load before cleaning temp dirs
    ds = xr.open_mfdataset(
        unzipped_file_paths, combine="by_coords", engine="netcdf4"
    ).load()

    print("DEBUG: Data variables in dataset ->", list(ds.data_vars))
    print("DEBUG: Coordinates ->", list(ds.coords))

    # Clean up any temporary unzip folders (safe now that we've loaded to memory)
    for p in unzipped_file_paths:
        if "unzipped_nc_" in p:
            try:
                shutil.rmtree(os.path.dirname(p))
            except Exception:
                pass

    # --- Standardize variable names (only rename if present) ---
    # ERA5-Land NetCDF typically already uses short names: t2m, tp, u10, v10, sp
    var_map_long_to_short = {
        "2m_temperature": "t2m",
        "maximum_2m_temperature_since_previous_post_processing": "t2m_max",
        "minimum_2m_temperature_since_previous_post_processing": "t2m_min",
        "total_precipitation": "tp",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "surface_pressure": "sp",
    }
    rename_vars = {
        old: new for old, new in var_map_long_to_short.items() if old in ds.data_vars
    }
    if rename_vars:
        ds = ds.rename(rename_vars)

    # --- Standardize dimension/coord names to: time, lat, lon ---
    dim_map = {}
    if "latitude" in ds.dims:
        dim_map["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_map["longitude"] = "lon"
    # Some readers (GRIB) use "valid_time"; NetCDF should be "time".
    if "valid_time" in ds.dims:
        dim_map["valid_time"] = "time"
    if dim_map:
        ds = ds.rename(dim_map)
    # Also fix coordinate names if present as coords instead of dims
    coord_map = {}
    if "latitude" in ds.coords:
        coord_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        coord_map["longitude"] = "lon"
    if "valid_time" in ds.coords:
        coord_map["valid_time"] = "time"
    if coord_map:
        ds = ds.rename(coord_map)

    # --- Sanity: check required vars exist now ---
    required = ["t2m", "tp", "u10", "v10", "sp"]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Missing variables after standardization: {missing}. "
            f"Present: {list(ds.data_vars)}"
        )

    # --- Derive wind speed ---
    ds["wind_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    # --- Return only requested time range on a standard 'time' coord ---
    time_coord = (
        "time"
        if "time" in ds.coords
        else ("valid_time" if "valid_time" in ds.coords else None)
    )
    if time_coord is None:
        raise KeyError(
            f"No time coordinate found in dataset. Coords: {list(ds.coords)}"
        )

    ds = ds.sel({time_coord: slice(start_date, end_date)})

    # If the coord name was 'valid_time', rename to 'time' now for consistency downstream
    if time_coord == "valid_time":
        ds = ds.rename({"valid_time": "time"})

    return ds


# 2021,Argentina,Patagonia,Fitz Roy,3405,-49.2719° N, -73.0435° E,05/02,10/02,Sean Villanueva O'Driscoll,BEL,Moonwalk Traverse,winner,solo alpine style
