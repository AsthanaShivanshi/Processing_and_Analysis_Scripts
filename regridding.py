import xarray as xr
import numpy as np
import os
import sys
import os
from pathlib import Path

BASE_DIR = Path(os.environ["BASE_DIR"])
ENVIRONMENT = BASE_DIR / "sasthana" / "MyPythonEnvNew"
INPUT_DIR = BASE_DIR / "raw_data" / "AsthanaShivanshi_MeteoSwiss_Datasets"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Data_processed"


def conservative_coarsening(
    infile,
    varname,
    block_size,
    outfile=None,
    latname='lat',
    lonname='lon'
):
    ds = xr.open_dataset(infile)
    da = ds[varname]

    dims = da.dims
    has_time = 'time' in dims

#Valid for Switzerlnad, as each grid cell is approx the same area (1000x1000 m squared due to regular E N projection)
    if 'E' in dims and 'N' in dims:
        print(f"{varname}: Projected grid (E/N), using arithmetic mean over {block_size}x{block_size} blocks.")
        da_coarse = da.coarsen(N=block_size, E=block_size, boundary='pad').mean()

    elif latname in ds and lonname in ds:
        print(f"{varname}: Curvilinear grid detected, using area-weighted averaging with padding.")

        lat = ds[latname].values
        lon = ds[lonname].values
        var = da

        if lat.shape != var.shape[-2:] or lon.shape != var.shape[-2:]:
            raise ValueError("lat/lon shape must match last two dimensions of variable")

        ny, nx = lat.shape

        # Calculate padding: NEEDED
        ny_pad = (block_size - ny % block_size) % block_size
        nx_pad = (block_size - nx % block_size) % block_size

        print(f"Padded from ({ny}, {nx}) to ({ny + ny_pad}, {nx + nx_pad})")

        # Padding variable using edge values
        var = var.pad(
            {var.dims[-2]: (0, ny_pad), var.dims[-1]: (0, nx_pad)},
            mode='edge'
        )
        lat = np.pad(lat, ((0, ny_pad), (0, nx_pad)), mode='edge')
        lon = np.pad(lon, ((0, ny_pad), (0, nx_pad)), mode='edge')

        ny_padded, nx_padded = lat.shape

        # Calculating area of each grid cell
        R = 6371000  # Earth radius (approx) in meters  https://en.wikipedia.org/wiki/Earth_radius
        dlat = np.deg2rad(np.diff(lat[:, 0]).mean())
        dlon = np.deg2rad(np.diff(lon[0, :]).mean())
        area = (R ** 2) * dlat * dlon * np.cos(np.deg2rad(lat))

        if not has_time:
            var = var.expand_dims('time')

        data = var.values

        # Reshaping for block averaging
        area_blocks = area.reshape(ny_padded // block_size, block_size,
                                   nx_padded // block_size, block_size)
        var_blocks = data.reshape(
            data.shape[0],
            ny_padded // block_size, block_size,
            nx_padded // block_size, block_size
        )

        weighted = (var_blocks * area_blocks).sum(axis=(2, 4))
        total_area = area_blocks.sum(axis=(1, 3))
        data_coarse = weighted / total_area

        # new lat/lon (COARSE)
        lat_coarse = lat.reshape(ny_padded // block_size, block_size,
                                 nx_padded // block_size, block_size).mean(axis=(1, 3))
        lon_coarse = lon.reshape(ny_padded // block_size, block_size,
                                 nx_padded // block_size, block_size).mean(axis=(1, 3))

        coords = {
            'lat': (['y', 'x'], lat_coarse),
            'lon': (['y', 'x'], lon_coarse)
        }

        if has_time:
            coords['time'] = var['time']
            dims = ('time', 'y', 'x')
        else:
            data_coarse = data_coarse.squeeze()
            dims = ('y', 'x')

        da_coarse = xr.DataArray(data_coarse, coords=coords, dims=dims, name=varname)

    else:
        raise ValueError("Could not detect grid type")

    if outfile:
        da_coarse.to_netcdf(outfile)

    return da_coarse


datasets = [
    ("RhiresD_1971_2023.nc", "RhiresD", "RhiresD_011deg_coarsened.nc"),
    ("TabsD_1971_2023.nc", "TabsD", "TabsD_011deg_coarsened.nc"),
    ("TminD_1971_2023.nc", "TminD", "TminD_011deg_coarsened.nc"),
    ("TmaxD_1971_2023.nc", "TmaxD", "TmaxD_011deg_coarsened.nc")
]

for infile, varname, outfile in datasets:
    infile_path=os.path.join(INPUT_DIR,infile)
    outfile_path=os.path.join(OUTPUT_DIR,outfile)
    conservative_coarsening(infile, varname, block_size=11, outfile=outfile_path)
