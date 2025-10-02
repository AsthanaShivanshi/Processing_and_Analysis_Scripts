import xarray as xr
import numpy as np
from pyextremes import EVA, get_return_periods
from closest_grid_cell import select_nearest_grid_cell

def get_extreme_return_levels_bm(
    nc_file: str,
    variable_name: str,
    lat: float,
    lon: float,
    block_size: str = "365D",  # block size for maxima, e.g. "365D" for annual
    return_periods=[20,50,100],
    time_slice=('1981-01-01', '2010-12-31'),
    return_all_periods=False
):
    ds = xr.open_dataset(nc_file).sel(time=slice(*time_slice))
    grid_data = select_nearest_grid_cell(ds, lat, lon, variable_name)
    series_pd = grid_data['data'].to_pandas()
    eva_model = EVA(series_pd)
    eva_model.get_extremes(method="BM", block_size=block_size)
    eva_model.fit_model()
    if return_all_periods:
        all_periods_df = get_return_periods(
            ts=series_pd,
            extremes=eva_model.extremes,
            extremes_method='BM',
            extremes_type="high",
            block_size=block_size,
            plotting_position="median"
        )
        return all_periods_df
    else:
        summary = eva_model.get_summary(
            return_period=return_periods,
            alpha=0.95,
            n_samples=1000,
        )
        return summary