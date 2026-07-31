import xarray as xr
import numpy as np


def daily_climatology(arr, time_coord=None):
    if isinstance(arr, xr.DataArray):
        if time_coord is None:
            time_coord = arr["time"]
        values = arr.values
    else:
        values = np.asarray(arr)

    if time_coord is None:
        raise ValueError("time_coord is required")

    if hasattr(time_coord, "dt"):
        dayofyear = np.asarray(time_coord.dt.dayofyear)
    else:
        dayofyear = np.asarray(time_coord)

    clim = np.full((366, *values.shape[1:]), np.nan)
    for doy in range(1, 367):
        mask = (dayofyear == doy)
        if np.any(mask):
            clim[doy - 1] = np.nanmean(values[mask], axis=0)
        else:
            clim[doy - 1] = np.nan
    return clim



def gridwise_perkins_skill_score(a, b, nbins=50):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"Shapes do not match: {a.shape} vs {b.shape}")

    pss = np.full(a.shape[1:], np.nan)

    for idx in np.ndindex(a.shape[1:]):
        a1 = a[(slice(None),) + idx]
        b1 = b[(slice(None),) + idx]

        mask = ~np.isnan(a1) & ~np.isnan(b1)
        if np.sum(mask) > 10:
            try:
                a_valid = a1[mask]
                b_valid = b1[mask]

                combined_data = np.concatenate([a_valid, b_valid])
                vmin = np.nanmin(combined_data)
                vmax = np.nanmax(combined_data)

                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-6

                bins = np.linspace(vmin, vmax, nbins + 1)
                hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                hist_b, _ = np.histogram(b_valid, bins=bins, density=True)

                hist_a_sum = np.sum(hist_a)
                hist_b_sum = np.sum(hist_b)

                if hist_a_sum > 0 and hist_b_sum > 0:
                    hist_a = hist_a / hist_a_sum
                    hist_b = hist_b / hist_b_sum
                    pss[idx] = np.sum(np.minimum(hist_a, hist_b))
            except Exception:
                pss[idx] = np.nan

    return pss


def _spatial_dims(da):
    return [d for d in da.dims if d != "time"]


def pss_gridwise_spatial_mean(pred, ref, nbins=50):
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0:
        return np.nan

    clim_a = daily_climatology(pred, pred["time"])
    clim_b = daily_climatology(ref, ref["time"])

    pss_map = gridwise_perkins_skill_score(clim_a, clim_b, nbins=nbins)

    spatial_dims = _spatial_dims(pred)
    if not spatial_dims:
        return float(np.nanmean(pss_map))

    pss_da = xr.DataArray(
        pss_map,
        dims=spatial_dims,
        coords={dim: pred.coords[dim] for dim in spatial_dims if dim in pred.coords},
    )

    return float(pss_da.mean(dim=spatial_dims, skipna=True).values)