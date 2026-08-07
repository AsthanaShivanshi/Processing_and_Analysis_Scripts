from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[4]

OBS_CFG = {
    "tas": {
        "file": REPO_ROOT / "Downscaling"/"Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc",
        "var": "TabsD",
    },
    "pr": {
        "file": REPO_ROOT / "Downscaling"/"Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc",
        "var": "RhiresD",
    },
}


def _load_mask(mask_path: Path | None, mask_var: str | None) -> xr.DataArray | None:
    if mask_path is None:
        return None
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    with xr.open_dataset(mask_path) as ds:
        if mask_var and mask_var in ds:
            return ds[mask_var].load()
        return ds[list(ds.data_vars)[0]].load()


def _apply_mask(da: xr.DataArray, mask: xr.DataArray | None) -> xr.DataArray:
    if mask is None:
        return da

    m = mask.isel(time=0, drop=True) if "time" in mask.dims else mask

    anchor = da
    for d in ("time", "member", "sample", "samples"):
        if d in anchor.dims:
            anchor = anchor.isel({d: 0}, drop=True)

    try:
        m2 = m.broadcast_like(anchor)
    except Exception as e:
        raise ValueError(f"Mask is not compatible with data grid/dimensions: {e}") from e

    return da.where(m2 > 0)


def _subset_years(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("Data has no 'time' dimension.")

    t = da["time"]



    try:
        years = t.dt.year
        out = da.where((years >= y0) & (years <= y1), drop=True)
    except Exception:

        vals = np.asarray(t.values)
        if np.issubdtype(vals.dtype, np.integer) or np.issubdtype(vals.dtype, np.floating):
            v = vals.astype(np.int64)

            if np.nanmax(v) > 100000:
                yr = v // 10000
            else:
                yr = v
            out = da.where((yr >= y0) & (yr <= y1), drop=True)
        else:

            out = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

    if out.sizes.get("time", 0) == 0:
        tmin = str(t.values[0]) if t.size else "NA"
        tmax = str(t.values[-1]) if t.size else "NA"
        raise ValueError(f"No data in requested period {y0}-{y1}. Available time appears to be: {tmin} ... {tmax}")

    return out


def _historical_mean(file_path: Path, var_name: str, y0: int, y1: int, mask: xr.DataArray | None) -> tuple[float, int]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with xr.open_dataset(file_path) as ds:
        da = ds[var_name] if var_name in ds.data_vars else ds[list(ds.data_vars)[0]]
        da = da.load()

    da = _apply_mask(da, mask)
    da = _subset_years(da, y0, y1)



    mean_val = float(da.mean(skipna=True).values)

    if not np.isfinite(mean_val):
        raise ValueError(f"Mean is NaN for {file_path.name}:{var_name}. Check mask and missing data.")

    n_time = int(da.sizes.get("time", 0))
    return mean_val, n_time


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute historical mean (1981-2010) only.")
    ap.add_argument("--tas_file", type=Path, default=OBS_CFG["tas"]["file"])
    ap.add_argument("--tas_var", type=str, default=OBS_CFG["tas"]["var"])
    ap.add_argument("--pr_file", type=Path, default=OBS_CFG["pr"]["file"])
    ap.add_argument("--pr_var", type=str, default=OBS_CFG["pr"]["var"])
    ap.add_argument("--hist_start", type=int, default=1981)
    ap.add_argument("--hist_end", type=int, default=2010)
    ap.add_argument("--mask_file", type=Path, default=None)
    ap.add_argument("--mask_var", type=str, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    mask = _load_mask(args.mask_file, args.mask_var)

    tas_mean, tas_n = _historical_mean(args.tas_file, args.tas_var, args.hist_start, args.hist_end, mask)
    pr_mean, pr_n = _historical_mean(args.pr_file, args.pr_var, args.hist_start, args.hist_end, mask)

    df = pd.DataFrame(
        [
            {"var": "tas", "hist_start": args.hist_start, "hist_end": args.hist_end, "n_time": tas_n, "hist_mean": tas_mean},
            {"var": "pr", "hist_start": args.hist_start, "hist_end": args.hist_end, "n_time": pr_n, "hist_mean": pr_mean},
        ]
    )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False, float_format="%.6f")
        print(f"[ok] wrote {args.out_csv}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()