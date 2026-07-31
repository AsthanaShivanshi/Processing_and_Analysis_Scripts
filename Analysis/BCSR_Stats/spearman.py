from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ConstantInputWarning, spearmanr

import config
from load_medians import apply_mask_exact, load_all_medians_masked, _subset_years, _sanitize

SEASONS = ["DJF", "MAM", "JJA", "SON"]

TABLE_ROWS = [
    "EQM + Bilinear",
    "CDF-t + Bilinear",
    "dOTC + Bilinear",
    "EQM + Bilinear + U-Net",
    "CDF-t + Bilinear + U-Net",
    "dOTC + Bilinear + U-Net",
    "EQM + Bilinear + U-Net + DDIM",
    "CDF-t + Bilinear + U-Net + DDIM",
    "dOTC + Bilinear + U-Net + DDIM",
    "CH2025 methodological baseline",
]


def _spatial_mean_all_dims(da: xr.DataArray) -> float:
    dims = list(da.dims)
    if not dims:
        v = float(da.values)
        return v if np.isfinite(v) else np.nan
    return float(da.mean(dim=dims, skipna=True).values)


def _daily_climatology(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da

    clim = da.groupby("time.dayofyear").mean("time", skipna=True)
    return clim.reindex(dayofyear=np.arange(1, 367))


def _season_for_doy(doy: int) -> str:
    if doy is None:
        return np.nan

    date = pd.Timestamp(year=2000, month=1, day=1) + pd.Timedelta(days=doy - 1)
    month = date.month
    if month in [12, 1, 2]:
        return "DJF"
    if month in [3, 4, 5]:
        return "MAM"
    if month in [6, 7, 8]:
        return "JJA"
    return "SON"


def _spearman_map_over_dayofyear(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    def _rho_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan

        x = x[m]
        y = y[m]

        if x.size < 2 or y.size < 2:
            return np.nan

        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            return np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            r = spearmanr(x, y).statistic

        return float(r) if np.isfinite(r) else np.nan

    return xr.apply_ufunc(
        _rho_1d,
        a,
        b,
        input_core_dims=[["dayofyear"], ["dayofyear"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def _seasonal_daily_climatology_spearman_gridwise_spatial_mean(
    tas_da: xr.DataArray, pr_da: xr.DataArray
) -> dict[str, float]:
    tas_da, pr_da = xr.align(tas_da, pr_da, join="inner")

    clim_tas = _daily_climatology(tas_da)
    clim_pr = _daily_climatology(pr_da)

    out: dict[str, float] = {}

    for s in SEASONS:
        doy_vals = [int(d) for d in clim_tas["dayofyear"].values if _season_for_doy(int(d)) == s]
        if not doy_vals:
            out[s] = np.nan
            continue

        tas_s = clim_tas.sel(dayofyear=doy_vals)
        pr_s = clim_pr.sel(dayofyear=doy_vals)

        rho_map = _spearman_map_over_dayofyear(tas_s, pr_s)
        out[s] = _spatial_mean_all_dims(rho_map)

    return out


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/Ensmedians"))
    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--out_csv", default="Analysis/BCSR_Stats/Tables/intervariable_spearman_2015_2023.csv")
    args = ap.parse_args()

    loaded = load_all_medians_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
    )

    obs_tas = xr.open_dataset(args.obs_tas_file)[args.obs_tas_var]
    obs_pr = xr.open_dataset(args.obs_pr_file)[args.obs_pr_var]
    obs_tas = apply_mask_exact(_subset_years(_sanitize(obs_tas, "tas"), args.eval_start, args.eval_end), loaded.hr_mask)
    obs_pr = apply_mask_exact(_subset_years(_sanitize(obs_pr, "pr"), args.eval_start, args.eval_end), loaded.hr_mask)

    obs_corr = _seasonal_daily_climatology_spearman_gridwise_spatial_mean(obs_tas, obs_pr)

    corr = pd.DataFrame(index=TABLE_ROWS, columns=SEASONS, dtype=float)
    for b in TABLE_ROWS:
        tas = loaded.data.get("tas", {}).get(b)
        pr = loaded.data.get("pr", {}).get(b)
        if tas is None or pr is None:
            continue
        vals = _seasonal_daily_climatology_spearman_gridwise_spatial_mean(tas, pr)
        for s in SEASONS:
            corr.loc[b, s] = vals[s]

    corr.loc["MCH (spatial analysis)", SEASONS] = [obs_corr[s] for s in SEASONS]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, float_format="%.3f")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()