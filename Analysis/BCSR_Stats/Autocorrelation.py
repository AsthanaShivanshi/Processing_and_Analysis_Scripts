from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from load_medians import apply_mask_exact, load_all_medians_masked, _subset_years, _sanitize

SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARS = ["tas", "pr"]

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


def _parse_lags(s: str) -> list[int]:
    out: list[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v < 1:
            raise ValueError(f"lag must be >= 1, got {v}")
        out.append(v)
    if not out:
        raise ValueError("no valid lags provided")
    return out


def _remove_seasonal_cycle(da: xr.DataArray) -> xr.DataArray:
    """
    Remove daily seasonal cycle:
    anomaly(t) = value(t) - climatology(dayofyear(t))
    """
    clim = da.groupby("time.dayofyear").mean("time", skipna=True)
    return da.groupby("time.dayofyear") - clim


def _detrend_linear_time(da: xr.DataArray) -> xr.DataArray:
    """Remove linear trend along time at each grid cell."""
    if da.sizes.get("time", 0) < 3:
        return da
    t = xr.DataArray(np.arange(da.sizes["time"]), dims="time", coords={"time": da.time})
    coeffs = da.polyfit(dim="time", deg=1, skipna=True)
    trend = xr.polyval(t, coeffs.polyfit_coefficients)
    return da - trend


def _fisher_z_spatial_mean_corr(corr_map: xr.DataArray) -> float:
    """
    Spatial mean correlation using Fisher-z:
      z = atanh(r), mean(z), then tanh(mean(z)).
    """
    eps = 1e-6
    r = corr_map.clip(min=-1 + eps, max=1 - eps)
    z = np.arctanh(r)

    dims = list(z.dims)
    if not dims:
        v = float(z.values)
        if not np.isfinite(v):
            return np.nan
        return float(np.tanh(v))

    z_mean = float(z.mean(dim=dims, skipna=True).values)
    if not np.isfinite(z_mean):
        return np.nan
    return float(np.tanh(z_mean))


def _seasonal_gridwise_autocorr_spatial_mean(
    da: xr.DataArray,
    lags: list[int],
) -> dict[tuple[str, int], float]:
    """
    Internal-variability persistence:
      1) remove seasonal cycle,
      2) detrend,
      3) compute gridcell-wise lag autocorrelation by season,
      4) spatially aggregate via Fisher-z mean.
    """
    out: dict[tuple[str, int], float] = {}

    da_anom = _remove_seasonal_cycle(da)
    da_resid = _detrend_linear_time(da_anom)

    for s in SEASONS:
        ds = da_resid.where(da_resid.time.dt.season == s, drop=True)

        if ds.sizes.get("time", 0) < 3:
            for lag in lags:
                out[(s, lag)] = np.nan
            continue

        for lag in lags:
            if ds.sizes["time"] <= lag:
                out[(s, lag)] = np.nan
                continue

            corr_map = xr.corr(ds, ds.shift(time=lag), dim="time")
            out[(s, lag)] = _fisher_z_spatial_mean_corr(corr_map)

    return out


def _col(var: str, season: str, lag: int) -> str:
    return f"{var}_{season}_lag{lag}"


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
    ap.add_argument("--lags", default="1,2,3")
    ap.add_argument("--out_csv", default="Analysis/BCSR_Stats/Tables/autocorrelation_tas_pr_2015_2023.csv")
    args = ap.parse_args()

    lags = _parse_lags(args.lags)

    loaded = load_all_medians_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=VARS,
    )

    obs_tas = xr.open_dataset(args.obs_tas_file)[args.obs_tas_var]
    obs_pr = xr.open_dataset(args.obs_pr_file)[args.obs_pr_var]

    obs_tas = apply_mask_exact(
        _subset_years(_sanitize(obs_tas, "tas"), args.eval_start, args.eval_end),
        loaded.hr_mask,
    )
    obs_pr = apply_mask_exact(
        _subset_years(_sanitize(obs_pr, "pr"), args.eval_start, args.eval_end),
        loaded.hr_mask,
    )

    cols = [_col(v, s, lag) for v in VARS for lag in lags for s in SEASONS]
    table = pd.DataFrame(index=TABLE_ROWS + ["MCH (spatial analysis)"], columns=cols, dtype=float)

    obs_vals_map = {
        "tas": _seasonal_gridwise_autocorr_spatial_mean(obs_tas, lags),
        "pr": _seasonal_gridwise_autocorr_spatial_mean(obs_pr, lags),
    }
    for v in VARS:
        for lag in lags:
            for s in SEASONS:
                table.loc["MCH (spatial analysis)", _col(v, s, lag)] = obs_vals_map[v][(s, lag)]

    for b in TABLE_ROWS:
        for v in VARS:
            da = loaded.data.get(v, {}).get(b)
            if da is None:
                continue
            vals = _seasonal_gridwise_autocorr_spatial_mean(da, lags)
            for lag in lags:
                for s in SEASONS:
                    table.loc[b, _col(v, s, lag)] = vals[(s, lag)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, float_format="%.3f")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()