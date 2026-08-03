from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ConstantInputWarning, spearmanr

import config
from load_pooled import apply_mask_exact, load_all_pooled_masked, _subset_years, _sanitize

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SAMPLE_DIM_CANDIDATES = ("member", "sample", "realization", "ensemble", "ens")

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

OBS_ROW = "MCH (spatial analysis)"


def _to_float_scalar(x: xr.DataArray) -> float:
    v = float(x.values)
    return v if np.isfinite(v) else np.nan


def _spatial_mean_all_dims(da: xr.DataArray) -> float:
    dims = list(da.dims)
    if not dims:
        return _to_float_scalar(da)
    return _to_float_scalar(da.mean(dim=dims, skipna=True))


def _daily_climatology(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da
    clim = da.groupby("time.dayofyear").mean("time", skipna=True)
    return clim.reindex(dayofyear=np.arange(1, 367))


def _season_for_doy(doy: int) -> str:
    date = pd.Timestamp(year=2000, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)
    m = date.month
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def _resolve_sample_dim_pair(tas_da: xr.DataArray, pr_da: xr.DataArray, sample_dim: str) -> str | None:
    if sample_dim != "auto":
        if sample_dim not in tas_da.dims or sample_dim not in pr_da.dims:
            raise ValueError(
                f"sample dimension '{sample_dim}' must exist in both tas/pr. "
                f"tas dims={tas_da.dims}, pr dims={pr_da.dims}"
            )
        return sample_dim
    for d in SAMPLE_DIM_CANDIDATES:
        if d in tas_da.dims and d in pr_da.dims:
            return d
    return None


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

    if "dayofyear" in a.dims and a.chunks is not None:
        a = a.chunk({"dayofyear": -1})
    if "dayofyear" in b.dims and b.chunks is not None:
        b = b.chunk({"dayofyear": -1})

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


def _seasonal_daily_climatology_spearman_membermean(
    tas_da: xr.DataArray,
    pr_da: xr.DataArray,
    sample_dim: str = "auto",
) -> dict[str, float]:
    """
    Fully vectorized across sample/spatial dimensions:
    - align once
    - climatology once
    - Spearman map once per season (no per-member Python loop)
    - spatial mean, then sample mean (if sample dim exists)
    """
    sd = _resolve_sample_dim_pair(tas_da, pr_da, sample_dim)
    tas_da, pr_da = xr.align(tas_da, pr_da, join="inner")

    clim_tas = _daily_climatology(tas_da)
    clim_pr = _daily_climatology(pr_da)

    doy = np.asarray(clim_tas["dayofyear"].values, dtype=int)
    season_labels = np.asarray([_season_for_doy(d) for d in doy], dtype=object)

    out: dict[str, float] = {}
    for s in SEASONS:
        doy_vals = doy[season_labels == s]
        if doy_vals.size == 0:
            out[s] = np.nan
            continue

        tas_s = clim_tas.sel(dayofyear=doy_vals.tolist())
        pr_s = clim_pr.sel(dayofyear=doy_vals.tolist())
        rho_map = _spearman_map_over_dayofyear(tas_s, pr_s)

        if sd is None:
            out[s] = _spatial_mean_all_dims(rho_map)
            continue

        spatial_dims = [d for d in rho_map.dims if d != sd]
        rho_spatial = rho_map.mean(dim=spatial_dims, skipna=True) if spatial_dims else rho_map
        out[s] = _to_float_scalar(rho_spatial.mean(dim=sd, skipna=True))

    return out


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"))
    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--sample_dim", default="auto")
    ap.add_argument(
        "--out_csv",
        default="Analysis/BCSR_Stats/Tables/intervariable_spearman_pooled_2015_2023.csv",
    )
    args = ap.parse_args()

    loaded = load_all_pooled_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
    )

    if loaded.missing.get("tas"):
        print(f"[warn] missing pooled tas baselines: {loaded.missing['tas']}")
    if loaded.missing.get("pr"):
        print(f"[warn] missing pooled pr baselines: {loaded.missing['pr']}")

    with xr.open_dataset(args.obs_tas_file) as ds_tas:
        obs_tas = ds_tas[args.obs_tas_var].load()
    with xr.open_dataset(args.obs_pr_file) as ds_pr:
        obs_pr = ds_pr[args.obs_pr_var].load()

    obs_tas = apply_mask_exact(
        _subset_years(_sanitize(obs_tas, "tas"), args.eval_start, args.eval_end),
        loaded.hr_mask,
    )
    obs_pr = apply_mask_exact(
        _subset_years(_sanitize(obs_pr, "pr"), args.eval_start, args.eval_end),
        loaded.hr_mask,
    )

    obs_corr = _seasonal_daily_climatology_spearman_membermean(obs_tas, obs_pr, sample_dim="auto")

    corr = pd.DataFrame(index=TABLE_ROWS + [OBS_ROW], columns=SEASONS, dtype=float)

    for b in TABLE_ROWS:
        tas = loaded.data.get("tas", {}).get(b)
        pr = loaded.data.get("pr", {}).get(b)
        if tas is None or pr is None:
            continue
        vals = _seasonal_daily_climatology_spearman_membermean(tas, pr, sample_dim=args.sample_dim)
        for s in SEASONS:
            corr.loc[b, s] = vals[s]

    for s in SEASONS:
        corr.loc[OBS_ROW, s] = obs_corr[s]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, float_format="%.3f")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()