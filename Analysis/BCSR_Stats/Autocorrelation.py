from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from load_means import (
    ROW_ORDER,
    apply_mask_exact,
    load_all_means_masked,
    _sanitize,
    _subset_years,
)

VARS = ["tas", "pr"]
TABLE_ROWS = ROW_ORDER

OBS_ROW = "MCH (spatial analysis)"
SAMPLE_DIM_CANDIDATES = ("member", "sample")
EPS = 1e-6


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


def _resolve_sample_dim(da: xr.DataArray, sample_dim: str) -> str | None:
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample dimension '{sample_dim}' not found in dims={da.dims}")
        return sample_dim
    for d in SAMPLE_DIM_CANDIDATES:
        if d in da.dims:
            return d
    return None


def _to_float_scalar(x: xr.DataArray) -> float:
    v = float(x.values)
    return v if np.isfinite(v) else np.nan


def _monthly_mean_autocorr_spatial_mean(
    da: xr.DataArray,
    lags: list[int],
    sample_dim: str = "auto",
) -> dict[int, float]:
    sd = _resolve_sample_dim(da, sample_dim)
    monthly = da.resample(time="MS").mean("time", skipna=True).chunk({"time": -1})

    n_time = monthly.sizes.get("time", 0)
    if n_time < 3:
        return {lag: np.nan for lag in lags}

    out: dict[int, float] = {}
    for lag in lags:
        if n_time <= lag:
            out[lag] = np.nan
            continue

        corr_map = xr.corr(monthly, monthly.shift(time=lag), dim="time")
        z = np.arctanh(corr_map.clip(min=-1 + EPS, max=1 - EPS))

        if sd is None:
            mean_dims = list(z.dims)
            z_mean = z.mean(dim=mean_dims, skipna=True) if mean_dims else z
            out[lag] = _to_float_scalar(np.tanh(z_mean))
            continue

        spatial_dims = [d for d in z.dims if d != sd]
        z_spatial = z.mean(dim=spatial_dims, skipna=True) if spatial_dims else z
        r_sample = np.tanh(z_spatial)
        out[lag] = _to_float_scalar(r_sample.mean(dim=sd, skipna=True))

    return out


def _col(lag: int) -> str:
    return f"lag{lag}"


def _load_obs(args, var: str, hr_mask: xr.DataArray) -> xr.DataArray:
    if var == "tas":
        with xr.open_dataset(args.obs_tas_file) as ds:
            da = ds[args.obs_tas_var]
    else:
        with xr.open_dataset(args.obs_pr_file) as ds:
            da = ds[args.obs_pr_var]

    da = apply_mask_exact(
        _subset_years(_sanitize(da, var), args.eval_start, args.eval_end),
        hr_mask,
    )
    return da


def _resolve_out_csv(args: argparse.Namespace, var: str, n_vars: int) -> Path:
    if args.out_csv:
        p = Path(args.out_csv)
        if n_vars == 1:
            return p
        if p.suffix.lower() == ".csv":
            return p.with_name(f"{p.stem}_{var}{p.suffix}")
        return p / f"autocorrelation_monthly_mean_ensmeans_{var}_{args.eval_start}_{args.eval_end}.csv"
    return Path(
        f"Analysis/BCSR_Stats/Tables/autocorrelation_monthly_mean_ensmeans_{var}_{args.eval_start}_{args.eval_end}.csv"
    )


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/Ensmeans"))

    # Backward-compatible: --var still works; default run is both vars
    ap.add_argument("--var", choices=VARS, default=None)
    ap.add_argument("--vars", nargs="+", choices=VARS, default=VARS)

    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--lags", default="1,2,3,4,5")
    ap.add_argument("--sample_dim", default="auto", choices=["auto", "member", "sample"])
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    lags = _parse_lags(args.lags)
    vars_to_run = [args.var] if args.var else args.vars
    vars_to_run = list(dict.fromkeys(vars_to_run))

    loaded = load_all_means_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=vars_to_run,
    )

    for var in vars_to_run:
        if loaded.missing.get(var):
            print(f"[warn] missing ensmean baselines for {var}: {loaded.missing[var]}")

        obs = _load_obs(args, var, loaded.hr_mask)
        cols = [_col(lag) for lag in lags]
        table = pd.DataFrame(index=TABLE_ROWS + [OBS_ROW], columns=cols, dtype=float)

        obs_vals = _monthly_mean_autocorr_spatial_mean(obs, lags, sample_dim="auto")
        for lag in lags:
            table.loc[OBS_ROW, _col(lag)] = obs_vals[lag]

        for baseline in TABLE_ROWS:
            da = loaded.data.get(var, {}).get(baseline)
            if da is None:
                continue
            vals = _monthly_mean_autocorr_spatial_mean(da, lags, sample_dim=args.sample_dim)
            for lag in lags:
                table.loc[baseline, _col(lag)] = vals[lag]

        out_csv = _resolve_out_csv(args, var, len(vars_to_run))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, float_format="%.3f")
        print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()