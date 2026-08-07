from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[4]
DOWNSCALING_ROOT = REPO_ROOT / "Downscaling"

DEFAULT_HIST_TAS = DOWNSCALING_ROOT / "GCM_pipeline/ALP-FINEv1.0/Ensmeans/Swiss/historical/day/tas/v20250415/MME_mean_Swiss_historical_tas_all.nc"
DEFAULT_SCEN_TAS = DOWNSCALING_ROOT / "GCM_pipeline/ALP-FINEv1.0/Ensmeans/Swiss/ssp370/day/tas/v20250415/MME_mean_Swiss_ssp370_tas_all.nc"
DEFAULT_HIST_PR = DOWNSCALING_ROOT / "GCM_pipeline/ALP-FINEv1.0/Ensmeans/Swiss/historical/day/pr/v20250415/MME_mean_Swiss_historical_pr_all.nc"
DEFAULT_SCEN_PR = DOWNSCALING_ROOT / "GCM_pipeline/ALP-FINEv1.0/Ensmeans/Swiss/ssp370/day/pr/v20250415/MME_mean_Swiss_ssp370_pr_all.nc"

SAMPLE_DIMS = ("member", "sample", "samples")


def _subset_period(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    if "time" not in da.dims:
        return da
    return da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))


def _resolve_sample_dim(da: xr.DataArray) -> str | None:
    for d in SAMPLE_DIMS:
        if d in da.dims:
            return d
    return None


def _safe_time_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean(dim="time", skipna=True) if "time" in da.dims else da


def _spatial_mean(da: xr.DataArray, sample_dim: str | None) -> float:
    dims = [d for d in da.dims if d != sample_dim]
    if dims:
        da = da.mean(dim=dims, skipna=True)
    if sample_dim and sample_dim in da.dims:
        da = da.mean(dim=sample_dim, skipna=True)
    v = float(da.values)
    return v if np.isfinite(v) else np.nan


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

    m_al, a_al = xr.align(m, anchor, join="inner")
    m2 = m_al.broadcast_like(a_al)
    return da.where(m2 > 0)


def _load_var(path: Path, var: str, mask: xr.DataArray | None) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with xr.open_dataset(path) as ds:
        da = ds[var] if var in ds.data_vars else ds[list(ds.data_vars)[0]]
        return _apply_mask(da.load(), mask)


def _trend_stats(
    hist: xr.DataArray,
    scen: xr.DataArray,
    *,
    hist_start: int,
    hist_end: int,
    scen_start: int,
    scen_end: int,
    change_mode: str,
    min_denom: float = 1e-6,
) -> tuple[float, float, float]:
    h = _subset_period(hist, hist_start, hist_end)
    s = _subset_period(scen, scen_start, scen_end)

    hist_mean_map = _safe_time_mean(h)
    scen_mean_map = _safe_time_mean(s)

    if change_mode == "absolute":
        change_map = scen_mean_map - hist_mean_map
    elif change_mode == "relative":
        denom_ok = np.abs(hist_mean_map) > min_denom
        change_map = xr.where(denom_ok, (scen_mean_map - hist_mean_map) / hist_mean_map * 100.0, np.nan)
    else:
        raise ValueError(f"Invalid change_mode: {change_mode}")

    sd = _resolve_sample_dim(hist_mean_map)
    hist_mean = _spatial_mean(hist_mean_map, sd)
    scen_mean = _spatial_mean(scen_mean_map, sd)
    change_val = _spatial_mean(change_map, sd)
    return hist_mean, scen_mean, change_val


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute coarse baseline/scenario means and changes for tas/pr.")
    ap.add_argument("--hist_tas", type=Path, default=DEFAULT_HIST_TAS)
    ap.add_argument("--scen_tas", type=Path, default=DEFAULT_SCEN_TAS)
    ap.add_argument("--hist_pr", type=Path, default=DEFAULT_HIST_PR)
    ap.add_argument("--scen_pr", type=Path, default=DEFAULT_SCEN_PR)
    ap.add_argument("--mask_file", "--mask_hr_file", dest="mask_file", type=Path, default=None)
    ap.add_argument("--mask_var", "--mask_hr_var", dest="mask_var", type=str, default=None)
    ap.add_argument("--hist_start", type=int, default=1981)
    ap.add_argument("--hist_end", type=int, default=2010)
    ap.add_argument("--scen_start", type=int, default=2070)
    ap.add_argument("--scen_end", type=int, default=2099)
    ap.add_argument("--min_denom", type=float, default=1e-6)
    ap.add_argument("--out_csv", type=Path, default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    mask = _load_mask(args.mask_file, args.mask_var)

    hist_tas = _load_var(args.hist_tas, "tas", mask)
    scen_tas = _load_var(args.scen_tas, "tas", mask)
    hist_pr = _load_var(args.hist_pr, "pr", mask)
    scen_pr = _load_var(args.scen_pr, "pr", mask)

    tas_h, tas_s, tas_abs = _trend_stats(
        hist_tas, scen_tas,
        hist_start=args.hist_start, hist_end=args.hist_end,
        scen_start=args.scen_start, scen_end=args.scen_end,
        change_mode="absolute", min_denom=args.min_denom
    )
    _, _, tas_pct = _trend_stats(
        hist_tas, scen_tas,
        hist_start=args.hist_start, hist_end=args.hist_end,
        scen_start=args.scen_start, scen_end=args.scen_end,
        change_mode="relative", min_denom=args.min_denom
    )

    pr_h, pr_s, pr_abs = _trend_stats(
        hist_pr, scen_pr,
        hist_start=args.hist_start, hist_end=args.hist_end,
        scen_start=args.scen_start, scen_end=args.scen_end,
        change_mode="absolute", min_denom=args.min_denom
    )
    _, _, pr_pct = _trend_stats(
        hist_pr, scen_pr,
        hist_start=args.hist_start, hist_end=args.hist_end,
        scen_start=args.scen_start, scen_end=args.scen_end,
        change_mode="relative", min_denom=args.min_denom
    )

    df = pd.DataFrame(
        [
            {
                "var": "tas",
                "hist_mean": tas_h,
                "scen_mean": tas_s,
                "trend_change_abs": tas_abs,
                "trend_change_pct": tas_pct,
            },
            {
                "var": "pr",
                "hist_mean": pr_h,
                "scen_mean": pr_s,
                "trend_change_abs": pr_abs,
                "trend_change_pct": pr_pct,
            },
        ]
    )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False, float_format="%.4f")
        print(f"[ok] wrote {args.out_csv}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()