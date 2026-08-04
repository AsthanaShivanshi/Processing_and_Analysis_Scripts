from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from load_means import load_all_means_masked



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




SAMPLE_DIM_CANDIDATES = ("member", "sample")


def _resolve_sample_dim(da: xr.DataArray, sample_dim: str = "auto") -> str | None:
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample dimension '{sample_dim}' not found in dims={da.dims}")
        return sample_dim
    for d in SAMPLE_DIM_CANDIDATES:
        if d in da.dims:
            return d
    return None


def _subset_period(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    if "time" not in da.dims:
        return da
    return da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))


def _spatial_mean(da: xr.DataArray, sample_dim: str | None) -> float:
    dims = [d for d in da.dims if d != sample_dim]
    if dims:
        da = da.mean(dim=dims, skipna=True)
    if sample_dim is not None and sample_dim in da.dims:
        da = da.mean(dim=sample_dim, skipna=True)
    v = float(da.values)
    return v if np.isfinite(v) else np.nan


def _year_coverage(da: xr.DataArray) -> tuple[int | None, int | None]:
    if "time" not in da.coords or da.sizes.get("time", 0) == 0:
        return None, None
    years = da["time"].dt.year.values
    return int(np.min(years)), int(np.max(years))


def trend_change_stats(
    da: xr.DataArray,
    hist_start: int,
    hist_end: int,
    scen_start: int,
    scen_end: int,
    sample_dim: str = "auto",
    min_denom: float = 1e-6,
    change_mode: str = "relative",  # "absolute" for tas, "relative" for pr
    change_col: str = "trend_change",
) -> dict[str, float]:
    sd = _resolve_sample_dim(da, sample_dim)

    hist = _subset_period(da, hist_start, hist_end)
    scen = _subset_period(da, scen_start, scen_end)

    if hist.sizes.get("time", 0) == 0 or scen.sizes.get("time", 0) == 0:
        return {"hist_mean": np.nan, "scen_mean": np.nan, change_col: np.nan}

    hist_mean_map = hist.mean(dim="time", skipna=True)
    scen_mean_map = scen.mean(dim="time", skipna=True)

    if change_mode == "absolute":
        trend_map = scen_mean_map - hist_mean_map
    elif change_mode == "relative":
        denom_ok = np.abs(hist_mean_map) > min_denom
        trend_map = xr.where(
            denom_ok,
            (scen_mean_map - hist_mean_map) / hist_mean_map * 100.0,
            np.nan,
        )
    else:
        raise ValueError(f"unsupported change_mode='{change_mode}'")

    return {
        "hist_mean": _spatial_mean(hist_mean_map, sd),
        "scen_mean": _spatial_mean(scen_mean_map, sd),
        change_col: _spatial_mean(trend_map, sd),
    }


def _build_parser() -> argparse.ArgumentParser:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ap = argparse.ArgumentParser(description="Compute historical vs scenario trend statistics.")
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/Ensmeans"))



    ap.add_argument("--var", choices=["tas", "pr"], default=None)
    ap.add_argument("--vars", nargs="+", choices=["tas", "pr"], default=["tas", "pr"])

    ap.add_argument("--mask_hr_file", required=True)
    ap.add_argument("--mask_hr_var", default="TabsD")

    ap.add_argument("--hist_start", type=int, default=1981)
    ap.add_argument("--hist_end", type=int, default=2010)

    ap.add_argument("--scen_start", type=int, default=2070)
    ap.add_argument("--scen_end", type=int, default=2099)

    ap.add_argument("--sample_dim", default="auto", choices=["auto", "member", "sample"])


    ap.add_argument("--min_denom", type=float, default=1e-6)
    ap.add_argument("--reference_row", default=None)
    ap.add_argument("--out_csv", default=None)
    return ap



def _resolve_out_csv(args: argparse.Namespace, var: str, n_vars: int) -> Path:
    if args.out_csv:
        p = Path(args.out_csv)
        if n_vars == 1:
            return p
        if p.suffix.lower() == ".csv":
            return p.with_name(f"{p.stem}_{var}{p.suffix}")
        return p / f"trend_{var}_{args.hist_start}_{args.hist_end}_to_{args.scen_start}_{args.scen_end}.csv"

    return Path(
        f"Analysis/BCSR_Stats/Tables/trend_{var}_{args.hist_start}_{args.hist_end}_to_{args.scen_start}_{args.scen_end}.csv"
    )


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    if args.hist_start > args.hist_end:
        raise ValueError("hist_start must be <= hist_end")
    if args.scen_start > args.scen_end:
        raise ValueError("scen_start must be <= scen_end")
    if args.min_denom <= 0:
        raise ValueError("min_denom must be > 0")

    ens_root = Path(args.ens_root).expanduser().resolve()
    mask_file = Path(args.mask_hr_file).expanduser().resolve()

    print(f"[info] cwd={Path.cwd()}")
    print(f"[info] ens_root={ens_root}")
    print(f"[info] mask_hr_file={mask_file}")

    if not ens_root.exists():
        raise FileNotFoundError(f"ens_root does not exist: {ens_root}")
    if not mask_file.exists():
        raise FileNotFoundError(f"mask_hr_file does not exist: {mask_file}")
    

    vars_to_run = [args.var] if args.var else args.vars
    vars_to_run = list(dict.fromkeys(vars_to_run))

    for var in vars_to_run:
        if var == "tas":
            change_mode = "absolute"
            change_col = "trend_change_abs"
            bias_col = "trend_bias_vs_reference_abs"
        else:
            change_mode = "relative"
            change_col = "trend_change_pct"
            bias_col = "trend_bias_vs_reference_pct"

        loaded = load_all_means_masked(
            ens_root=ens_root,
            mask_hr_file=mask_file,
            mask_hr_var=args.mask_hr_var,
            eval_start=min(args.hist_start, args.scen_start),
            eval_end=max(args.hist_end, args.scen_end),
            variables=[var],
        )

        missing = loaded.missing.get(var, [])
        available = list(loaded.data.get(var, {}).keys())

        if missing:
            print(f"[warn] missing baselines for {var}: {missing}")
        print(f"[info] loaded baselines for {var}: {available}")

        if not available:
            raise RuntimeError(
                f"No baselines loaded for var={var}. "
                f"Check --ens_root, template paths, and file names."
            )

        table = pd.DataFrame(index=TABLE_ROWS, columns=["hist_mean", "scen_mean", change_col], dtype=float)

        for baseline in TABLE_ROWS:
            da = loaded.data.get(var, {}).get(baseline)
            if da is None:
                print(f"[warn] baseline not loaded: {baseline}")
                continue

            y0, y1 = _year_coverage(da)
            print(f"[info] {var} | {baseline}: year coverage={y0}..{y1}, dims={da.dims}")

            vals = trend_change_stats(
                da=da,
                hist_start=args.hist_start,
                hist_end=args.hist_end,
                scen_start=args.scen_start,
                scen_end=args.scen_end,
                sample_dim=args.sample_dim,
                min_denom=args.min_denom,
                change_mode=change_mode,
                change_col=change_col,
            )
            for k, v in vals.items():
                table.loc[baseline, k] = v

        if table[change_col].isna().all():
            raise RuntimeError(
                f"All values in '{change_col}' are NaN for var={var}. "
                "Likely time-window mismatch or failed data loading."
            )

        if args.reference_row and args.reference_row in table.index:
            ref_val = table.loc[args.reference_row, change_col]
            table[bias_col] = table[change_col] - ref_val
        elif args.reference_row:
            print(f"[warn] reference_row not found in table index: {args.reference_row}")

        out_csv = _resolve_out_csv(args, var, len(vars_to_run))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, float_format="%.4f")
        print(f"[ok] wrote {out_csv}")

        del loaded, table
        gc.collect()


if __name__ == "__main__":
    main()