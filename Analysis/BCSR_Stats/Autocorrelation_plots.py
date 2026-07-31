from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
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
    out = []
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


def _finite_concat(arrays: list[np.ndarray]) -> np.ndarray:
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)


def _nan_template_like(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims and da.sizes.get("time", 0) > 0:
        return da.isel(time=0, drop=True) * np.nan
    return da.squeeze(drop=True) * np.nan


def _remove_seasonal_cycle(da: xr.DataArray) -> xr.DataArray:
    clim = da.groupby("time.dayofyear").mean("time", skipna=True)
    return da.groupby("time.dayofyear") - clim


def _detrend_linear_time(da: xr.DataArray) -> xr.DataArray:
    if da.sizes.get("time", 0) < 3:
        return da
    t = xr.DataArray(np.arange(da.sizes["time"]), dims="time", coords={"time": da.time})
    coeffs = da.polyfit(dim="time", deg=1, skipna=True)
    trend = xr.polyval(t, coeffs.polyfit_coefficients)
    return da - trend


def _seasonal_gridwise_autocorr_maps(
    da: xr.DataArray,
    lags: list[int],
) -> dict[tuple[str, int], xr.DataArray]:
    """
    Anomaly persistence maps:
      1) remove seasonal cycle
      2) detrend
      3) seasonal lag autocorrelation maps
    """
    out: dict[tuple[str, int], xr.DataArray] = {}
    nan_template = _nan_template_like(da)

    da_resid = _detrend_linear_time(_remove_seasonal_cycle(da))

    for s in SEASONS:
        ds = da_resid.where(da_resid.time.dt.season == s, drop=True)

        for lag in lags:
            if ds.sizes.get("time", 0) <= lag:
                out[(s, lag)] = nan_template
                continue
            out[(s, lag)] = xr.corr(ds, ds.shift(time=lag), dim="time")

    return out


def _add_small_top_legend(ax: plt.Axes, label: str) -> None:

    handle = Patch(facecolor="none", edgecolor="none", label=label)
    leg = ax.legend(
        handles=[handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7,
        frameon=True,
        handlelength=0.0,
        handletextpad=0.2,
        borderpad=0.2,
    )

    handles = []
    if hasattr(leg, "legendHandles"):  # older matplotlib
        handles = leg.legendHandles
    elif hasattr(leg, "legend_handles"):  # newer matplotlib
        handles = leg.legend_handles

    for h in handles:
        try:
            h.set_visible(False)
        except Exception:
            pass


def _plot_diff_matrix_for_var_lag_season(
    var: str,
    lag: int,
    season: str,
    obs_maps: dict[tuple[str, int], xr.DataArray],
    model_maps_by_baseline: dict[str, dict[tuple[str, int], xr.DataArray]],
    out_png: Path,
) -> None:
    nrows, ncols = 5, 2

    diffs_for_scale = []
    for b in TABLE_ROWS:
        if b not in model_maps_by_baseline:
            continue
        diff = model_maps_by_baseline[b][(season, lag)] - obs_maps[(season, lag)]
        diffs_for_scale.append(diff.values)

    finite_vals = _finite_concat(diffs_for_scale)
    if finite_vals.size == 0:
        print(f"[warn] no finite values for plotting {var}, {season}, lag={lag}")
        return

    if var == "tas":
        vmax = float(np.nanquantile(np.abs(finite_vals), 0.99))
        if np.isclose(vmax, 0.0):
            vmax = 1e-6
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = float(np.nanquantile(finite_vals, 0.01))
        vmax = float(np.nanquantile(finite_vals, 0.99))
        if np.isclose(vmin, vmax):
            eps = 1e-6 if vmax == 0 else 0.05 * abs(vmax)
            vmin -= eps
            vmax += eps
        cmap = "viridis"

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.4 * ncols, 2.9 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)
    flat_axes = axes.ravel()

    mappable = None
    for i, b in enumerate(TABLE_ROWS):
        ax = flat_axes[i]

        if b not in model_maps_by_baseline:
            ax.text(0.5, 0.5, "NA", ha="center", va="center", transform=ax.transAxes)
            _add_small_top_legend(ax, b)
            ax.set_axis_off()
            continue

        diff = model_maps_by_baseline[b][(season, lag)] - obs_maps[(season, lag)]
        mappable = diff.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        _add_small_top_legend(ax, b)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for k in range(len(TABLE_ROWS), nrows * ncols):
        flat_axes[k].set_axis_off()

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=flat_axes.tolist(), shrink=0.9, pad=0.01)
        cbar.set_label("Anomaly persistence difference (model - MCH)")

    fig.suptitle(
        f"{var} | {season} | lag={lag} | anomaly persistence (gridwise, model - MCH)",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[ok] wrote {out_png}")


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

    ap.add_argument("--plot_lags", default="1,2,3", help="e.g. '1' or '1,2,3'")
    ap.add_argument("--plot_dir", default="Analysis/BCSR_Stats/Plots/Autocorrelation_Diff_2015_2023")
    args = ap.parse_args()

    plot_lags = _parse_lags(args.plot_lags)

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

    obs_maps = {
        "tas": _seasonal_gridwise_autocorr_maps(obs_tas, plot_lags),
        "pr": _seasonal_gridwise_autocorr_maps(obs_pr, plot_lags),
    }

    model_maps: dict[str, dict[str, dict[tuple[str, int], xr.DataArray]]] = {"tas": {}, "pr": {}}
    for b in TABLE_ROWS:
        for v in VARS:
            da = loaded.data.get(v, {}).get(b)
            if da is None:
                continue
            model_maps[v][b] = _seasonal_gridwise_autocorr_maps(da, plot_lags)

    plot_dir = Path(args.plot_dir)
    for v in VARS:
        for lag in plot_lags:
            for s in SEASONS:
                out_png = plot_dir / f"{v}_{s}_gridwise_anomaly_autocorr_diff_lag{lag}.png"
                _plot_diff_matrix_for_var_lag_season(
                    var=v,
                    lag=lag,
                    season=s,
                    obs_maps=obs_maps[v],
                    model_maps_by_baseline=model_maps[v],
                    out_png=out_png,
                )


if __name__ == "__main__":
    main()