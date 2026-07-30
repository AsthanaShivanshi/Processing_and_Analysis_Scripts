from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

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


def _daily_spatial_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean(dim=[d for d in da.dims if d != "time"], skipna=True)


def _remove_seasonal_mean(ts: xr.DataArray) -> xr.DataArray:
    clim = ts.groupby("time.season").mean("time", skipna=True)
    return ts.groupby("time.season") - clim


def _seasonal_spearman(tas_ts: xr.DataArray, pr_ts: xr.DataArray) -> dict[str, float]:
    tas_ts, pr_ts = xr.align(tas_ts, pr_ts, join="inner")
    out: dict[str, float] = {}
    for s in SEASONS:
        t = tas_ts.where(tas_ts.time.dt.season == s, drop=True).values
        p = pr_ts.where(pr_ts.time.dt.season == s, drop=True).values
        m = np.isfinite(t) & np.isfinite(p)
        out[s] = float(spearmanr(t[m], p[m]).correlation) if m.sum() >= 3 else np.nan
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

    obs_corr = _seasonal_spearman(
        _remove_seasonal_mean(_daily_spatial_mean(obs_tas)),
        _remove_seasonal_mean(_daily_spatial_mean(obs_pr)),
    )

    corr = pd.DataFrame(index=TABLE_ROWS, columns=SEASONS, dtype=float)
    for b in TABLE_ROWS:
        tas = loaded.data.get("tas", {}).get(b)
        pr = loaded.data.get("pr", {}).get(b)
        if tas is None or pr is None:
            continue
        vals = _seasonal_spearman(
            _remove_seasonal_mean(_daily_spatial_mean(tas)),
            _remove_seasonal_mean(_daily_spatial_mean(pr)),
        )
        for s in SEASONS:
            corr.loc[b, s] = vals[s]

    corr.loc["Observations", SEASONS] = [obs_corr[s] for s in SEASONS]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, float_format="%.4f")

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(TABLE_ROWS)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(TABLE_ROWS))

    for i, s in enumerate(SEASONS):
        ax = axes[i]
        ax.axhline(obs_corr[s], color="black", lw=2)
        for j, b in enumerate(TABLE_ROWS):
            ax.scatter(j, corr.loc[b, s], color=colors[j], s=40)
        ax.set_title(s)
        ax.set_xticks(x)
        ax.set_xticklabels(TABLE_ROWS, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Spearman r (tas vs pr)")
        ax.grid(alpha=0.25)

    handles = [plt.Line2D([0], [0], color="black", lw=2, label="Observations")]
    handles += [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[j], label=TABLE_ROWS[j]) for j in range(len(TABLE_ROWS))]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()