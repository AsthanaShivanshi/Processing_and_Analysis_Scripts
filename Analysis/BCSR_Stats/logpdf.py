from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

import os

import config
from load_pooled import (
    ROW_ORDER,
    _sanitize,
    _subset_years,
    apply_mask_exact,
    load_all_pooled_masked,
)
from plotstyle import save_figure, style_axis

eval_start, eval_end = 2015, 2023
obs_label = "MCH (spatial analysis)"
eps = 1e-12

base = Path(config.BASE_DIR) / "sasthana/Downscaling"
ens_root = base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"
mask_file = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"

OBS_CFG = {
    "pr": {
        "file": base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc",
        "var": "RhiresD",
        "xlabel": "Precipitation (mm/day)",
        "wet_only": True,
        "to_celsius": False,
    },
    "tas": {
        "file": base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc",
        "var": "TabsD",
        "xlabel": "Temperature (°C)",
        "wet_only": False,
        "to_celsius": True,
    },
}

CB_COLORS = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#661100", "#6699CC", "#EE7733",
    "#0077BB", "#33BBEE", "#009988", "#EE3377",
]


def _spatial_mean_then_pool_time_member(da: xr.DataArray) -> np.ndarray:
    if "time" not in da.dims:
        return np.array([], dtype=float)
    spatial_dims = [d for d in da.dims if d not in ("time", "member")]
    if spatial_dims:
        da = da.mean(dim=spatial_dims, skipna=True)
    vals = np.asarray(da.values, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def _kde_logpdf(vals: np.ndarray, xg: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(vals, bw_method="scott")
    pdf = kde(xg)
    return np.log(np.clip(pdf, eps, None))


def _norm_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


def _class_of(name: str) -> str:
    n = _norm_name(name)
    if "ddim" in n:
        return "ddim"
    if "unet" in n:
        return "unet"
    return "other"


def _assign_unique_colors(labels: list[str]) -> dict[str, str]:
    ordered = sorted(labels)
    if len(ordered) > len(CB_COLORS):
        raise RuntimeError(
            f"Need {len(ordered)} unique colors but only {len(CB_COLORS)} provided."
        )
    return {lab: CB_COLORS[i] for i, lab in enumerate(ordered)}


def _assign_distinct_linestyles(labels: list[str]) -> dict[str, tuple[float, tuple[float, ...]]]:
    ddim_patterns = [(0, (6, 2, 1.2, 2)), (0, (8, 2, 1.2, 2)), (0, (10, 2, 1.2, 2))]
    unet_patterns = [(0, (6, 2.2)), (0, (8, 2.2)), (0, (10, 2.2))]
    other_patterns = [(0, (1, 2.2)), (0, (1, 3.0)), (0, (1, 4.0)), (0, (1, 5.0))]

    out: dict[str, tuple[float, tuple[float, ...]]] = {}
    c_ddim = c_unet = c_other = 0

    for lab in sorted(labels):
        cls = _class_of(lab)
        if cls == "ddim":
            out[lab] = ddim_patterns[c_ddim % len(ddim_patterns)]
            c_ddim += 1
        elif cls == "unet":
            out[lab] = unet_patterns[c_unet % len(unet_patterns)]
            c_unet += 1
        else:
            out[lab] = other_patterns[c_other % len(other_patterns)]
            c_other += 1
    return out


def _legend_rank(name: str) -> tuple[int, str]:
    cls = _class_of(name)
    if cls == "ddim":
        return (0, name)
    if cls == "unet":
        return (1, name)
    return (2, name)


def _to_celsius_if_needed(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals
    med = np.nanmedian(vals)
    if np.isfinite(med) and med > 150:
        return vals - 273.15
    return vals


sns.set_theme(style="white", context="paper", font_scale=1.35)
plt.rcParams.update({
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

loaded = load_all_pooled_masked(
    ens_root=ens_root,
    mask_hr_file=mask_file,
    mask_hr_var="TabsD",
    eval_start=eval_start,
    eval_end=eval_end,
    variables=["pr", "tas"],
)

series_by_var: dict[str, dict[str, np.ndarray]] = {}

for var in ("pr", "tas"):
    cfg = OBS_CFG[var]
    obs_file = Path(cfg["file"])
    if not obs_file.exists():
        print(f"[warn] missing obs file for {var}: {obs_file}")
        continue

    with xr.open_dataset(obs_file) as ds_obs:
        obs_da = ds_obs[cfg["var"]].load()

    obs_da = apply_mask_exact(
        _subset_years(_sanitize(obs_da, var), eval_start, eval_end),
        loaded.hr_mask,
    )

    series: dict[str, np.ndarray] = {}
    for baseline in ROW_ORDER:
        da = loaded.data.get(var, {}).get(baseline)
        if da is None:
            continue
        vals = _spatial_mean_then_pool_time_member(da)
        if cfg["to_celsius"]:
            vals = _to_celsius_if_needed(vals)
        if cfg["wet_only"]:
            vals = vals[vals > 0]
        if vals.size >= 20:
            series[baseline] = vals

    obs_vals = _spatial_mean_then_pool_time_member(obs_da)
    if cfg["to_celsius"]:
        obs_vals = _to_celsius_if_needed(obs_vals)
    if cfg["wet_only"]:
        obs_vals = obs_vals[obs_vals > 0]
    if obs_vals.size >= 20:
        series[obs_label] = obs_vals

    if series:
        series_by_var[var] = series
    else:
        print(f"[warn] no valid series for {var}")

if not series_by_var:
    raise RuntimeError("No valid series available for plotting.")

all_non_obs_labels: list[str] = []
for series in series_by_var.values():
    all_non_obs_labels.extend([k for k in series.keys() if k != obs_label])

color_map = _assign_unique_colors(sorted(set(all_non_obs_labels)))
ls_map = _assign_distinct_linestyles(sorted(set(all_non_obs_labels)))

fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.4))
axes = np.atleast_1d(axes)

panel_order = ["pr", "tas"]
panel_letters = {"pr": "(a)", "tas": "(b)"}

for ax, var in zip(axes, panel_order):
    if var not in series_by_var:
        ax.set_axis_off()
        continue

    cfg = OBS_CFG[var]
    series = series_by_var[var]

    all_vals = np.concatenate(list(series.values()))
    x_min = float(np.quantile(all_vals, 0.001))
    x_max = float(np.quantile(all_vals, 0.999))
    if var == "pr":
        x_min = max(1e-6, x_min)
    xg = np.linspace(x_min, x_max, 700)

    for label, vals in series.items():
        y = _kde_logpdf(vals, xg)
        if label == obs_label:
            ax.plot(
                xg,
                y,
                color="black",
                lw=3.0,
                linestyle="-",
                alpha=1.0,
                zorder=100,
                label=label,
            )
        else:
            ax.plot(
                xg,
                y,
                color=color_map[label],
                lw=1.8,
                linestyle=ls_map[label],
                alpha=0.95,
                zorder=20,
                label=label,
            )

    style_axis(
        ax,
        xlabel=cfg["xlabel"],
        ylabel="log p(x)",
        grid=False,
    )
    ax.grid(False)
    ax.margins(x=0.01)
    ax.text(
        0.02,
        0.97,
        panel_letters[var],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )



handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

dedup = {}
for h, l in zip(handles, labels):
    dedup[l] = h


legend_labels = sorted(dedup.keys(), key=lambda n: (-1, "") if n == obs_label else _legend_rank(n))
fig.legend(
    [dedup[l] for l in legend_labels],
    legend_labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    edgecolor="0.75",
    fontsize=13,
    handlelength=3.8,
    columnspacing=1.6,
    borderpad=0.9,
    labelspacing=1.0,
    bbox_to_anchor=(0.5, -0.10), 
)

fig.subplots_adjust(bottom=0.31, wspace=0.20)


out = Path(base / "Processing_and_Analysis_Scripts/Analysis/BCSR_Stats/Figures/logpdf.pdf")
os.makedirs(out.parent, exist_ok=True)
save_figure(fig, out)
print(f"[ok] wrote {out}")

plt.show()