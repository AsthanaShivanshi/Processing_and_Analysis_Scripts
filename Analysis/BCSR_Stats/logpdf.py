from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
EPS = 1e-30
NBINS = 50
QLO = 0.01
QHI = 0.99

base = Path(config.BASE_DIR) / "sasthana/Downscaling"
ens_root = base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"
mask_file = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"

OBS_CFG = {
    "pr": {
        "file": base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc",
        "var": "RhiresD",
        "xlabel": "Precipitation (mm/day)",
    },
    "tas": {
        "file": base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc",
        "var": "TabsD",
        "xlabel": "Temperature (°C)",
    },
}


def _values(da: xr.DataArray) -> np.ndarray:
    vals = np.asarray(da.values, dtype=np.float32).ravel()
    return vals[np.isfinite(vals)]


def _finite_quantile_range(da: xr.DataArray, qlo: float = QLO, qhi: float = QHI) -> tuple[float, float] | None:
    vals = _values(da)
    if vals.size == 0:
        return None

    lo, hi = np.quantile(vals, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _hist_logpdf(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size < 2:
        return None
    pdf, edges = np.histogram(values, bins=bins, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, np.log(np.maximum(pdf, EPS))


def _save_logpdf_nc(var: str, curves: dict[str, tuple[np.ndarray, np.ndarray]], out_nc: Path) -> None:
    if not curves:
        return

    labels = list(curves.keys())
    x = next(iter(curves.values()))[0]
    y = np.full((len(labels), len(x)), np.nan, dtype=np.float32)

    for i, label in enumerate(labels):
        y[i, :] = curves[label][1].astype(np.float32, copy=False)

    ds = xr.Dataset(
        data_vars={"ln_pdf": (("label", "x"), y)},
        coords={"label": labels, "x": x.astype(np.float32, copy=False)},
        attrs={"variable": var, "note": "Natural log of PDF"},
    )
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)


plt.rcParams.update(
    {
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes = np.atleast_1d(axes)
panel_order = ["pr", "tas"]
panel_letters = {"pr": "(a)", "tas": "(b)"}
cmap = plt.get_cmap("tab20")
linestyles = ["-", "--", ":", "-."]

for ax, var in zip(axes, panel_order):
    cfg = OBS_CFG[var]

    loaded = load_all_pooled_masked(
        ens_root=ens_root,
        mask_hr_file=mask_file,
        mask_hr_var="TabsD",
        eval_start=eval_start,
        eval_end=eval_end,
        variables=[var],
    )

    with xr.open_dataset(Path(cfg["file"])) as ds_obs:
        obs_da = ds_obs[cfg["var"]]

        obs_da = apply_mask_exact(
            _subset_years(_sanitize(obs_da, var), eval_start, eval_end),
            loaded.hr_mask,
        ).load()

    items: list[tuple[str, xr.DataArray]] = [(obs_label, obs_da)]
    for baseline in ROW_ORDER:
        da = loaded.data.get(var, {}).get(baseline)
        if da is not None:
            items.append((baseline, da))

    ranges: list[tuple[float, float]] = []
    for _, da in items:
        mm = _finite_quantile_range(da)
        if mm is not None:
            ranges.append(mm)

    if not ranges:
        del loaded, obs_da, items
        gc.collect()
        continue

    x_lo = min(lo for lo, _ in ranges)
    x_hi = max(hi for _, hi in ranges)
    if x_lo == x_hi:
        x_hi = x_lo + 1e-6

    bins = np.linspace(x_lo, x_hi, NBINS + 1, dtype=np.float32)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i, (label, da) in enumerate(items):
        vals = _values(da)
        if vals.size < 2:
            continue

        out = _hist_logpdf(vals, bins)
        if out is None:
            continue

        x, y = out
        curves[label] = (x, y)

        if label == obs_label:
            ax.plot(x, y, color="black", lw=3.0, label=label, zorder=100)
        else:
            ax.plot(
                x,
                y,
                color=cmap(i % 20),
                lw=1.8,
                ls=linestyles[i % len(linestyles)],
                label=label,
            )

    _save_logpdf_nc(
        var=var,
        curves=curves,
        out_nc=base
        / "Processing_and_Analysis_Scripts/Analysis/BCSR_Stats/Figures"
        / f"logpdf_{var}.nc",
    )

    style_axis(ax, xlabel=cfg["xlabel"], ylabel="ln(PDF)", grid=False)
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

    del loaded, obs_da, items, curves, ranges, bins
    gc.collect()

handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

dedup = {}
for h, l in zip(handles, labels):
    dedup[l] = h

fig.legend(
    [dedup[l] for l in dedup],
    list(dedup.keys()),
    loc="lower center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, -0.08),
)

fig.subplots_adjust(bottom=0.25, wspace=0.18)

out = base / "Processing_and_Analysis_Scripts/Analysis/BCSR_Stats/Figures/logpdf.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, out)
plt.close(fig)
print(f"[ok] wrote {out}")