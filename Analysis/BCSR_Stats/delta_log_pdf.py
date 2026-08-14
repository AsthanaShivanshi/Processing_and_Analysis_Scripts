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

MODEL_COLORS = {
    "MCH (spatial analysis)": "black",
    "Coarse": "#7f7f7f",
    "Bicubic": "#1f77b4",
    "Bilinear": "#ff7f0e",
    "UNet": "#2ca02c",
    "DDIM": "#d62728",
    "CFM": "#9467bd",
}

MODEL_STYLES = {
    "MCH (spatial analysis)": dict(color="black", ls="-", lw=1.0, zorder=10),
    "Coarse": dict(color="#7f7f7f", ls="--", lw=0.8, zorder=10),
    "Bicubic": dict(color="#1f77b4", ls="-.", lw=0.8, zorder=10),
    "Bilinear": dict(color="#ff7f0e", ls="-.", lw=0.8, zorder=10),
    "UNet": dict(color="#2ca02c", ls="-", lw=1.0, zorder=10),
    "DDIM": dict(color="#d62728", ls="-", lw=1.0, zorder=10),
    "CFM": dict(color="#9467bd", ls="-", lw=1.0, zorder=10),
}


def _values(da: xr.DataArray) -> np.ndarray:
    vals = np.asarray(da.values, dtype=np.float32).ravel()
    return vals[np.isfinite(vals)]


def _finite_quantile_range(
    da: xr.DataArray,
    qlo: float = QLO,
    qhi: float = QHI,
) -> tuple[float, float] | None:
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


plt.rcParams.update(
    {
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex="col")
axes = np.asarray(axes)

panel_order = ["pr", "tas"]
panel_letters = {
    ("pr", "pdf"): "(a)",
    ("tas", "pdf"): "(b)",
    ("pr", "delta"): "(c)",
    ("tas", "delta"): "(d)",
}
cmap = plt.get_cmap("tab20")
linestyles = ["-", "--", ":", "-."]

for col, var in enumerate(panel_order):
    ax_pdf = axes[0, col]
    ax_delta = axes[1, col]
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
    obs_curve: tuple[np.ndarray, np.ndarray] | None = None

    for i, (label, da) in enumerate(items):
        vals = _values(da)
        if vals.size < 2:
            continue

        out = _hist_logpdf(vals, bins)
        if out is None:
            continue

        x, y = out
        curves[label] = (x, y)

        sty = dict(MODEL_STYLES.get(label, {}))
        if label not in sty:
            sty = {
                "color": MODEL_COLORS.get(label, cmap(i % 20)),
                "ls": linestyles[i % len(linestyles)],
                "lw": 1.8,
            }

        ax_pdf.plot(
            x,
            y,
            label=label,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
            **sty,
        )

        if label == obs_label:
            obs_curve = (x, y)

    if obs_curve is not None:
        x_obs, y_obs = obs_curve
        ax_delta.axhline(0.0, color="black", lw=1.5, ls="--", label="0 (perfect)")

        for i, (label, (x, y)) in enumerate(curves.items()):
            if label == obs_label:
                continue
            delta = y - y_obs
            sty = dict(MODEL_STYLES.get(label, {}))
            if label not in sty:
                sty = {
                    "color": MODEL_COLORS.get(label, cmap(i % 20)),
                    "ls": linestyles[i % len(linestyles)],
                    "lw": 1.8,
                }

            ax_delta.plot(
                x,
                delta,
                label=label,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                **sty,
            )

    style_axis(ax_pdf, xlabel=cfg["xlabel"], ylabel="ln(PDF)", grid=False)
    style_axis(ax_delta, xlabel=cfg["xlabel"], ylabel="Δ ln(PDF)", grid=False)

    ax_pdf.grid(True, which="major", alpha=0.18, linestyle="--", linewidth=0.8)
    ax_pdf.minorticks_on()
    ax_pdf.grid(True, which="minor", alpha=0.08, linestyle=":", linewidth=0.6)

    ax_delta.grid(True, which="major", alpha=0.18, linestyle="--", linewidth=0.8)
    ax_delta.minorticks_on()
    ax_delta.grid(True, which="minor", alpha=0.08, linestyle=":", linewidth=0.6)

    ax_pdf.set_title(f"{var.upper()} log-PDF")
    ax_delta.set_title(f"{var.upper()} delta vs obs")

    ax_pdf.text(
        0.02,
        0.97,
        panel_letters[(var, "pdf")],
        transform=ax_pdf.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax_delta.text(
        0.02,
        0.97,
        panel_letters[(var, "delta")],
        transform=ax_delta.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )

    del loaded, obs_da, items, curves, ranges, bins
    gc.collect()

handles, labels = [], []
for ax in axes.ravel():
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
    ncol=4,
    frameon=True,
    fancybox=False,
    edgecolor="0.75",
    facecolor="white",
    framealpha=0.95,
    handlelength=2.6,
    columnspacing=1.4,
    bbox_to_anchor=(0.5, -0.02),
)

fig.subplots_adjust(bottom=0.12, wspace=0.18, hspace=0.26)

out = base / "Processing_and_Analysis_Scripts/Analysis/BCSR_Stats/Figures/logpdf_delta.png"
out.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, out)
plt.close(fig)
print(f"[ok] wrote {out}")