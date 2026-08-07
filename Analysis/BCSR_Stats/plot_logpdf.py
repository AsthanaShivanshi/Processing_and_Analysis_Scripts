from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotstyle import save_figure, style_axis

obs_label = "MCH (spatial analysis)"

def plot_from_nc(nc_file: str | Path, out_pdf: str | Path) -> None:
    ds = xr.open_dataset(nc_file)

    fig, ax = plt.subplots(figsize=(9, 6))

    labels = list(ds["label"].values)
    for label in labels:
        x = ds["x"].sel(label=label).values
        y = ds["ln_pdf"].sel(label=label).values
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue

        if str(label) == obs_label:
            ax.plot(x[m], y[m], color="black", lw=3.2, label=str(label), zorder=100)
        else:
            ax.plot(x[m], y[m], lw=1.8, label=str(label), alpha=0.95)

    style_axis(ax, xlabel="Value", ylabel="ln(PDF)", grid=False)
    ax.legend(frameon=True)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_pdf)
    plt.close(fig)