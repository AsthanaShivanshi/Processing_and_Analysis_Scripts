from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASONS = ["DJF", "MAM", "JJA", "SON"]
OBS_LABEL = "MeteoSwiss Spatial Analysis (MCH), 2015-2023"


def plot_spearman(
    in_csv: str | Path = "Analysis/BCSR_Stats/Tables/intervariable_spearman_2015_2023.csv",
    out_png: str | Path = "Analysis/BCSR_Stats/Figures/intervariable_spearman_2015_2023.png",
    dpi: int = 400,
):
    df = pd.read_csv(in_csv, index_col=0)

    if OBS_LABEL not in df.index:
        raise ValueError(f"'{OBS_LABEL}' row not found in {in_csv}")

    obs_row = df.loc[OBS_LABEL]
    corr = df.drop(index=OBS_LABEL)
    table_rows = corr.index.tolist()

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(table_rows)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(table_rows))

    for i, s in enumerate(SEASONS):
        ax = axes[i]
        ax.axhline(float(obs_row[s]), color="black", lw=2)
        for j, b in enumerate(table_rows):
            ax.scatter(j, float(corr.loc[b, s]), color=colors[j], s=40)
        ax.set_title(s)
        ax.set_xticks(x)
        ax.set_xticklabels(table_rows, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Spearman r (tas vs pr)")
        ax.grid(alpha=0.25)

    handles = [plt.Line2D([0], [0], color="black", lw=2, label=OBS_LABEL)]
    handles += [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[j], label=table_rows[j])
        for j in range(len(table_rows))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    print(f"[ok] wrote {out_png}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="Analysis/BCSR_Stats/Tables/intervariable_spearman_2015_2023.csv")
    ap.add_argument("--out_png", default="Analysis/BCSR_Stats/Figures/intervariable_spearman_2015_2023.png")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    plot_spearman(
        in_csv=args.in_csv,
        out_png=args.out_png,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()