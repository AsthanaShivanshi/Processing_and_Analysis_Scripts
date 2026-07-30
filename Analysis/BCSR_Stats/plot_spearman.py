from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotstyle import apply_paper_style, get_model_color, save_paper_figure

SEASONS = ["DJF", "MAM", "JJA", "SON"]
OBS_LABELS = ["Observations", "MCH (spatial analysis)", "MeteoSwiss Spatial Analysis (MCH)"]


def _get_obs_label(df: pd.DataFrame) -> str:
    for label in OBS_LABELS:
        if label in df.index:
            return label
    raise ValueError(f"No observation row found in {df.index.tolist()}")


def _family_color(name: str) -> str:
    if "DDIM" in name:
        return get_model_color("DDIM")
    if "U-Net" in name or "UNet" in name:
        return get_model_color("UNet")
    if "Bilinear" in name:
        return get_model_color("Bilinear")
    return get_model_color("Coarse")


def _method_marker(name: str) -> str:
    if "EQM" in name:
        return "o"
    if "CDF-t" in name:
        return "s"
    if "dOTC" in name:
        return "^"
    return "D"


def plot_spearman(
    in_csv: str | Path = "Analysis/BCSR_Stats/Tables/intervariable_spearman_2015_2023.csv",
    out_pdf: str | Path = "Analysis/BCSR_Stats/Figures/intervariable_spearman_2015_2023.pdf",
    dpi: int = 400,
):
    apply_paper_style()

    df = pd.read_csv(in_csv, index_col=0)
    obs_label = _get_obs_label(df)

    obs_row = df.loc[obs_label]
    corr = df.drop(index=obs_label)
    table_rows = corr.index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(table_rows))

    for i, s in enumerate(SEASONS):
        ax = axes[i]
        ax.axhline(float(obs_row[s]), color="black", lw=1.0, ls="dotted", zorder=2)
        for j, b in enumerate(table_rows):
            ax.scatter(
                j,
                float(corr.loc[b, s]),
                color=_family_color(b),
                marker=_method_marker(b),
                s=55,
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )
        ax.set_title(s, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(table_rows, rotation=40, ha="right", fontsize=10)
        ax.set_ylabel("Spearman r", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)

    handles = [plt.Line2D([0], [0], color="black", lw=1.0, linestyle="dotted", label=obs_label)]
    handles += [
        plt.Line2D(
            [0], [0],
            marker=_method_marker(b),
            linestyle="",
            color=_family_color(b),
            markeredgecolor="black",
            markeredgewidth=0.2,
            label=b,
        )
        for b in table_rows
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_pdf)
    plt.close(fig)

    print(f"[ok] wrote {out_pdf}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="Analysis/BCSR_Stats/Tables/intervariable_spearman_2015_2023.csv")
    ap.add_argument("--out_pdf", default="Analysis/BCSR_Stats/Figures/intervariable_spearman_2015_2023.pdf")
    ap.add_argument("--dpi", type=int, default=800)
    args = ap.parse_args()

    plot_spearman(
        in_csv=args.in_csv,
        out_pdf=args.out_pdf,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()