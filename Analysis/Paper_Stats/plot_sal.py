import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sal_seasonal(
    csv_paths,
    seasons=("DJF", "MAM", "JJA", "SON"),
    metrics=("S", "A", "L"),
    season_col="season",
    model_col="model",
    model_order=None,
    model_colors=None,
    model_markers=None,
    save_path=None,
    figsize=(15, 5),
    dpi=300,
):
    if isinstance(csv_paths, (str,)):
        csv_paths = [csv_paths]

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    if model_order is None:
        model_order = list(df[model_col].dropna().unique())

    if model_colors is None:
        model_colors = {
            "Bilinear": "#1f77b4",
            "Bicubic": "#2ca02c",
            "UNet": "#d62728",
            "DDIM_median": "#9467bd",
            "DDIM_mean": "#ff7f0e",
        }

    if model_markers is None:
        model_markers = {
            "Bilinear": "o",
            "Bicubic": "s",
            "UNet": "^",
            "DDIM_median": "D",
            "DDIM_mean": "P",
        }

    season_x = np.arange(len(seasons))
    offsets = np.linspace(-0.15, 0.15, len(model_order))

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True)

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(model_order):
            d = df[df[model_col] == model].set_index(season_col).reindex(seasons)

            ax.scatter(
                season_x + offsets[i],
                d[metric],
                s=90,
                marker=model_markers.get(model, "o"),
                color=model_colors.get(model, "gray"),
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=model,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(season_x)
        ax.set_xticklabels(seasons)
        ax.set_xlabel("Season")
        ax.set_ylabel(metric)
        ax.set_title(f"Seasonal mean {metric}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(model_order),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes