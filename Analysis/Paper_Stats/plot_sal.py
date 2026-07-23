import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plotstyle import apply_paper_style, get_model_color

_METRIC_LABELS = {
    "S": "Structure (S)",
    "A": "Amplitude (A)",
    "L": "Location (L)",
}

_ENSEMBLE_LABELS = {
    ("ddim", "ensemble_sample"): "DDIM samples",
    ("ddim", "ensemble_mean"): "DDIM mean",
    ("ddim", "ensemble_median"): "DDIM median",
    ("cfm", "ensemble_sample"): "CFM samples",
    ("cfm", "ensemble_mean"): "CFM mean",
    ("cfm", "ensemble_median"): "CFM median",
}

_ENSEMBLE_COLORS = {
    ("ddim", "ensemble_sample"): "#000000",
    ("ddim", "ensemble_mean"): "#555555",
    ("ddim", "ensemble_median"): "#999999",
    ("cfm", "ensemble_sample"): "#7e9f1f",
    ("cfm", "ensemble_mean"): "#9fbe34",
    ("cfm", "ensemble_median"): "#c2db66",
}


def plot_sal_box_seasonal(
    csv_path,
    season="JJA",
    save_path=None,
    models=(
        "Bilinear",
        "Bicubic",
        "UNet",
        "DDIM samples",
        "DDIM median",
        "CFM samples",
        "CFM median",
    ),
    figsize=(14, 4),
    dpi=1500,
    x_tick_fontsize=8,
    y_tick_fontsize=9,
    title_fontsize=11,
    y_label_fontsize=9,
    x_tick_rotation=30,
    plot_kind="violin",  # "violin" or "box"
    x_spacing=1.25,
):
    apply_paper_style()

    df = pd.read_csv(csv_path).copy()
    df["season"] = df["season"].astype(str).str.strip().str.upper()
    df["model_norm"] = df["model"].astype(str).str.strip().str.lower()
    df["type_norm"] = df["type"].astype(str).str.strip().str.lower()

    season = str(season).strip().upper()
    df = df[df["season"] == season].copy()

    if df.empty:
        raise ValueError(f"No rows found for season='{season}' in {csv_path}")

    requested = {str(m).strip().lower() for m in models}
    entries = []

    # Deterministic models requested by exact name
    for m in models:
        m_clean = str(m).strip()
        m_norm = m_clean.lower()
        if "ddim" in m_norm or "cfm" in m_norm:
            continue
        d = df[
            (df["model_norm"] == m_norm) &
            (df["type_norm"] == "deterministic")
        ]
        if not d.empty:
            entries.append((m_clean, d, get_model_color(m_clean)))

    for (model_norm, type_norm), label in _ENSEMBLE_LABELS.items():
        if label.lower() not in requested:
            continue
        d = df[
            (df["model_norm"] == model_norm) &
            (df["type_norm"] == type_norm)
        ]
        if not d.empty:
            entries.append((label, d, _ENSEMBLE_COLORS[(model_norm, type_norm)]))

    if not entries:
        raise ValueError(f"No plottable groups found for season='{season}'")

    labels = [e[0] for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    metrics = ["S", "A", "L"]

    plot_kind = str(plot_kind).strip().lower()
    if plot_kind not in {"violin", "box"}:
        raise ValueError("plot_kind must be either 'violin' or 'box'")

    for ax, metric in zip(axes, metrics):
        data = [e[1][metric].dropna().values for e in entries]
        positions = np.arange(len(labels)) * x_spacing + 1

        if plot_kind == "violin":
            vp = ax.violinplot(
                data,
                positions=positions,
                widths=min(0.8, 0.75 * x_spacing),
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body, (_, _, color) in zip(vp["bodies"], entries):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.25)
                body.set_linewidth(1.2)

            # Overlay IQR + median
            for x, vals in zip(positions, data):
                if len(vals) == 0:
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.vlines(x, q1, q3, color="black", linewidth=2.0, zorder=3)
                ax.scatter([x], [med], color="black", s=10, zorder=4)

        else:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=min(0.6, 0.55 * x_spacing),
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
            )

            for patch, (_, _, color) in zip(bp["boxes"], entries):
                patch.set_facecolor(color)
                patch.set_alpha(0.30)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(_METRIC_LABELS[metric], fontsize=title_fontsize)
        ax.set_ylabel(metric, fontsize=y_label_fontsize)

        ax.tick_params(axis="x", labelsize=x_tick_fontsize, rotation=x_tick_rotation)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

        ax.tick_params(axis="y", labelsize=y_tick_fontsize)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Pooled framewise SAL values for {season} (test set, 2015-2023)",
        y=1.03,
        fontsize=12,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes