import pandas as pd
import matplotlib.pyplot as plt

from plotstyle import apply_paper_style, get_model_color

_METRIC_LABELS = {
    "S": "Structure (S)",
    "A": "Amplitude (A)",
    "L": "Location (L)",
}


def plot_sal_box_seasonal(
    csv_path,
    season="JJA",
    save_path=None,
    models=("Bilinear", "Bicubic", "UNet", "DDIM", "DDIM mean", "DDIM median"),
    figsize=(14, 4),
    dpi=1500,
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

    # Explicit plotting entries (no helper abstraction)
    entries = []

    # Deterministic models as-is
    for m in models:
        if str(m).strip().lower() == "ddim":
            continue
        d = df[(df["model_norm"] == str(m).strip().lower()) & (df["type_norm"] == "deterministic")]
        if not d.empty:
            entries.append((m, d, get_model_color(m)))

    # DDIM pooled samples + mean + median as separate boxes
    d_samples = df[(df["model_norm"] == "ddim") & (df["type_norm"] == "ensemble_sample")]
    d_mean = df[(df["model_norm"] == "ddim") & (df["type_norm"] == "ensemble_mean")]
    d_median = df[(df["model_norm"] == "ddim") & (df["type_norm"] == "ensemble_median")]

    if not d_samples.empty:
        entries.append(("DDIM samples", d_samples, "#000000"))
    if not d_mean.empty:
        entries.append(("DDIM mean", d_mean, "#555555"))
    if not d_median.empty:
        entries.append(("DDIM median", d_median, "#999999"))

    if not entries:
        raise ValueError(f"No plottable groups found for season='{season}'")

    labels = [e[0] for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    metrics = ["S", "A", "L"]

    for ax, metric in zip(axes, metrics):
        data = [e[1][metric].dropna().values for e in entries]

        bp = ax.boxplot(
            data,
            labels=labels,
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

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"SAL distributions for {season}", y=1.04)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes