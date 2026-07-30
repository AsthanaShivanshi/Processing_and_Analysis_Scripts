import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plotstyle import apply_paper_style, get_model_color, save_paper_figure

_METRIC_LABELS = {
    "S": "Structure",
    "A": "Amplitude",
    "L": "Location",
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


def _resolve_label(model_norm: str, type_norm: str, model_raw: str) -> str:
    key = (model_norm, type_norm)
    if key in _ENSEMBLE_LABELS:
        return _ENSEMBLE_LABELS[key]
    return str(model_raw).strip()


def _resolve_color(model_norm: str, type_norm: str, label: str) -> str:
    key = (model_norm, type_norm)
    if key in _ENSEMBLE_COLORS:
        return _ENSEMBLE_COLORS[key]

    l = label.strip().lower()
    if l == "bilinear":
        return get_model_color("Bilinear")
    if l == "bicubic":
        return get_model_color("Bicubic")
    if l == "unet":
        return get_model_color("UNet")
    return "#808080"


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
    figsize=(14, 4.8),
    dpi=1500,
    x_tick_fontsize=12,
    y_tick_fontsize=12,
    y_label_fontsize=14,
    suptitle_fontsize=20,
    x_tick_rotation=30,
    plot_kind="violin",
    x_spacing=1.25,
    combine_mam_jja=False,
):
    apply_paper_style()

    df = pd.read_csv(csv_path).copy()
    df["season"] = df["season"].astype(str).str.strip().str.upper()
    df["model_norm"] = df["model"].astype(str).str.strip().str.lower()
    df["type_norm"] = df["type"].astype(str).str.strip().str.lower()

    if combine_mam_jja:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df[df["time"].dt.month.isin([3, 4, 5, 6, 7, 8])].copy()
            season_label = "Mar-Aug"
        else:
            df = df[df["season"].isin(["MAM", "JJA"])].copy()
            season_label = "MAM+JJA"
    else:
        season = str(season).strip().upper()
        df = df[df["season"] == season].copy()
        season_label = season

    if df.empty:
        raise ValueError(f"No rows found for requested period in {csv_path}")

    requested = [str(m).strip() for m in models]
    requested_norm = {m.lower() for m in requested}

    entries = []
    grouped = df.groupby(["model_norm", "type_norm", "model"], dropna=False)

    for (model_norm, type_norm, model_raw), g in grouped:
        label = _resolve_label(model_norm, type_norm, model_raw)
        if label.lower() not in requested_norm:
            continue

        vals = {}
        valid = True
        for metric in ("S", "A", "L"):
            arr = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                valid = False
                break
            vals[metric] = arr

        if not valid:
            continue

        entries.append(
            {
                "label": label,
                "color": _resolve_color(model_norm, type_norm, label),
                "S": vals["S"],
                "A": vals["A"],
                "L": vals["L"],
            }
        )

    order = {m.lower(): i for i, m in enumerate(requested)}
    entries.sort(key=lambda e: order.get(e["label"].lower(), 10_000))

    if not entries:
        raise ValueError("No model entries left after filtering.")

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharex=True)
    x = np.arange(len(entries)) * x_spacing
    labels = [e["label"] for e in entries]

    for ax, metric in zip(axes, ("S", "A", "L")):
        for i, e in enumerate(entries):
            y = e[metric]
            color = e["color"]

            if plot_kind == "box":
                bp = ax.boxplot(
                    [y],
                    positions=[x[i]],
                    widths=0.75,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.85)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(0.8)
                for w in bp["whiskers"] + bp["caps"]:
                    w.set_color("black")
                    w.set_linewidth(0.8)
            else:
                vp = ax.violinplot(
                    [y],
                    positions=[x[i]],
                    widths=0.9,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for b in vp["bodies"]:
                    b.set_facecolor(color)
                    b.set_edgecolor("black")
                    b.set_alpha(0.8)
                    b.set_linewidth(0.6)

                q1, med, q3 = np.percentile(y, [25, 50, 75])
                ax.vlines(x[i], q1, q3, color="black", linewidth=1.3, zorder=3)
                ax.scatter(x[i], med, color="black", s=14, zorder=4)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(_METRIC_LABELS[metric], fontsize=y_label_fontsize)
        ax.set_ylabel(metric, fontsize=y_label_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=x_tick_rotation, ha="right", fontsize=x_tick_fontsize)
        ax.tick_params(axis="y", labelsize=y_tick_fontsize)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    if save_path is not None:
        save_paper_figure(fig, save_path)

    return fig, axes