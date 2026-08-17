from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plotstyle import add_bottom_legend, save_paper_figure, style_model_line

sns.set_style("whitegrid")

TEMP_METRICS = (
    ("CRPS", "CRPS ↓"),
    ("SSIM", "1-SSIM ↓"),
    ("RMSE", "RMSE ↓"),
    ("MAE", "MAE ↓"),
    ("PITD", "PITD ↓"),
)

PRECIP_METRICS = TEMP_METRICS + (("FSS", "1-FSS ↓"),)

MODEL_STYLE_OVERRIDES = {
    "DDIM": {"color": "black", "linewidth": 3.2},
    "CFM": {"color": "green", "linewidth": 2.4},
}


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> str:
    lookup = {col.strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError(f"Could not find any of {candidates}. Available columns: {df.columns.tolist()}")


def _metric_value(row: pd.Series, metric_name: str) -> float:
    candidates = (metric_name, f"{metric_name}_mean", f"{metric_name}_avg")
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                return float(v)
    return np.nan


def _extract_variable_data(
    df: pd.DataFrame,
    variable: str,
    models: tuple[str, ...] | list[str],
    metric_specs: tuple[tuple[str, str], ...],
    model_col: str,
    variable_col: str,
) -> np.ndarray:
    mask = (
        df[variable_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains(variable.lower(), na=False)
    )
    subset = df.loc[mask].copy()
    subset["_model_key"] = subset[model_col].astype(str).str.strip().str.lower()
    subset = subset.set_index("_model_key").reindex([m.lower() for m in models])

    mean_values = []

    for metric_name, _ in metric_specs:
        mean = np.array([_metric_value(row, metric_name) for _, row in subset.iterrows()], dtype=float)

        if metric_name in ("SSIM", "FSS"):
            mean = 1.0 - mean

        mean_values.append(mean)

    return np.asarray(mean_values, dtype=float).T


def _normalise_metric_data(mean_data: np.ndarray) -> np.ndarray:
    norm_mean = np.full_like(mean_data, np.nan, dtype=float)

    for j in range(mean_data.shape[1]):
        vals = mean_data[:, j].ravel()
        vals = vals[np.isfinite(vals)]
        scale = float(np.max(vals)) if vals.size else 1.0
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        norm_mean[:, j] = mean_data[:, j] / scale

    return norm_mean


def _style_for_model(model: str) -> dict:
    base = dict(style_model_line(model))
    base.pop("zorder", None)
    base.update(MODEL_STYLE_OVERRIDES.get(model, {}))
    return base


def _plot_kiviat_axis(ax: plt.Axes, mean_data: np.ndarray, models, metric_labels):
    n_metrics = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    for i, model in enumerate(models):
        line_kwargs = _style_for_model(model)
        values = mean_data[i].tolist()
        closed_values = values + values[:1]
        ax.plot(closed_angles, closed_values, label=model, zorder=3, **line_kwargs)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_rlabel_position(90)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", pad=18)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=1, alpha=0.65)


def plot_kiviat_from_csv(
    csv_path: str | Path,
    save_name: str | Path | None = None,
    models: tuple[str, ...] = ("Coarse", "Bicubic", "Bilinear", "UNet", "DDIM", "CFM"),
    temperature_name: str = "temp",
    precipitation_name: str = "precip",
):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    model_col = _find_column(df, ("model", "models", "method", "methods"))
    variable_col = _find_column(df, ("variable", "var", "variables"))

    temp_mean = _extract_variable_data(df, temperature_name, models, TEMP_METRICS, model_col, variable_col)
    pr_mean = _extract_variable_data(df, precipitation_name, models, PRECIP_METRICS, model_col, variable_col)

    temp_mean = _normalise_metric_data(temp_mean)
    pr_mean = _normalise_metric_data(pr_mean)

    temp_labels = [label for _, label in TEMP_METRICS]
    precip_labels = [label for _, label in PRECIP_METRICS]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7.5),
        subplot_kw={"polar": True},
    )

    _plot_kiviat_axis(axes[0], temp_mean, models, temp_labels)
    _plot_kiviat_axis(axes[1], pr_mean, models, precip_labels)

    fig.subplots_adjust(wspace=0.35, bottom=0.18, top=0.92)

    handles, labels = axes[0].get_legend_handles_labels()
    add_bottom_legend(fig, handles, labels, ncol=len(models))
    if fig.legends:
        fig.legends[-1].set_bbox_to_anchor((0.5, 0.06), transform=fig.transFigure)

    if save_name is not None:
        save_paper_figure(fig, save_name)

    return fig, axes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--temp-name", default="temp")
    ap.add_argument("--precip-name", default="precip")
    args = ap.parse_args()

    plot_kiviat_from_csv(
        csv_path=args.csv,
        save_name=args.out,
        temperature_name=args.temp_name,
        precipitation_name=args.precip_name,
    )


if __name__ == "__main__":
    main()