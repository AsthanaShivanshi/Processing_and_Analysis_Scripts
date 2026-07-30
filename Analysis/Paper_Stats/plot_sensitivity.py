import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pathlib import Path

import plotstyle
from plotstyle import apply_paper_style
from pareto import pareto_minimise


def _y_forward(y, y_break=700.0, y_compression=0.35):
    y = np.asarray(y, dtype=float)
    return np.where(y <= y_break, y, y_break + (y - y_break) * y_compression)


def _y_inverse(y, y_break=700.0, y_compression=0.35):
    y = np.asarray(y, dtype=float)
    return np.where(y <= y_break, y, y_break + (y - y_break) / y_compression)


def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _sample_front(front_df, max_labels=10):
    if front_df.empty or len(front_df) <= max_labels:
        return front_df
    idx = np.linspace(0, len(front_df) - 1, max_labels, dtype=int)
    return front_df.iloc[idx]


def plot_sensitivity(
    csv_path,
    metric="CRPS",
    model="DDIM",
    save_path=None,
    unet_time_seconds=0.2,
    dpi=400,
    annotate=True,
    Title=None,
    figsize=(12, 8),
    x_label_fontsize=15,
    y_label_fontsize=15,
    title_fontsize=18,
    tick_fontsize=12,
    legend_fontsize=11,
    y_break=700.0,            # linear up to this value
    y_compression=0.35,       # compression above break (0 < value <= 1)
    max_annotations=10,
):
    apply_paper_style()

    if not (0 < float(y_compression) <= 1):
        raise ValueError("y_compression must be in (0, 1].")

    data = pd.read_csv(csv_path).copy()
    data.columns = data.columns.str.strip()

    metric = metric.strip().upper()
    model = model.strip().upper()
    if model not in {"DDIM", "FM"}:
        raise ValueError("model must be either 'DDIM' or 'FM'.")

    if "denoising_steps" not in data.columns and "num_steps" in data.columns:
        data["denoising_steps"] = data["num_steps"]

    required_common = {"inference_time_mins", "num_samples", "denoising_steps"}

    if metric == "CRPS":
        model_precip = f"CRPS_precip_{model}"
        model_temp = f"CRPS_temp_{model}"
        unet_precip = "CRPS_precip_UNet"
        unet_temp = "CRPS_temp_UNet"

        required_metric = {model_precip, model_temp, unet_precip, unet_temp}
        missing = (required_common | required_metric) - set(data.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        data = _to_numeric(
            data,
            [
                model_precip,
                model_temp,
                unet_precip,
                unet_temp,
                "inference_time_mins",
                "num_samples",
                "denoising_steps",
            ],
        )

        for variable in ("precip", "temp"):
            model_source = f"CRPS_{variable}_{model}"
            unet_source = f"CRPS_{variable}_UNet"

            vmin = np.nanmin(data[model_source].to_numpy(dtype=float))
            vmax = np.nanmax(data[model_source].to_numpy(dtype=float))
            vrange = vmax - vmin
            if not np.isfinite(vrange) or vrange == 0:
                raise ValueError(f"Cannot normalize '{model_source}': invalid or constant values.")

            data[f"{model_source}_plot"] = (data[model_source] - vmin) / vrange
            data[f"{unet_source}_plot"] = (data[unet_source] - vmin) / vrange

        precip_col = f"CRPS_precip_{model}_plot"
        temp_col = f"CRPS_temp_{model}_plot"
        precip_unet_col = "CRPS_precip_UNet_plot"
        temp_unet_col = "CRPS_temp_UNet_plot"

        xlabel = "Normalised CRPS (lower is better)"
        default_save_name = f"Inference_Time_vs_CRPS_{model}.png"

    elif metric == "SSIM":
        model_precip = f"SSIM_precip_{model}_median"
        model_temp = f"SSIM_temp_{model}_median"
        unet_precip = "SSIM_precip_UNet"
        unet_temp = "SSIM_temp_UNet"

        required_metric = {model_precip, model_temp, unet_precip, unet_temp}
        missing = (required_common | required_metric) - set(data.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        data = _to_numeric(
            data,
            [
                model_precip,
                model_temp,
                unet_precip,
                unet_temp,
                "inference_time_mins",
                "num_samples",
                "denoising_steps",
            ],
        )

        data["one_minus_SSIM_precip_model"] = 1 - data[model_precip]
        data["one_minus_SSIM_temp_model"] = 1 - data[model_temp]
        data["one_minus_SSIM_precip_UNet"] = 1 - data[unet_precip]
        data["one_minus_SSIM_temp_UNet"] = 1 - data[unet_temp]

        precip_col = "one_minus_SSIM_precip_model"
        temp_col = "one_minus_SSIM_temp_model"
        precip_unet_col = "one_minus_SSIM_precip_UNet"
        temp_unet_col = "one_minus_SSIM_temp_UNet"

        xlabel = "1 − SSIM (lower is better)"
        default_save_name = f"Inference_Time_vs_SSIM_{model}.png"

    else:
        raise ValueError("metric must be either 'CRPS' or 'SSIM'.")

    time_col = "inference_time_mins"
    plot_data = data.dropna(subset=[precip_col, temp_col, time_col]).copy()
    if plot_data.empty:
        raise ValueError("No valid rows remain after dropping NaNs for plotting.")

    precip_front = pareto_minimise(plot_data, precip_col, time_col).sort_values(precip_col)
    temp_front = pareto_minimise(plot_data, temp_col, time_col).sort_values(temp_col)

    fig, ax = plt.subplots(figsize=figsize)

    # background points (lighter, less clutter)
    ax.scatter(
        plot_data[precip_col],
        plot_data[time_col],
        color=plotstyle.get_variable_color("precip"),
        alpha=0.16,
        s=22,
        label=f"{model} precipitation samples",
    )
    ax.scatter(
        plot_data[temp_col],
        plot_data[time_col],
        color=plotstyle.get_variable_color("temp"),
        alpha=0.16,
        s=22,
        label=f"{model} temperature samples",
    )

    # Pareto fronts
    ax.step(
        precip_front[precip_col],
        precip_front[time_col],
        where="post",
        color=plotstyle.get_variable_color("precip"),
        linewidth=2.6,
        alpha=0.98,
        zorder=20,
        label="Pareto front: precipitation",
    )
    ax.step(
        temp_front[temp_col],
        temp_front[time_col],
        where="post",
        color=plotstyle.get_variable_color("temp"),
        linewidth=2.6,
        alpha=0.98,
        zorder=20,
        label="Pareto front: temperature",
    )

    # U-Net reference points
    precip_unet_values = data[precip_unet_col].dropna()
    temp_unet_values = data[temp_unet_col].dropna()
    if precip_unet_values.empty or temp_unet_values.empty:
        raise ValueError("No valid U-Net metric values were found.")

    x_u_p = float(precip_unet_values.iloc[0])
    x_u_t = float(temp_unet_values.iloc[0])
    y_u = float(unet_time_seconds) / 60.0

    # halo
    ax.scatter(
        [x_u_p, x_u_t],
        [y_u, y_u],
        s=270,
        marker="o",
        color="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=34,
        clip_on=False,
    )
    # stars
    ax.scatter(
        x_u_p,
        y_u,
        color=plotstyle.get_variable_color("precip"),
        marker="*",
        s=190,
        edgecolor="black",
        linewidth=1.0,
        zorder=35,
        clip_on=False,
        label="U-Net precipitation",
    )
    ax.scatter(
        x_u_t,
        y_u,
        color=plotstyle.get_variable_color("temp"),
        marker="*",
        s=190,
        edgecolor="black",
        linewidth=1.0,
        zorder=35,
        clip_on=False,
        label="U-Net temperature",
    )

    if annotate:
        texts = []
        p_lab = _sample_front(precip_front, max_labels=max_annotations)
        t_lab = _sample_front(temp_front, max_labels=max_annotations)

        for _, row in p_lab.iterrows():
            texts.append(
                ax.text(
                    row[precip_col],
                    row[time_col],
                    f"({int(row['num_samples'])}, {int(row['denoising_steps'])})",
                    fontsize=7,
                    color=plotstyle.get_variable_color("precip"),
                )
            )
        for _, row in t_lab.iterrows():
            texts.append(
                ax.text(
                    row[temp_col],
                    row[time_col],
                    f"({int(row['num_samples'])}, {int(row['denoising_steps'])})",
                    fontsize=7,
                    color=plotstyle.get_variable_color("temp"),
                )
            )

        if texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.4},
            )

    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel("Inference time (minutes)", fontsize=y_label_fontsize, fontweight="bold")
    if Title is not None:
        ax.set_title(Title, fontsize=title_fontsize, fontweight="bold")

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # piecewise y transform
    if y_compression < 1:
        ax.set_yscale(
            "function",
            functions=(
                lambda y: _y_forward(y, y_break=y_break, y_compression=y_compression),
                lambda y: _y_inverse(y, y_break=y_break, y_compression=y_compression),
            ),
        )

    # limits + ticks:
    # 0..y_break every 100, above y_break every 200 (compressed region, less clutter)
    y_all = np.concatenate(
        [
            plot_data[time_col].to_numpy(dtype=float),
            np.array([y_u], dtype=float),
        ]
    )
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_max = float(np.nanmax(y_all))
        y_top = max(y_break, np.ceil(y_max / 100.0) * 100.0)
        ax.set_ylim(0.0, y_top)

        low_ticks = np.arange(0.0, y_break + 1e-9, 100.0)
        if y_top > y_break:
            high_start = y_break + 200.0
            high_ticks = np.arange(high_start, y_top + 1e-9, 200.0)
            yticks = np.concatenate([low_ticks, high_ticks])
        else:
            yticks = low_ticks
        ax.set_yticks(yticks)

    ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper right", ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path / default_save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, precip_front, temp_front, data