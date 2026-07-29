import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pathlib import Path

import plotstyle
from plotstyle import apply_paper_style
from pareto import pareto_minimise


def plot_sensitivity(
    csv_path,
    metric="CRPS",
    model="DDIM",
    save_path=None,
    unet_time_seconds=0.2,
    title_year=2012,
    dpi=1500,
    annotate=True,
    Title=None,
    figsize=(12, 8),
    x_label_fontsize=15,
    y_label_fontsize=15,
    title_fontsize=18,
    suptitle_fontsize=20,
    tick_fontsize=12,
    legend_fontsize=12,
):
    apply_paper_style()

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

        for variable in ("precip", "temp"):
            model_source = f"CRPS_{variable}_{model}"
            unet_source = f"CRPS_{variable}_UNet"

            minimum = data[model_source].min()
            maximum = data[model_source].max()
            value_range = maximum - minimum

            if value_range == 0:
                raise ValueError(f"Cannot normalize '{model_source}': all values are identical.")

            data[f"{model_source}_plot"] = (data[model_source] - minimum) / value_range
            data[f"{unet_source}_plot"] = (data[unet_source] - minimum) / value_range

        precip_col = f"CRPS_precip_{model}_plot"
        temp_col = f"CRPS_temp_{model}_plot"
        precip_unet_col = "CRPS_precip_UNet_plot"
        temp_unet_col = "CRPS_temp_UNet_plot"

        xlabel = "Normalised CRPS"
        plot_title = f"Inference time–CRPS trade-off for {model} ensembles"
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

        data["one_minus_SSIM_precip_model"] = 1 - data[model_precip]
        data["one_minus_SSIM_temp_model"] = 1 - data[model_temp]
        data["one_minus_SSIM_precip_UNet"] = 1 - data[unet_precip]
        data["one_minus_SSIM_temp_UNet"] = 1 - data[unet_temp]

        precip_col = "one_minus_SSIM_precip_model"
        temp_col = "one_minus_SSIM_temp_model"
        precip_unet_col = "one_minus_SSIM_precip_UNet"
        temp_unet_col = "one_minus_SSIM_temp_UNet"

        xlabel = "1 − SSIM"
        plot_title = f"Inference time–SSIM trade-off for {model} ensembles"
        default_save_name = f"Inference_Time_vs_SSIM_{model}.png"

    else:
        raise ValueError("metric must be either 'CRPS' or 'SSIM'.")

    time_col = "inference_time_mins"
    plot_data = data.dropna(subset=[precip_col, temp_col, time_col]).copy()

    precip_front = pareto_minimise(plot_data, precip_col, time_col).sort_values(precip_col)
    temp_front = pareto_minimise(plot_data, temp_col, time_col).sort_values(temp_col)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        plot_data[precip_col],
        plot_data[time_col],
        color=plotstyle.get_variable_color("precip"),
        alpha=0.28,
        s=34,
        label=f"{model} precipitation",
    )
    ax.scatter(
        plot_data[temp_col],
        plot_data[time_col],
        color=plotstyle.get_variable_color("temp"),
        alpha=0.28,
        s=34,
        label=f"{model} temperature",
    )

    ax.step(
        precip_front[precip_col],
        precip_front[time_col],
        where="post",
        color=plotstyle.get_variable_color("precip"),
        linewidth=2.5,
        alpha=0.95,
        zorder=20,
        label="Pareto front: precipitation",
    )
    ax.step(
        temp_front[temp_col],
        temp_front[time_col],
        where="post",
        color=plotstyle.get_variable_color("temp"),
        linewidth=2.5,
        alpha=0.95,
        zorder=20,
        label="Pareto front: temperature",
    )

    precip_unet_values = data[precip_unet_col].dropna()
    temp_unet_values = data[temp_unet_col].dropna()
    if precip_unet_values.empty or temp_unet_values.empty:
        raise ValueError("No valid U-Net metric values were found.")

    unet_time_minutes = unet_time_seconds / 60.0

    ax.scatter(
        precip_unet_values.iloc[0],
        unet_time_minutes,
        color="#9ecae1",
        s=95,
        edgecolor="black",
        linewidth=0.8,
        zorder=30,
        label="U-Net precipitation",
    )
    ax.scatter(
        temp_unet_values.iloc[0],
        unet_time_minutes,
        color="#f4a6a6",
        s=95,
        edgecolor="black",
        linewidth=0.8,
        zorder=30,
        label="U-Net temperature",
    )

    if annotate:
        texts = []
        for _, row in precip_front.iterrows():
            texts.append(
                ax.text(
                    row[precip_col],
                    row[time_col],
                    f"({int(row['num_samples'])}, {int(row['denoising_steps'])})",
                    fontsize=8,
                    color=plotstyle.get_variable_color("precip"),
                )
            )
        for _, row in temp_front.iterrows():
            texts.append(
                ax.text(
                    row[temp_col],
                    row[time_col],
                    f"({int(row['num_samples'])}, {int(row['denoising_steps'])})",
                    fontsize=8,
                    color=plotstyle.get_variable_color("temp"),
                )
            )
        adjust_text(
            texts,
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.5},
        )

    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel("Inference time (minutes)", fontsize=y_label_fontsize, fontweight="bold")
    ax.set_title(Title or plot_title, fontsize=title_fontsize, fontweight="bold", pad=10)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path / default_save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, precip_front, temp_front, data