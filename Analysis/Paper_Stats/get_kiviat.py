import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotstyle import style_model_line, style_model_fill, style_highlight_scatter
from plotstyle import add_bottom_legend, save_paper_figure

DEFAULT_METRICS = (
    ("CRPS", "CRPS ↓"),
    ("LSD", "LSD (median) ↓"),
    ("SSIM", "1-SSIM (median) ↓"),
    ("RMSE", "RMSE (median) ↓"),
    ("MAE", "MAE (median) ↓"),
)


def _find_column(df, candidates):
    columns = {col.strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {df.columns.tolist()}"
    )


def _find_metric_column(df, metric):


    metric = metric.lower()

    for col in df.columns:
        clean_col = (
            col.replace("↓", "")
            .replace("↑", "")
            .strip()
            .lower()
        )
        if clean_col == metric:
            return col

    for col in df.columns:
        if metric in col.lower():
            return col

    raise ValueError(
        f"Could not find metric '{metric}'. "
        f"Available columns: {df.columns.tolist()}"
    )


def _extract_variable_data(
    df,
    variable,
    models,
    metric_specs,
    model_col,
    variable_col,
):
    variable_mask = (
        df[variable_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains(variable.lower(), na=False)
    )

    subset = df.loc[variable_mask].copy()

    if subset.empty:
        raise ValueError(f"No rows found for variable '{variable}'.")

    # Use lowercase model names internally to avoid capitalization problems.
    subset["_model_key"] = (
        subset[model_col].astype(str).str.strip().str.lower()
    )
    model_keys = [model.lower() for model in models]
    subset = subset.set_index("_model_key").reindex(model_keys)

    missing_models = [
        model
        for model, key in zip(models, model_keys)
        if key not in df[model_col].astype(str).str.strip().str.lower().values
    ]
    if missing_models:
        raise ValueError(
            f"Missing models for '{variable}': {missing_models}"
        )

    metric_values = []

    for metric_name, _ in metric_specs:
        metric_col = _find_metric_column(df, metric_name)
        values = pd.to_numeric(subset[metric_col], errors="coerce").to_numpy()

        # Convert SSIM so every radial axis means "lower is better".
        if metric_name.upper() == "SSIM":
            values = 1 - values

        metric_values.append(values)

    data = np.asarray(metric_values, dtype=float).T

    if np.isnan(data).any():
        raise ValueError(
            f"Missing or non-numeric metric values for '{variable}'."
        )

    return data


def _normalise_metrics(data):

    maxima = np.nanmax(data, axis=0)

    return np.divide(
        data,
        maxima,
        out=np.zeros_like(data, dtype=float),
        where=maxima != 0,
    )


def _plot_kiviat_axis(ax, data, title, models, metric_labels):

    number_of_metrics = len(metric_labels)

    angles = np.linspace(
        0,
        2 * np.pi,
        number_of_metrics,
        endpoint=False,
    ).tolist()
    closed_angles = angles + angles[:1]

    for index, model in enumerate(models):
        values = data[index].tolist()
        closed_values = values + values[:1]

        ax.plot(
            closed_angles,
            closed_values,
            label=model,
            **style_model_line(model),
        )

        if model == "DDIM":
            ax.fill(
                closed_angles,
                closed_values,
                **style_model_fill(model),
            )
            ax.scatter(
                closed_angles,
                closed_values,
                **style_highlight_scatter(model),
            )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_rlabel_position(90)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        metric_labels,
        fontsize=12,
        fontweight="bold",
    )

    ax.tick_params(axis="x", pad=20)
    ax.grid(True, linestyle="--", linewidth=1, alpha=0.65)
    ax.set_title(title, fontsize=18, fontweight="bold", y=1.15)


def plot_kiviat_from_csv(
    csv_path,
    save_name=None,
    models=("Coarse", "Bicubic", "Bilinear", "UNet", "DDIM"),
    metric_specs=DEFAULT_METRICS,
    temperature_name="temp",
    precipitation_name="precip",
    Title=None,
):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    model_col = _find_column(
        df,
        ["model", "models", "method", "methods"],
    )
    variable_col = _find_column(
        df,
        ["var", "variable", "variables"],
    )

    temperature = _extract_variable_data(
        df,
        temperature_name,
        models,
        metric_specs,
        model_col,
        variable_col,
    )
    precipitation = _extract_variable_data(
        df,
        precipitation_name,
        models,
        metric_specs,
        model_col,
        variable_col,
    )

    temperature = _normalise_metrics(temperature)
    precipitation = _normalise_metrics(precipitation)

    metric_labels = [label for _, label in metric_specs]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 8),
        subplot_kw={"polar": True},
    )

    _plot_kiviat_axis(
        axes[0],
        temperature,
        "Temperature",
        models,
        metric_labels,
    )
    _plot_kiviat_axis(
        axes[1],
        precipitation,
        "Precipitation",
        models,
        metric_labels,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    add_bottom_legend(fig, handles, labels, ncol=len(models))

    fig.suptitle(
        Title,
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 0.96])

    if save_name is not None:
        save_paper_figure(fig, save_name)

    return fig, axes