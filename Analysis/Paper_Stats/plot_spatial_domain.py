from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotstyle import (
    apply_paper_style,
    get_variable_cmap,
    save_paper_figure,
)


def _prepare_spatial_field(field, name):
    if not isinstance(field, xr.DataArray):
        raise TypeError(
            f"{name} must be an xarray.DataArray, "
            f"not {type(field).__name__}."
        )

    field = field.squeeze(drop=True)

    if field.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional after squeezing. "
            f"Received dimensions {field.dims} with shape {field.shape}."
        )

    return field


def _calculate_extent(fields, lon_name, lat_name):


    lon_min = min(float(field[lon_name].min()) for field in fields)
    lon_max = max(float(field[lon_name].max()) for field in fields)
    lat_min = min(float(field[lat_name].min()) for field in fields)
    lat_max = max(float(field[lat_name].max()) for field in fields)

    return [lon_min, lon_max, lat_min, lat_max]


def _add_map_features(
    ax,
    coastline=True,
    borders=True,
    gridlines=True,
    feature_resolution="10m",
):
    if coastline:
        ax.add_feature(
            cfeature.COASTLINE.with_scale(feature_resolution),
            linewidth=0.8,
            edgecolor="0.25",
            zorder=20,
        )

    if borders:
        ax.add_feature(
            cfeature.BORDERS.with_scale(feature_resolution),
            linestyle=":",
            linewidth=1.0,
            edgecolor="0.25",
            zorder=20,
        )

    if gridlines:
        gridliner = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linestyle="--",
            linewidth=1.0,
            color="0.45",
            alpha=0.65,
            zorder=10,
        )
        gridliner.top_labels = False
        gridliner.right_labels = False


def plot_spatial_domain(
    precipitation,
    temperature,
    *,
    date=None,
    save_path=None,
    lon_name="lon",
    lat_name="lat",
    extent=None,
    precipitation_cmap=None,
    temperature_cmap=None,
    precipitation_label="Precipitation (mm/day)",
    temperature_label="Temperature (°C)",
    precipitation_title=None,
    temperature_title=None,
    figure_title=None,
    figsize=(18, 8),
    projection=None,
    data_crs=None,
    coastline=True,
    borders=True,
    gridlines=True,
    feature_resolution="10m",
    show=False,
):
    apply_paper_style()

    precipitation = _prepare_spatial_field(
        precipitation,
        "precipitation",
    )
    temperature = _prepare_spatial_field(
        temperature,
        "temperature",
    )

    for field_name, field in (
        ("precipitation", precipitation),
        ("temperature", temperature),
    ):
        missing_coordinates = {
            lon_name,
            lat_name,
        } - set(field.coords)

        if missing_coordinates:
            raise ValueError(
                f"{field_name} is missing coordinates "
                f"{sorted(missing_coordinates)}."
            )

    if projection is None:
        projection = ccrs.PlateCarree()

    if data_crs is None:
        data_crs = ccrs.PlateCarree()

    if precipitation_cmap is None:
        precipitation_cmap = get_variable_cmap("precip")

    if temperature_cmap is None:
        temperature_cmap = get_variable_cmap("temp")

    if extent is None:
        extent = _calculate_extent(
            [precipitation, temperature],
            lon_name,
            lat_name,
        )

    if len(extent) != 4 or not np.all(np.isfinite(extent)):
        raise ValueError(
            "extent must contain four finite values: "
            "[lon_min, lon_max, lat_min, lat_max]."
        )

    date_suffix = f" on {date}" if date is not None else ""

    if precipitation_title is None:
        precipitation_title = f"Daily precipitation{date_suffix}"

    if temperature_title is None:
        temperature_title = f"Daily temperature{date_suffix}"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    precip_plot = precipitation.plot(
        ax=axes[0],
        x=lon_name,
        y=lat_name,
        transform=data_crs,
        cmap=precipitation_cmap,
        extend="max",
        add_colorbar=True,
        cbar_kwargs={
            "label": precipitation_label,
            "orientation": "horizontal",
            "pad": 0.08,
            "shrink": 0.85,
            "aspect": 35,
        },
    )

    temperature_plot = temperature.plot(
        ax=axes[1],
        x=lon_name,
        y=lat_name,
        transform=data_crs,
        cmap=temperature_cmap,
        add_colorbar=True,
        cbar_kwargs={
            "label": temperature_label,
            "orientation": "horizontal",
            "pad": 0.08,
            "shrink": 0.85,
            "aspect": 35,
        },
    )

    for ax in axes:
        ax.set_extent(extent, crs=data_crs)
        _add_map_features(
            ax,
            coastline=coastline,
            borders=borders,
            gridlines=gridlines,
            feature_resolution=feature_resolution,
        )

    axes[0].set_title(precipitation_title)
    axes[1].set_title(temperature_title)

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=20, fontweight="bold")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_paper_figure(fig, save_path)

    if show:
        plt.show()

    return fig, axes