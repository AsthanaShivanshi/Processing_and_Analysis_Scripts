import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams["pdf.compression"]= 9

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotstyle import apply_paper_style, get_variable_cmap

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")


#TBR for SR set ., MCH. 
#To be run as : python Analysis/Paper_Stats/plot_SR_frames.py --variable pr --indices 580  1050 3010 or 
####
# python Analysis/Paper_Stats/plot_SR_frames.py --variable tas --indices 580  1050 3010



TIME_SLICE = slice("2015-01-01", "2023-12-31") #Test set

DATA_SOURCES_PRECIP = {
    "Observations": {"file": "data_1971_2023/HR_files_full/RhiresD_1971_2023.nc", "var": "RhiresD", "sample": None},
    "Coarse": {"file": "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc", "var": "RhiresD", "sample": None},
    "Bilinear": {"file": "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc", "var": "RhiresD", "sample": None},
    "UNet": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc", "var": "precip", "sample": None},
    "DDIM_Sample_2": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "precip", "sample": 1},
    "DDIM_Sample_5": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "precip", "sample": 4},
    "DDIM_Sample_10": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "precip", "sample": 9},
}



DATA_SOURCES_TEMP = {
    "Observations": {"file": "data_1971_2023/HR_files_full/TabsD_1971_2023.nc", "var": "TabsD", "sample": None},
    "Coarse": {"file": "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc", "var": "TabsD", "sample": None},
    "Bilinear": {"file": "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc", "var": "TabsD", "sample": None},
    "UNet": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc", "var": "temp", "sample": None},
    "DDIM_Sample_1": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "temp", "sample": 1},
    "DDIM_Sample_2": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "temp", "sample": 2},
    "DDIM_Sample_10": {"file": "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "var": "temp", "sample": 9},
}



VARIABLE_CONFIG = {
    "pr": {
        "name": "Precipitation",
        "cmap": get_variable_cmap("precip"),
        "vmin": 0,
        "vmax": 20,
        "unit": r"mm day$^{-1}$",
    },
    "tas": {
        "name": "Temperature",
        "cmap": get_variable_cmap("temp"),
        "vmin": -20,
        "vmax": 30,
        "unit": "°C",
    },
}




def load_mask(path: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if "TabsD" in ds.data_vars:
            return ds["TabsD"].load()
        first_var = next(iter(ds.data_vars))
        return ds[first_var].load()


def get_data_sources(variable: str):
    if variable == "pr":
        return DATA_SOURCES_PRECIP
    if variable == "tas":
        return DATA_SOURCES_TEMP
    raise ValueError(f"Unknown variable: {variable}")


def get_date_labels(data_sources, indices):
    obs = data_sources.get("Observations")
    if obs is None or not Path(obs["file"]).exists():
        return [f"idx={i}" for i in indices]

    with xr.open_dataset(obs["file"]) as ds:
        da = ds[obs["var"]].sel(time=TIME_SLICE)
        labels = []
        for i in indices:
            if 0 <= i < da.sizes["time"]:
                labels.append(np.datetime_as_string(da.time.isel(time=i).values, unit="D"))
            else:
                labels.append(f"idx={i}")
    return labels


def load_frame(source_name, source_cfg, idx, mask_hr, mask_lr):
    file_path = source_cfg["file"]
    var_name = source_cfg["var"]
    sample_idx = source_cfg["sample"]

    if not Path(file_path).exists():
        print(f"[warn] File not found: {file_path}")
        return None, None, None

    with xr.open_dataset(file_path) as ds:
        da = ds[var_name].sel(time=TIME_SLICE).astype("float32")
        n_time = da.sizes["time"]
        if idx < 0 or idx >= n_time:
            print(f"[warn] Index {idx} out of bounds for {source_name} (length: {n_time})")
            return None, None, None
        frame = da.isel(time=idx).values

    if frame.ndim == 3:
        frame = frame[sample_idx] if sample_idx is not None else np.nanmean(frame, axis=0)

    mask = mask_lr if source_name == "Coarse" else mask_hr
    frame = np.ma.masked_invalid(np.where(mask.values, frame, np.nan))
    return frame, mask["lat"].values, mask["lon"].values


def style_map_axis(ax):
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([5.8, 10.6, 45.7, 47.9], crs=ccrs.PlateCarree())
    ax.axis("off")


def plot_sr_frames(variable, indices, output_dir="Analysis/Paper_Stats/Figures"):
    if len(indices) != 3:
        raise ValueError("Must provide exactly 3 indices")

    apply_paper_style()

    data_sources = get_data_sources(variable)
    var_cfg = VARIABLE_CONFIG[variable]

    mask_hr = load_mask("../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc")
    mask_lr = load_mask("../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc")

    row_names = [k for k in data_sources if k != "Observations"] + ["Observations"]
    row_display_names = {
        "Observations": "Ground truth (MCH test set)",
    }
    date_labels = get_date_labels(data_sources, indices)

    n_rows, n_cols = len(row_names), len(indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.2 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
        facecolor="white",
    )

    mesh = None
    for i, source_name in enumerate(row_names):
        for j, idx in enumerate(indices):
            ax = axes[i, j]
            frame, lat, lon = load_frame(source_name, data_sources[source_name], idx, mask_hr, mask_lr)

            if frame is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="red")
                ax.axis("off")
                continue

            mesh = ax.pcolormesh(
                lon,
                lat,
                frame,
                cmap=var_cfg["cmap"],
                vmin=var_cfg["vmin"],
                vmax=var_cfg["vmax"],
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True
            )
            style_map_axis(ax)

            if i == 0:
                ax.set_title(date_labels[j], pad=6)
            if j == 0:
                label = row_display_names.get(source_name, source_name)
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right")



    if mesh is not None:
        cbar = fig.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.06,
            aspect=50,
        )
        cbar.set_label(var_cfg["unit"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Samples_SR_{variable}_{'-'.join(map(str, indices))}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)




def main():
    parser = argparse.ArgumentParser(description="Plot SR frames for multiple baselines and time indices")
    parser.add_argument("--variable", type=str, required=True, choices=["pr", "tas"], help="Variable: pr or tas")
    parser.add_argument("--indices", type=int, nargs=3, required=True, metavar=("IDX1", "IDX2", "IDX3"), help="Three time indices")
    parser.add_argument("--output_dir", type=str, default="Analysis/Paper_Stats/Figures", help="Output directory")
    args = parser.parse_args()

    plot_sr_frames(args.variable, args.indices, args.output_dir)


if __name__ == "__main__":
    main()



