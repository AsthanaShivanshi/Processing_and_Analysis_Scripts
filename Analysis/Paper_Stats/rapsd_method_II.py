from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from numpy.fft import fft2
from tqdm import tqdm

def normalize_data(data, mean=None, std=None, norm_mean=0.0, norm_std=1.0):
    """
    Normalizes the input data to have zero mean and the specified standard deviation.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    normalized_data = (data - mean) * norm_std / std + norm_mean
    return normalized_data, mean, std


SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
PROCESSING_ROOT = PAPER_STATS_DIR.parent.parent.parent

START = "2015-01-01"
END = "2023-12-31"

SAMPLE_DIMS = ("member", "sample", "samples")
EPS = 1e-30
GRID_SPACING_KM = 1.0
NYQUIST_KM = 2.0 * GRID_SPACING_KM  # 2 km
PRECIP_MIN_MEAN = 0.048  # mm/day (Harris et al. 0.002 mm/hr * 24)

# Broken x-axis: make 1–12 km prominent, keep the rest to 80 km.
WL_LEFT_MAX_KM = 80.0
WL_BREAK_KM = 12.0
WL_MIN_KM = 1.0

LEFT_TICKS_KM = np.array([50, 40, 30, 20, 12], dtype=float)
RIGHT_TICKS_KM = np.array([12, 10, 8, 6, 4, 2, 1], dtype=float)

MODEL_COLORS = {
    "Coarse": "#7f7f7f",
    "Bicubic": "#1f77b4",
    "Bilinear": "#ff7f0e",
    "UNet": "#2ca02c",
    "DDIM": "#d62728",
    "CFM": "#9467bd",
}


def _load_field(path, var, mask=None, clip=False):
    with xr.open_dataset(path) as ds:
        da = ds[var].sel(time=slice(START, END)).load()
    if clip:
        da = da.clip(min=0)
    return da.where(mask) if mask is not None else da


def _load_mask(path, var):
    with xr.open_dataset(path) as ds:
        return ds[var].load()


def _sample_dim(da):
    return next((d for d in SAMPLE_DIMS if d in da.dims), None)


def _get_2d_spectrum_info(data: np.ndarray, x_length: float, y_length: float):
    _, Ny, Nx = data.shape

    # Multiply by 2*pi to match k = 2*pi/lambda notation
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=x_length / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=y_length / Ny)

    fft_data = np.fft.fft2(data, axes=(1, 2))
    return (Ny, Nx), (kx, ky), fft_data


def _collect_frames(
    da: xr.DataArray,
    valid_times: np.ndarray | None = None,
) -> np.ndarray:
    sd = _sample_dim(da)
    nt = da.sizes.get("time", 1)
    time_indices = valid_times if valid_times is not None else np.arange(nt)

    frames: list[np.ndarray] = []
    skipped = 0

    for t in tqdm(time_indices, desc="RAPSD", leave=False):
        frame_da = da.isel(time=int(t)) if "time" in da.dims else da

        if sd and sd in frame_da.dims:
            for s in range(frame_da.sizes[sd]):
                frame = frame_da.isel({sd: s}).values.astype(float)
                finite = np.isfinite(frame)
                if finite.sum() < 4 or np.nanstd(frame[finite]) < 1e-6:
                    skipped += 1
                    continue
                frame_mean = np.nanmean(frame[finite])
                frame = np.where(finite, frame, frame_mean)
                frames.append(frame)
        else:
            frame = frame_da.values.astype(float)
            finite = np.isfinite(frame)
            if finite.sum() < 4 or np.nanstd(frame[finite]) < 1e-6:
                skipped += 1
                continue
            frame_mean = np.nanmean(frame[finite])
            frame = np.where(finite, frame, frame_mean)
            frames.append(frame)

    if skipped:
        print(f"  [rapsd] skipped {skipped} near-constant/empty frames")
    if not frames:
        raise RuntimeError("No valid frames for RAPSD.")

    return np.stack(frames, axis=0)


def _get_1dpsd(
    data: np.ndarray,
    x_length: float,
    y_length: float,
    data_std: float = 1,
    rotation_angle: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # Normalize data
    data = np.asarray(normalize_data(data, norm_std=data_std)[0])

    # Get 2D spectrum information
    (Ny, Nx), (kx, ky), fft_data = _get_2d_spectrum_info(data, x_length, y_length)
    Nt = data.shape[0]

    # Get radial wavenumber coordinate (from cartesian to polar coordinates)
    ky_grid, kx_grid = np.meshgrid(kx, ky)  # yes it's reversed

    # Apply rotation to the grid
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    kx_grid_rotated = rotation_matrix[0, 0] * kx_grid + rotation_matrix[0, 1] * ky_grid
    ky_grid_rotated = rotation_matrix[1, 0] * kx_grid + rotation_matrix[1, 1] * ky_grid

    # Calculate the new wavenumber grid after rotation
    k_grid = np.sqrt(kx_grid_rotated**2 + ky_grid_rotated**2)

    # Define 1D wavenumber bins
    Nbins = int(np.sqrt(Nx * Ny) / 2)
    Nbins = max(Nbins, 2)
    eps = 1e-5
    k_bins = np.linspace(0, np.max(k_grid) + eps, Nbins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Compute 1D PSD for each time step t
    ps_t = np.zeros([len(k_centers), Nt])
    for t in range(Nt):
        ps_t[:, t] = np.histogram(
            k_grid, bins=k_bins, weights=np.abs(fft_data[t, :, :]) ** 2
        )[0]

    psd_t = ps_t / (Nx * Ny) ** 2
    hist = np.histogram(k_grid, bins=k_bins)[0]
    mask = hist != 0
    psd_t = psd_t[mask, :] / hist[mask, None]
    psd = np.mean(psd_t, axis=1)

    # Return cycles/km so the existing wavelength logic stays unchanged.
    freqs = (k_centers[mask][::-1]) / (2 * np.pi)
    return freqs, psd[::-1]


def rapsd_timemean(
    da: xr.DataArray,
    valid_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frames = _collect_frames(da, valid_times=valid_times)
    x_length = frames.shape[-1] * GRID_SPACING_KM
    y_length = frames.shape[-2] * GRID_SPACING_KM
    freqs, psd = _get_1dpsd(frames, x_length, y_length)
    return psd, freqs


def _get_valid_precip_times(obs: xr.DataArray) -> np.ndarray:
    nt = obs.sizes.get("time", 1)
    valid = []
    for t in range(nt):
        frame = obs.isel(time=t).values.astype(float)
        finite = np.isfinite(frame)
        if finite.sum() > 0 and np.nanmean(frame[finite]) >= PRECIP_MIN_MEAN:
            valid.append(t)
    print(f"  [precip] {len(valid)}/{nt} frames pass low-rain filter")
    return np.array(valid, dtype=int)


def _ralsd(rapsd_pred: np.ndarray, rapsd_ref: np.ndarray) -> float:
    N = len(rapsd_pred)
    d = 10.0 * np.log10(np.maximum(rapsd_pred, EPS) / np.maximum(rapsd_ref, EPS))
    return float(np.sqrt(np.sum(d**2) / N))


STYLE = {
    "Obs": dict(color="black", ls="-", lw=1.5),
    "Coarse": dict(ls="--", lw=0.98),
    "Bicubic": dict(ls="--", lw=0.98),
    "Bilinear": dict(ls="--", lw=0.98),
    "UNet": dict(ls="-", lw=1.0),
    "DDIM": dict(ls="-", lw=1.5),
    "CFM": dict(ls="-", lw=1.5),
}


def _add_x_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    d = 0.012
    kwargs_left = dict(
        transform=ax_left.transAxes, color="0.35", clip_on=False, lw=1.0
    )
    kwargs_right = dict(
        transform=ax_right.transAxes, color="0.35", clip_on=False, lw=1.0
    )

    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)


def _format_broken_pair(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    show_ylabel: bool,
    show_xticklabels: bool,
) -> None:
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    ax_left.tick_params(axis="y", right=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    if not show_xticklabels:
        ax_left.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="x", labelbottom=False)

    if not show_ylabel:
        ax_left.set_ylabel("")


def _plot_variable(
    fig: plt.Figure,
    ax_top_left: plt.Axes,
    ax_top_right: plt.Axes,
    ax_bot_left: plt.Axes,
    ax_bot_right: plt.Axes,
    wl_km: np.ndarray,
    ps_obs: np.ndarray,
    model_spectra: dict[str, tuple[np.ndarray, float]],
    var_label: str,
    psd_units: str,
    show_ylabel: bool = True,
) -> None:
    ralsd_vals = np.array([v[1] for v in model_spectra.values()], dtype=float)
    norm = mcolors.Normalize(
        vmin=float(np.min(ralsd_vals)),
        vmax=float(np.max(ralsd_vals)),
    )
    cmap = plt.cm.viridis

    model_colors = {
        name: MODEL_COLORS.get(name, "C0")
        for name in model_spectra.keys()
    }

    for ax in (ax_top_left, ax_top_right):
        ax.semilogy(wl_km, ps_obs, zorder=5, **STYLE["Obs"])
        for name, (ps_pred, _) in model_spectra.items():
            style = dict(STYLE.get(name, {}))
            style["color"] = model_colors[name]
            ax.semilogy(wl_km, ps_pred, markersize=0, **style)

        ax.grid(True, which="both", alpha=0.3, ls="--")
        ax.set_xlim(
            WL_LEFT_MAX_KM if ax is ax_top_left else WL_BREAK_KM,
            WL_BREAK_KM if ax is ax_top_left else WL_MIN_KM,
        )
        for sc in [1, 2, 5, 10, 12, 20, 50, 80]:
            if wl_km.min() <= sc <= wl_km.max():
                ax.axvline(sc, color="gray", lw=0.8, ls=":", alpha=0.45)

    ax_top_left.set_title(var_label, fontsize=14, fontweight="bold")
    if show_ylabel:
        ax_top_left.set_ylabel(f"Power Spectral Density\n({psd_units})", fontsize=12)

    ax_top_left.tick_params(axis="both", labelsize=11)
    ax_top_right.tick_params(axis="both", labelsize=11)

    all_ratios = []
    for ax in (ax_bot_left, ax_bot_right):
        ax.axhline(1.0, color="black", lw=1.0, ls="-", zorder=3)
        for name, (ps_pred, _) in model_spectra.items():
            ratio = ps_pred / np.maximum(ps_obs, EPS)
            all_ratios.append(ratio)
            style = dict(STYLE.get(name, {}))
            style["color"] = model_colors[name]
            ax.plot(wl_km, ratio, markersize=0, **style)

        ax.grid(True, which="both", alpha=0.3, ls="--")
        ax.set_xlim(
            WL_LEFT_MAX_KM if ax is ax_bot_left else WL_BREAK_KM,
            WL_BREAK_KM if ax is ax_bot_left else WL_MIN_KM,
        )
        for sc in [1, 2, 5, 10, 12, 20, 50]:
            if wl_km.min() <= sc <= wl_km.max():
                ax.axvline(sc, color="gray", lw=0.8, ls=":", alpha=0.45)

    max_ratio = np.nanmax([np.nanmax(r) for r in all_ratios]) if all_ratios else 3.0
    y_max = max(4.0, float(max_ratio) * 1.1)
    ax_bot_left.set_ylim(0, y_max)
    ax_bot_right.set_ylim(0, y_max)

    if show_ylabel:
        ax_bot_left.set_ylabel("Spectral ratio\n(model / obs)  [–]", fontsize=11)

    ax_bot_left.set_xlabel("Wavelength (km)", fontsize=12)
    ax_bot_right.set_xlabel("Wavelength (km)", fontsize=12)
    ax_bot_left.tick_params(axis="both", labelsize=11)
    ax_bot_right.tick_params(axis="both", labelsize=11)

    ax_top_left.set_xticks(LEFT_TICKS_KM)
    ax_top_right.set_xticks(RIGHT_TICKS_KM)
    ax_bot_left.set_xticks(LEFT_TICKS_KM)
    ax_bot_right.set_xticks(RIGHT_TICKS_KM)

    ax_top_left.set_xticklabels([str(int(v)) for v in LEFT_TICKS_KM], fontsize=10)
    ax_top_right.set_xticklabels([str(int(v)) for v in RIGHT_TICKS_KM], fontsize=10)
    ax_bot_left.set_xticklabels([str(int(v)) for v in LEFT_TICKS_KM], fontsize=10)
    ax_bot_right.set_xticklabels([str(int(v)) for v in RIGHT_TICKS_KM], fontsize=10)

    _format_broken_pair(
        ax_top_left, ax_top_right, show_ylabel=show_ylabel, show_xticklabels=False
    )
    _format_broken_pair(
        ax_bot_left, ax_bot_right, show_ylabel=show_ylabel, show_xticklabels=True
    )
    _add_x_break_marks(ax_top_left, ax_top_right)
    _add_x_break_marks(ax_bot_left, ax_bot_right)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=[ax_top_left, ax_top_right, ax_bot_left, ax_bot_right],
        fraction=0.022,
        pad=0.015,
    )
    cbar.set_label("RALSD (dB)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    score_text = "\n".join(
        f"{name}: {ralsd:.2f} dB" for name, (_, ralsd) in model_spectra.items()
    )
    ax_top_right.text(
        0.03,
        0.05,
        score_text,
        transform=ax_top_right.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.82,
            edgecolor="0.8",
        ),
    )
    ax_top_right.text(
        0.03,
        0.98,
        "Zoomed 1–12 km region",
        transform=ax_top_right.transAxes,
        fontsize=9,
        color="0.35",
        va="top",
        ha="left",
    )


def _add_shared_legend(fig: plt.Figure, model_spectra_per_col: list[dict]) -> None:
    """Shared legend at the bottom with model names only."""
    model_names = list(model_spectra_per_col[0].keys())

    handles = []
    labels = []

    handles.append(mlines.Line2D([], [], **STYLE["Obs"]))
    labels.append("Obs (reference)")

    for name in model_names:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=MODEL_COLORS.get(name, "C0"),
                ls=STYLE.get(name, {}).get("ls", "-"),
                lw=STYLE.get(name, {}).get("lw", 1.5),
            )
        )
        labels.append(name)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        fontsize=10,
        framealpha=0.92,
        bbox_to_anchor=(0.5, -0.02),
        title="Model lines",
        title_fontsize=10,
    )


def main() -> None:
    mask_lr = _load_mask(
        PROCESSING_ROOT
        / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc",
        "TabsD",
    )
    mask_hr = _load_mask(
        PROCESSING_ROOT
        / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        "TabsD",
    )

    obs_temp = _load_field(
        PROCESSING_ROOT
        / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc",
        "TabsD",
    )
    obs_precip = _load_field(
        PROCESSING_ROOT
        / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc",
        "RhiresD",
        clip=True,
    )

    coarse_temp = _load_field(
        PROCESSING_ROOT
        / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc",
        "TabsD",
        mask_lr,
    ).interp_like(mask_hr, method="nearest")

    coarse_precip = _load_field(
        PROCESSING_ROOT
        / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc",
        "RhiresD",
        mask_lr,
        clip=True,
    ).interp_like(mask_hr, method="nearest")

    VAR_DEFS = [
        (
            "temp",
            obs_temp,
            "Temperature (K)",
            False,
            "K²  (cycles pixel⁻¹)⁻¹",
            {
                "Coarse": coarse_temp,
                "Bicubic": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc",
                    "TabsD",
                    mask_hr,
                ),
                "Bilinear": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc",
                    "TabsD",
                    mask_hr,
                ),
                "UNet": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
                    "temp",
                    mask_hr,
                ),
                "DDIM": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
                    "temp",
                    mask_hr,
                ),
                "CFM": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
                    "temp",
                    mask_hr,
                ),
            },
        ),
        (
            "precip",
            obs_precip,
            "Precipitation (mm/day)",
            True,
            "(mm day⁻¹)²  (cycles pixel⁻¹)⁻¹",
            {
                "Coarse": coarse_precip,
                "Bicubic": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc",
                    "RhiresD",
                    mask_hr,
                    clip=True,
                ),
                "Bilinear": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc",
                    "RhiresD",
                    mask_hr,
                    clip=True,
                ),
                "UNet": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
                "DDIM": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
                "CFM": _load_field(
                    PROCESSING_ROOT
                    / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
            },
        ),
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.7, 1.0, 1.7],
        height_ratios=[2, 1],
        wspace=0.05,
        hspace=0.18,
    )

    axes = {
        "temp": (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1], sharey=None),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1], sharey=None),
        ),
        "precip": (
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 3], sharey=None),
            fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[1, 3], sharey=None),
        ),
    }

    rows = []
    model_spectra_per_col: list[dict] = []

    for col, (variable, obs, var_label, is_precip, psd_units, models) in enumerate(
        tqdm(VAR_DEFS, desc="variables")
    ):
        print(f"\n=== {variable} ===")

        valid_times = _get_valid_precip_times(obs) if is_precip else None


        rapsd_obs, freqs = rapsd_timemean(obs, valid_times=valid_times)

        positive = freqs > 0
        wavelengths = np.full_like(freqs, np.inf, dtype=float)
        wavelengths[positive] = 1.0 / (freqs[positive] / GRID_SPACING_KM)
        valid = positive & (wavelengths >= NYQUIST_KM)

        k = freqs[valid]
        ps_obs = rapsd_obs[valid]
        wl_km = 1.0 / (k / GRID_SPACING_KM)


        sort_idx = np.argsort(wl_km)
        wl_km = wl_km[sort_idx]
        ps_obs = ps_obs[sort_idx]

        model_spectra: dict[str, tuple[np.ndarray, float]] = {}

        for name, pred in tqdm(models.items(), desc=variable, leave=False):
            rapsd_pred, _ = rapsd_timemean(pred, valid_times=valid_times)
            ps_pred = rapsd_pred[valid][sort_idx]
            ralsd_val = _ralsd(ps_pred, ps_obs)

            print(f"  {name:10s}  RALSD = {ralsd_val:.4f} dB")
            rows.append({"model": name, "variable": variable, "RALSD_mean": ralsd_val})
            model_spectra[name] = (ps_pred, ralsd_val)

        model_spectra_per_col.append(model_spectra)

        ax_tl, ax_tr, ax_bl, ax_br = axes[variable]
        _plot_variable(
            fig=fig,
            ax_top_left=ax_tl,
            ax_top_right=ax_tr,
            ax_bot_left=ax_bl,
            ax_bot_right=ax_br,
            wl_km=wl_km,
            ps_obs=ps_obs,
            model_spectra=model_spectra,
            var_label=var_label,
            psd_units=psd_units,
            show_ylabel=(col == 0),
        )

    _add_shared_legend(fig, model_spectra_per_col)

    fig.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.18)
    (PAPER_STATS_DIR / "Figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PAPER_STATS_DIR / "Figures" / "rapsd_combined_method_II.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved rapsd_combined_II.png")

    df = (
        pd.DataFrame(rows)
        .sort_values(["variable", "model"])
        .reset_index(drop=True)[["model", "variable", "RALSD_mean"]]
    )
    out_csv = PAPER_STATS_DIR / "SR_metrics_rapsd_ralsd_method_II.csv"

    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()