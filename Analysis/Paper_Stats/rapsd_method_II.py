from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
PROCESSING_ROOT = PAPER_STATS_DIR.parent.parent.parent

START = "2015-01-01"
END = "2023-12-31"

SAMPLE_DIMS = ("member", "sample", "samples")
EPS = 1e-30
GRID_SPACING_KM = 1.0
NYQUIST_KM = 2.0 * GRID_SPACING_KM
PRECIP_MIN_MEAN = 0.05  # mm/day (Harris et al. 0.002 mm/hr * 24)

# Memory knob: number of 2-D frames FFT'd at once. Lower => less RAM.
CHUNK_FRAMES = 32

# Broken x-axis: continuous 50 -> 12 km, zoom 12 -> 1 km.
WL_LEFT_MAX_KM = 50.0
WL_BREAK_KM = 12.0
WL_MIN_KM = 1.0

LEFT_TICKS_KM = np.array([50, 40, 30, 20, 12], dtype=float)
RIGHT_TICKS_KM = np.array([12, 8, 6, 4, 3, 2, 1], dtype=float)

RATIO_YLIM = (0.0, 2.0)

MODEL_COLORS = {
    "Coarse": "#7f7f7f",
    "Bicubic": "#1f77b4",
    "Bilinear": "#ff7f0e",
    "UNet": "#2ca02c",
    "DDIM": "#d62728",
    "CFM": "#9467bd",
}

STYLE = {
    "Obs": dict(color="black", ls="-", lw=1.8),
    "Coarse": dict(ls="--", lw=1.0),
    "Bicubic": dict(ls="--", lw=1.0),
    "Bilinear": dict(ls="--", lw=1.0),
    "UNet": dict(ls="-", lw=1.2),
    "DDIM": dict(ls="-", lw=1.5),
    "CFM": dict(ls="-", lw=1.5),
}


# ── IO helpers ────────────────────────────────────────────────────────────────
def _load_field(path, var, mask=None, clip=False):
    """Lazy: data stays on disk and is streamed one timestep at a time."""
    ds = xr.open_dataset(path, chunks={"time": 16})
    da = ds[var].sel(time=slice(START, END))
    if clip:
        da = da.clip(min=0)
    return da.where(mask) if mask is not None else da


def _load_mask(path, var):
    with xr.open_dataset(path) as ds:
        return ds[var].load()


def _sample_dim(da):
    return next((d for d in SAMPLE_DIMS if d in da.dims), None)


# ── Streaming RAPSD ───────────────────────────────────────────────────────────
def _iter_frames(da: xr.DataArray, valid_times=None):
    """Yield cleaned 2-D frames one at a time; never stacks the full record."""
    sd = _sample_dim(da)
    nt = da.sizes.get("time", 1)
    time_indices = valid_times if valid_times is not None else np.arange(nt)
    skipped = 0

    for t in tqdm(time_indices, desc="RAPSD", leave=False):
        frame_da = da.isel(time=int(t)) if "time" in da.dims else da
        has_samples = bool(sd and sd in frame_da.dims)
        n_s = frame_da.sizes[sd] if has_samples else 1

        for s in range(n_s):
            sub = frame_da.isel({sd: s}) if has_samples else frame_da
            frame = np.asarray(sub.values, dtype=np.float32)
            finite = np.isfinite(frame)
            if finite.sum() < 4 or np.nanstd(frame[finite]) < 1e-6:
                skipped += 1
                continue
            yield np.where(finite, frame, np.nanmean(frame[finite]))

    if skipped:
        print(f"  [rapsd] skipped {skipped} near-constant/empty frames")


def _radial_binning(Ny, Nx, x_length, y_length, rotation_angle=0.0):
    """Precompute the radial wavenumber bin index for every pixel (once)."""
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=x_length / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=y_length / Ny)
    ky_grid, kx_grid = np.meshgrid(kx, ky)  # yes, reversed

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    k_grid = np.sqrt((c * kx_grid - s * ky_grid) ** 2 + (s * kx_grid + c * ky_grid) ** 2)

    Nbins = max(int(np.sqrt(Nx * Ny) / 2), 2)
    k_bins = np.linspace(0, np.max(k_grid) + 1e-5, Nbins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    bin_idx = np.clip(np.digitize(k_grid.ravel(), k_bins) - 1, 0, len(k_centers) - 1)
    counts = np.bincount(bin_idx, minlength=len(k_centers))
    return k_centers, bin_idx, counts


def rapsd_timemean(
    da: xr.DataArray,
    valid_times: np.ndarray | None = None,
    data_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-mean radially averaged PSD computed in a streaming fashion.

    Peak memory is O(CHUNK_FRAMES) instead of O(n_frames). The global
    normalisation is applied at the end as a scalar, which is exact except for
    the k=0 bin (dropped later by the freqs > 0 filter).
    """
    state = {"binning": None, "psd_sum": None, "n_frames": 0}
    total_sum = 0.0
    total_sqsum = 0.0
    total_n = 0

    def _flush(chunk: list[np.ndarray]) -> None:
        if not chunk:
            return
        block = np.stack(chunk, axis=0)
        Ny, Nx = block.shape[1:]

        if state["binning"] is None:
            state["binning"] = _radial_binning(
                Ny, Nx, Nx * GRID_SPACING_KM, Ny * GRID_SPACING_KM
            )
            state["psd_sum"] = np.zeros(len(state["binning"][0]), dtype=np.float64)

        _, bin_idx, _ = state["binning"]
        power = np.abs(np.fft.fft2(block, axes=(1, 2))) ** 2
        del block

        for p in power:
            state["psd_sum"] += np.bincount(
                bin_idx,
                weights=p.ravel().astype(np.float64),
                minlength=state["psd_sum"].size,
            )
            state["n_frames"] += 1
        del power

    chunk: list[np.ndarray] = []
    for frame in _iter_frames(da, valid_times=valid_times):
        total_sum += float(frame.sum())
        total_sqsum += float((frame.astype(np.float64) ** 2).sum())
        total_n += frame.size

        chunk.append(frame)
        if len(chunk) >= CHUNK_FRAMES:
            _flush(chunk)
            chunk.clear()

    _flush(chunk)
    chunk.clear()
    gc.collect()

    if state["psd_sum"] is None or state["n_frames"] == 0:
        raise RuntimeError("No valid frames for RAPSD.")

    k_centers, _, counts = state["binning"]
    npix2 = float(counts.sum()) ** 2  # (Nx * Ny)^2

    mask = counts != 0
    psd = (state["psd_sum"][mask] / state["n_frames"]) / npix2 / counts[mask]

    mean = total_sum / total_n
    std = np.sqrt(max(total_sqsum / total_n - mean**2, 1e-30))
    psd = psd * (data_std / std) ** 2

    freqs = (k_centers[mask][::-1]) / (2 * np.pi)  # cycles / km
    return psd[::-1], freqs


def _get_valid_precip_times(obs: xr.DataArray) -> np.ndarray:
    nt = obs.sizes.get("time", 1)
    valid = []
    for t in tqdm(range(nt), desc="precip filter", leave=False):
        frame = np.asarray(obs.isel(time=t).values, dtype=np.float32)
        finite = np.isfinite(frame)
        if finite.sum() > 0 and np.nanmean(frame[finite]) >= PRECIP_MIN_MEAN:
            valid.append(t)
    print(f"  [precip] {len(valid)}/{nt} frames pass low-rain filter")
    return np.array(valid, dtype=int)


def _ralsd(rapsd_pred: np.ndarray, rapsd_ref: np.ndarray) -> float:
    N = len(rapsd_pred)
    d = 10.0 * np.log10(np.maximum(rapsd_pred, EPS) / np.maximum(rapsd_ref, EPS))
    return float(np.sqrt(np.sum(d**2) / N))


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _add_x_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    d = 0.012
    kl = dict(transform=ax_left.transAxes, color="0.35", clip_on=False, lw=1.0)
    kr = dict(transform=ax_right.transAxes, color="0.35", clip_on=False, lw=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kl)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kl)
    ax_right.plot((-d, +d), (-d, +d), **kr)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kr)


def _format_broken_pair(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    show_ylabel: bool,
    show_xticklabels: bool,
) -> None:
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    ax_left.tick_params(axis="y", right=False, labelleft=show_ylabel)
    ax_right.tick_params(axis="y", left=False, right=False, labelleft=False)

    ax_left.tick_params(axis="x", labelbottom=show_xticklabels)
    ax_right.tick_params(axis="x", labelbottom=show_xticklabels)

    if not show_ylabel:
        ax_left.set_ylabel("")


def _plot_variable(
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
    colors = {n: MODEL_COLORS.get(n, "C0") for n in model_spectra}

    for ax in (ax_top_left, ax_top_right):
        ax.semilogy(wl_km, ps_obs, zorder=5, **STYLE["Obs"])
        for name, (ps_pred, _) in model_spectra.items():
            ax.semilogy(wl_km, ps_pred, **dict(STYLE.get(name, {}), color=colors[name]))
        ax.grid(True, which="major", alpha=0.25, ls=":")
        ax.set_xlim(
            (WL_LEFT_MAX_KM, WL_BREAK_KM)
            if ax is ax_top_left
            else (WL_BREAK_KM, WL_MIN_KM)
        )

    for ax in (ax_bot_left, ax_bot_right):
        ax.axhline(1.0, color="black", lw=1.0, zorder=3)
        for name, (ps_pred, _) in model_spectra.items():
            ax.plot(
                wl_km,
                ps_pred / np.maximum(ps_obs, EPS),
                **dict(STYLE.get(name, {}), color=colors[name]),
            )
        ax.grid(True, which="major", alpha=0.25, ls=":")
        ax.set_xlim(
            (WL_LEFT_MAX_KM, WL_BREAK_KM)
            if ax is ax_bot_left
            else (WL_BREAK_KM, WL_MIN_KM)
        )
        ax.set_ylim(*RATIO_YLIM)

    ax_top_left.set_title(var_label, fontsize=13, fontweight="bold", loc="left")

    if show_ylabel:
        ax_top_left.set_ylabel(f"PSD ({psd_units})", fontsize=11)
        ax_bot_left.set_ylabel("Model / Obs", fontsize=11)

    for ax, ticks in (
        (ax_top_left, LEFT_TICKS_KM),
        (ax_top_right, RIGHT_TICKS_KM),
        (ax_bot_left, LEFT_TICKS_KM),
        (ax_bot_right, RIGHT_TICKS_KM),
    ):
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(v)) for v in ticks], fontsize=9)
        ax.tick_params(axis="both", labelsize=10)

    ax_bot_left.set_xlabel("Wavelength (km)", fontsize=11, loc="right")

    _format_broken_pair(
        ax_top_left, ax_top_right, show_ylabel=show_ylabel, show_xticklabels=False
    )
    _format_broken_pair(
        ax_bot_left, ax_bot_right, show_ylabel=show_ylabel, show_xticklabels=True
    )
    _add_x_break_marks(ax_top_left, ax_top_right)
    _add_x_break_marks(ax_bot_left, ax_bot_right)

    score_text = "RALSD (dB)\n" + "\n".join(
        f"{name:<9s}{ralsd:5.2f}" for name, (_, ralsd) in model_spectra.items()
    )
    ax_top_right.text(
        0.97,
        0.97,
        score_text,
        transform=ax_top_right.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.85"),
    )


def _add_shared_legend(fig: plt.Figure, model_names: list[str]) -> None:
    handles = [mlines.Line2D([], [], **STYLE["Obs"])]
    labels = ["Obs (reference)"]
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
        ncol=len(labels),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    D_SETUP = PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"
    D_DDIM = (
        PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference"
    )
    D_FM = (
        PROCESSING_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference"
    )
    D_OBS = (
        PROCESSING_ROOT
        / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full"
    )

    mask_lr = _load_mask(D_SETUP / "Swiss_Mask_LR.nc", "TabsD")
    mask_hr = _load_mask(D_SETUP / "Swiss_Mask_HR.nc", "TabsD")

    # Lazy factories: a model is only opened when its turn comes, then released.
    VAR_DEFS = [
        (
            "temp",
            lambda: _load_field(D_OBS / "TabsD_1971_2023.nc", "TabsD"),
            "Temperature",
            False,
            "K² km",
            {
                "Coarse": lambda: _load_field(
                    D_SETUP / "TabsD_step2_coarse.nc", "TabsD", mask_lr
                ).interp_like(mask_hr, method="nearest"),
                "Bicubic": lambda: _load_field(
                    D_SETUP / "TabsD_step3_interp_bicubic.nc", "TabsD", mask_hr
                ),
                "Bilinear": lambda: _load_field(
                    D_SETUP / "TabsD_step3_interp_bilinear.nc", "TabsD", mask_hr
                ),
                "UNet": lambda: _load_field(
                    D_DDIM / "unet_downscaled_test_set_2015_2023.nc", "temp", mask_hr
                ),
                "DDIM": lambda: _load_field(
                    D_DDIM / "ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
                    "temp",
                    mask_hr,
                ),
                "CFM": lambda: _load_field(
                    D_FM / "fm_downscaled_test_set_allframes_steps10_samples10.nc",
                    "temp",
                    mask_hr,
                ),
            },
        ),
        (
            "precip",
            lambda: _load_field(D_OBS / "RhiresD_1971_2023.nc", "RhiresD", clip=True),
            "Precipitation",
            True,
            "(mm day⁻¹)² km",
            {
                "Coarse": lambda: _load_field(
                    D_SETUP / "RhiresD_step2_coarse.nc", "RhiresD", mask_lr, clip=True
                ).interp_like(mask_hr, method="nearest"),
                "Bicubic": lambda: _load_field(
                    D_SETUP / "RhiresD_step3_interp_bicubic.nc",
                    "RhiresD",
                    mask_hr,
                    clip=True,
                ),
                "Bilinear": lambda: _load_field(
                    D_SETUP / "RhiresD_step3_interp_bilinear.nc",
                    "RhiresD",
                    mask_hr,
                    clip=True,
                ),
                "UNet": lambda: _load_field(
                    D_DDIM / "unet_downscaled_test_set_2015_2023.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
                "DDIM": lambda: _load_field(
                    D_DDIM / "ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
                "CFM": lambda: _load_field(
                    D_FM / "fm_downscaled_test_set_allframes_steps10_samples10.nc",
                    "precip",
                    mask_hr,
                    clip=True,
                ),
            },
        ),
    ]

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.4, 1.0, 1.4],
        height_ratios=[2.2, 1.0],
        wspace=0.06,
        hspace=0.10,
    )

    # Shared y-axis across the whole top row and across the whole bottom row.
    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_t1 = fig.add_subplot(gs[0, 1], sharey=ax_t0)
    ax_b1 = fig.add_subplot(gs[1, 1], sharey=ax_b0)
    ax_t2 = fig.add_subplot(gs[0, 2], sharey=ax_t0)
    ax_b2 = fig.add_subplot(gs[1, 2], sharey=ax_b0)
    ax_t3 = fig.add_subplot(gs[0, 3], sharey=ax_t0)
    ax_b3 = fig.add_subplot(gs[1, 3], sharey=ax_b0)

    axes = {
        "temp": (ax_t0, ax_t1, ax_b0, ax_b1),
        "precip": (ax_t2, ax_t3, ax_b2, ax_b3),
    }

    rows = []
    model_names: list[str] = []

    for col, (variable, obs_fn, var_label, is_precip, psd_units, model_fns) in enumerate(
        tqdm(VAR_DEFS, desc="variables")
    ):
        print(f"\n=== {variable} ===", flush=True)

        obs = obs_fn()
        valid_times = _get_valid_precip_times(obs) if is_precip else None
        rapsd_obs, freqs = rapsd_timemean(obs, valid_times=valid_times)
        del obs
        gc.collect()

        positive = freqs > 0
        wavelengths = np.full_like(freqs, np.inf, dtype=float)
        wavelengths[positive] = 1.0 / (freqs[positive] / GRID_SPACING_KM)
        valid = positive & (wavelengths >= NYQUIST_KM)

        ps_obs = rapsd_obs[valid]
        wl_km = 1.0 / (freqs[valid] / GRID_SPACING_KM)

        sort_idx = np.argsort(wl_km)
        wl_km = wl_km[sort_idx]
        ps_obs = ps_obs[sort_idx]

        model_spectra: dict[str, tuple[np.ndarray, float]] = {}
        for name, load_fn in tqdm(model_fns.items(), desc=variable, leave=False):
            pred = load_fn()
            rapsd_pred, _ = rapsd_timemean(pred, valid_times=valid_times)
            del pred
            gc.collect()

            ps_pred = rapsd_pred[valid][sort_idx]
            ralsd_val = _ralsd(ps_pred, ps_obs)
            print(f"  {name:10s}  RALSD = {ralsd_val:.4f} dB", flush=True)
            rows.append({"model": name, "variable": variable, "RALSD_mean": ralsd_val})
            model_spectra[name] = (ps_pred, ralsd_val)

        model_names = list(model_spectra.keys())

        ax_tl, ax_tr, ax_bl, ax_br = axes[variable]
        _plot_variable(
            ax_tl,
            ax_tr,
            ax_bl,
            ax_br,
            wl_km,
            ps_obs,
            model_spectra,
            var_label,
            psd_units,
            show_ylabel=(col == 0),
        )

    _add_shared_legend(fig, model_names)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.13)

    (PAPER_STATS_DIR / "Figures").mkdir(parents=True, exist_ok=True)
    out_png = PAPER_STATS_DIR / "Figures" / "rapsd_combined_method_II.png"
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png}")

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