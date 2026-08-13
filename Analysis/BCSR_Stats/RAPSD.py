from __future__ import annotations

import gc
from functools import lru_cache

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.fft import fft2, fftshift
from tqdm import tqdm

EPS = 1e-30
SAMPLE_DIM_CANDIDATES = ("member", "sample", "samples")

# Harris et al. (2022): drop frames with domain-mean rain < 0.002 mm/hr.
# Daily data -> 0.002 * 24 = 0.048 mm/day. Keep this aligned with the paper scripts.
PRECIP_MIN_MEAN = 0.05

WL_LEFT_MAX_KM = 50.0
WL_BREAK_KM = 12.0
WL_MIN_KM = 1.0

LEFT_TICKS_KM = np.array([50, 40, 30, 20, 12], dtype=float)
RIGHT_TICKS_KM = np.array([12, 9, 7, 5, 3, 1], dtype=float)

RATIO_YLIM = (0.0, 2.0)

MODEL_COLORS = {
    "Obs": "black",
    "Coarse": "#7f7f7f",
    "Bicubic": "#1f77b4",
    "Bilinear": "#ff7f0e",
    "UNet": "#2ca02c",
    "DDIM": "#d62728",
    "CFM": "#9467bd",
}

DEFAULT_STYLES = {
    "Obs": dict(color="black", ls="-", lw=1.8),
    "Coarse": dict(ls="--", lw=1.0),
    "Bicubic": dict(ls="--", lw=1.0),
    "Bilinear": dict(ls="--", lw=1.0),
    "UNet": dict(ls="-", lw=1.2),
    "DDIM": dict(ls="-", lw=1.5),
    "CFM": dict(ls="-", lw=1.5),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_sample_dim(da: xr.DataArray, sample_dim: str | None = "auto") -> str | None:
    if sample_dim is None:
        return None
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample_dim='{sample_dim}' not found in dims={da.dims}")
        return sample_dim
    return next((d for d in SAMPLE_DIM_CANDIDATES if d in da.dims), None)


@lru_cache(maxsize=16)
def _max_radius(shape: tuple[int, int]) -> int:
    return min(shape) // 2


def _compute_rapsd(img: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Per-frame RAPSD kernel matching the backup script:
    - float64
    - finite mask
    - fill NaNs with mean of valid pixels
    - demean
    - fft2 + fftshift
    - |FFT|^2 / (h*w)
    - rounded radial bins
    - truncate at min(h, w)//2
    """
    img = np.asarray(img, dtype=np.float64)

    valid = np.isfinite(img)
    if valid.sum() < 4:
        return np.array([])

    img = np.where(valid, img, img[valid].mean())
    img = img - img.mean()

    if np.std(img) < 1e-6:
        return np.array([])

    h, w = img.shape
    power = (np.abs(fftshift(fft2(img))) ** 2) / (h * w)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = min(h, w) // 2

    radius_int = np.round(radius).astype(int)
    rapsd = np.zeros(max_radius)
    for r in range(max_radius):
        annulus = radius_int == r
        if annulus.sum() > 0:
            rapsd[r] = np.mean(power[annulus])

    if normalize:
        s = rapsd.sum()
        if s > 0:
            rapsd = rapsd / s

    return rapsd


def wavenumber_to_wavelength_km(
    k: np.ndarray, shape: tuple[int, int], grid_spacing_km: float = 1.0
) -> np.ndarray:
    """
    Convert radial wavenumber index to wavelength in km using the same
    convention as the backup script.
    """
    max_r = _max_radius(tuple(shape))
    freq = np.asarray(k, dtype=np.float64) / max_r * 0.5
    with np.errstate(divide="ignore"):
        return grid_spacing_km / freq


def _interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    valid = np.isfinite(x_src) & np.isfinite(y_src)
    if valid.sum() < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)

    x = x_src[valid]
    y = y_src[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.full_like(x_tgt, np.nan, dtype=float)
    inside = (x_tgt >= x[0]) & (x_tgt <= x[-1])
    out[inside] = np.interp(x_tgt[inside], x, y)
    return out


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


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def rapsd(
    da: xr.DataArray,
    time_dim: str | None = "time",
    sample_dim: str | None = "auto",
    normalize: bool = False,
    desc: str = "RAPSD",
    chunk_size: int = 50,
    min_frame_mean: float | None = None,
    return_counts: bool = False,
):
    """
    Compute the mean Radially Averaged Power Spectral Density.

    The per-frame kernel matches the backup script exactly.
    Evaluation remains unpaired: each dataset is processed independently.
    """
    sd = _resolve_sample_dim(da, sample_dim)

    real_dims = [d for d in [time_dim, sd] if d is not None and d in da.dims]
    spatial_dims = [d for d in da.dims if d not in real_dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")

    da = da.transpose(*real_dims, *spatial_dims)

    has_time = bool(time_dim) and time_dim in da.dims
    n_time = da.sizes[time_dim] if has_time else 1
    n_sample = da.sizes[sd] if sd and sd in da.dims else 1
    n_frames = n_time * n_sample

    profiles: list[np.ndarray] = []
    skipped = 0
    dry_skipped = 0

    with tqdm(total=n_frames, desc=desc, unit="frame", leave=False) as pbar:
        for t_start in range(0, n_time, chunk_size):
            t_end = min(t_start + chunk_size, n_time)
            sub = da.isel({time_dim: slice(t_start, t_end)}) if has_time else da
            chunk = np.asarray(sub.values, dtype=np.float64)
            chunk = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])

            for frame in chunk:
                if min_frame_mean is not None:
                    finite = np.isfinite(frame)
                    if finite.sum() == 0 or frame[finite].mean() < min_frame_mean:
                        dry_skipped += 1
                        pbar.update(1)
                        continue

                profile = _compute_rapsd(frame, normalize=normalize)
                if profile.size == 0:
                    skipped += 1
                else:
                    profiles.append(profile)
                pbar.update(1)

            del chunk
            gc.collect()

    if dry_skipped:
        print(f"[rapsd] {dry_skipped}/{n_frames} frames below min_frame_mean")
    if skipped:
        print(f"[rapsd] skipped {skipped}/{n_frames} near-constant/empty frames")

    if not profiles:
        raise RuntimeError("All frames were skipped — no valid spectra computed.")

    min_len = min(p.size for p in profiles)
    stack = np.stack([p[:min_len] for p in profiles], axis=0)
    ps_mean = stack.mean(axis=0)
    k = np.arange(min_len, dtype=np.float64)

    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)

    if return_counts:
        counts = {
            "n_frames": n_frames,
            "n_used": len(profiles),
            "n_below_threshold": dry_skipped,
            "n_near_constant": skipped,
        }
        return k[keep], ps_mean[keep], counts

    return k[keep], ps_mean[keep]


def ralsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    k_mod: np.ndarray,
    ps_mod: np.ndarray,
) -> float:
    """
    RALSD = sqrt(mean((10 log10(S_true / S_pred))^2)).
    """
    if k_ref.size == 0 or k_mod.size == 0:
        return np.nan

    n = min(k_ref.size, k_mod.size)
    p_ref = np.asarray(ps_ref[:n], dtype=np.float64)
    p_mod = np.asarray(ps_mod[:n], dtype=np.float64)
    mask = np.isfinite(p_ref) & np.isfinite(p_mod) & (p_ref > 0) & (p_mod > 0)

    if not np.any(mask):
        return np.nan

    log_ratio = 10.0 * np.log10(np.maximum(p_ref[mask] / np.maximum(p_mod[mask], EPS), EPS))
    return float(np.sqrt(np.mean(log_ratio**2)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rapsd(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_key: str = "Obs",
    title: str = "RAPSD",
    ax: plt.Axes | None = None,
    style_map: dict | None = None,
    field_shape: tuple[int, int] | None = None,
    grid_spacing_km: float = 1.0,
) -> plt.Axes:
    """
    Backward-compatible single-axis plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    def _x(k):
        if field_shape is None:
            return k
        return wavenumber_to_wavelength_km(k, tuple(field_shape), grid_spacing_km)

    k_obs, ps_obs = spectra[obs_key]
    ax.semilogy(_x(k_obs), ps_obs, color="black", lw=2.5, ls="-", label=obs_key)

    for name, (k, ps) in spectra.items():
        if name == obs_key:
            continue
        score = ralsd(k_obs, ps_obs, k, ps)
        label = f"{name}  (RALSD={score:.2f} dB)"
        sty = dict((style_map or {}).get(name, {}))
        ax.semilogy(_x(k), ps, label=label, **sty)

    if field_shape is None:
        ax.set_xlabel("Wavenumber (k)", fontsize=12)
    else:
        ax.set_xlabel("Wavelength (km)", fontsize=12)
        ax.invert_xaxis()

    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_rapsd_broken(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_key: str = "Obs",
    title: str = "RAPSD",
    field_shape: tuple[int, int] | None = None,
    grid_spacing_km: float = 1.0,
    style_map: dict | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """
    Backup-style broken-axis plot:
    - wavelength space
    - left panel: 50 -> 12 km
    - right panel: 12 -> 1 km
    - ratio panel below
    """
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.5],
        height_ratios=[2.2, 1.0],
        wspace=0.06,
        hspace=0.10,
    )

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1], sharey=ax_tl)
    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_tl)
    ax_br = fig.add_subplot(gs[1, 1], sharey=ax_bl)

    def _x(k):
        if field_shape is None:
            return k
        return wavenumber_to_wavelength_km(k, tuple(field_shape), grid_spacing_km)

    obs_k, obs_ps = spectra[obs_key]
    obs_x = _x(obs_k)

    colors = {name: MODEL_COLORS.get(name, "C0") for name in spectra}

    for ax in (ax_tl, ax_tr):
        ax.semilogy(obs_x, obs_ps, color="black", lw=1.8, ls="-", zorder=5, label=obs_key)
        for name, (k, ps) in spectra.items():
            if name == obs_key:
                continue
            x = _x(k)
            sty = dict((style_map or {}).get(name, {}))
            sty.setdefault("color", colors[name])
            ax.semilogy(x, ps, **sty)
        ax.grid(True, which="major", alpha=0.25, ls=":")
        ax.set_xlim((WL_LEFT_MAX_KM, WL_BREAK_KM) if ax is ax_tl else (WL_BREAK_KM, WL_MIN_KM))

    for ax in (ax_bl, ax_br):
        ax.axhline(1.0, color="black", lw=1.0, zorder=3)
        for name, (k, ps) in spectra.items():
            if name == obs_key:
                continue
            x = _x(k)
            obs_on_x = _interp_1d(obs_x, obs_ps, x)
            sty = dict((style_map or {}).get(name, {}))
            sty.setdefault("color", colors[name])
            ax.plot(x, ps / np.maximum(obs_on_x, EPS), **sty)
        ax.grid(True, which="major", alpha=0.25, ls=":")
        ax.set_xlim((WL_LEFT_MAX_KM, WL_BREAK_KM) if ax is ax_bl else (WL_BREAK_KM, WL_MIN_KM))
        ax.set_ylim(*RATIO_YLIM)

    ax_tl.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax_tl.set_ylabel("PSD", fontsize=11)
    ax_bl.set_ylabel("Model / Obs", fontsize=11)
    ax_bl.set_xlabel("Wavelength (km)", fontsize=11, loc="right")

    for ax, ticks in (
        (ax_tl, LEFT_TICKS_KM),
        (ax_tr, RIGHT_TICKS_KM),
        (ax_bl, LEFT_TICKS_KM),
        (ax_br, RIGHT_TICKS_KM),
    ):
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(v)) for v in ticks], fontsize=9)
        ax.tick_params(axis="both", labelsize=10)

    _format_broken_pair(ax_tl, ax_tr, show_ylabel=True, show_xticklabels=False)
    _format_broken_pair(ax_bl, ax_br, show_ylabel=True, show_xticklabels=True)
    _add_x_break_marks(ax_tl, ax_tr)
    _add_x_break_marks(ax_bl, ax_br)

    score_text = "RALSD (dB)\n" + "\n".join(
        f"{name:<9s}{ralsd(obs_k, obs_ps, k, ps):5.2f}"
        for name, (k, ps) in spectra.items()
        if name != obs_key
    )
    ax_tr.text(
        0.97,
        0.97,
        score_text,
        transform=ax_tr.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.85"),
    )

    handles = [mlines.Line2D([], [], color="black", lw=1.8, ls="-")]
    labels = [obs_key]
    for name in spectra:
        if name == obs_key:
            continue
        sty = dict((style_map or {}).get(name, {}))
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=colors[name],
                lw=sty.get("lw", 1.5),
                ls=sty.get("ls", "-"),
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

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.13)
    return fig, (ax_tl, ax_tr, ax_bl, ax_br)


def spectral_ratio(
    da_pred: xr.DataArray,
    da_obs: xr.DataArray,
    time_dim: str | None = "time",
    sample_dim: str | None = "auto",
    normalize: bool = False,
    min_frame_mean: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    k_pred, ps_pred = rapsd(
        da_pred,
        time_dim=time_dim,
        sample_dim=sample_dim,
        normalize=normalize,
        min_frame_mean=min_frame_mean,
        desc="RAPSD pred",
    )
    k_obs, ps_obs = rapsd(
        da_obs,
        time_dim=time_dim,
        sample_dim=None,
        normalize=normalize,
        min_frame_mean=min_frame_mean,
        desc="RAPSD obs",
    )

    n = min(k_pred.size, k_obs.size)
    if n == 0:
        raise ValueError("Empty spectra returned by rapsd().")

    ratio = ps_pred[:n] / np.maximum(ps_obs[:n], EPS)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(k_pred[:n], ratio, lw=2, color="steelblue")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Spectral ratio (pred / obs)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return ax