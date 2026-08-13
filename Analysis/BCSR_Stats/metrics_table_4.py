from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config
from PSS import pss_gridwise_spatial_mean
from RAPSD import EPS, PRECIP_MIN_MEAN, ralsd, rapsd, wavenumber_to_wavelength_km
from mbe import mbe_gridwise_spatial_mean
from load_means import load_all_means_masked
from load_pooled import (
    ROW_ORDER,
    _sanitize,
    _subset_years,
    apply_mask_exact,
    load_all_pooled_masked,
)

SAMPLE_DIMS = ("member", "sample", "samples")

# Spectral-ratio panels: identical, fixed y-range on ALL ratio subplots.
RATIO_YLIM = (0.0, 1.75)

# Broken x-axis (wavelength, km): continuous 50 -> 12 km, zoom 12 -> 2 km.
# WL_MIN_KM is the Nyquist wavelength for a 1 km grid; nothing exists below it,
# so the zoom panel must not extend past it (otherwise it renders blank).
WL_LEFT_MAX_KM = 50.0
WL_BREAK_KM = 12.0
WL_MIN_KM = 1.0

LEFT_TICKS_KM = np.array([50, 36, 24, 12], dtype=float)
RIGHT_TICKS_KM = np.array([12, 9, 7, 5, 3, 1], dtype=float)

# Black is reserved for OBS, so it is deliberately absent from the palette.
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#808000",
    "#008080", "#bcbd22", "#17becf", "#8c564b", "#469990",
]
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _norm_name(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _resolve_baseline_da(
    data_map: dict[str, xr.DataArray],
    baseline: str,
) -> xr.DataArray | None:
    if baseline in data_map:
        return data_map[baseline]

    target = _norm_name(baseline)
    for k, v in data_map.items():
        if _norm_name(k) == target:
            return v

    cands = [
        (k, v)
        for k, v in data_map.items()
        if target in _norm_name(k) or _norm_name(k) in target
    ]
    if not cands:
        return None

    cands.sort(key=lambda kv: len(_norm_name(kv[0])), reverse=True)
    return cands[0][1]


def _resolve_sample_dim(da: xr.DataArray) -> str | None:
    for d in SAMPLE_DIMS:
        if d in da.dims:
            return d
    return None


def _load_obs(args, loaded):
    with xr.open_dataset(args.obs_tas_file, chunks={"time": 365}) as ds_tas:
        obs_tas = _sanitize(ds_tas[args.obs_tas_var], "tas").load()
    with xr.open_dataset(args.obs_pr_file, chunks={"time": 365}) as ds_pr:
        obs_pr = _sanitize(ds_pr[args.obs_pr_var], "pr").load()

    obs_tas = apply_mask_exact(
        _subset_years(obs_tas, args.eval_start, args.eval_end),
        loaded.hr_mask,
    )
    obs_pr = apply_mask_exact(
        _subset_years(obs_pr, args.eval_start, args.eval_end),
        loaded.hr_mask,
    )
    return {"tas": obs_tas, "pr": obs_pr}


def _compute_mbe_for_var(loaded, obs_eval: xr.DataArray, var: str) -> pd.DataFrame:
    table = pd.DataFrame(index=ROW_ORDER, columns=["MBE"], dtype=float)
    available = loaded.data.get(var, {})

    for baseline in ROW_ORDER:
        pred = _resolve_baseline_da(available, baseline)
        if pred is None:
            continue

        pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
        if pred_eval.sizes.get("time", 0) == 0:
            continue

        table.loc[baseline, "MBE"] = mbe_gridwise_spatial_mean(pred_eval, ref_eval)

    return table


def _compute_pss_for_var(args, loaded, obs_eval: xr.DataArray, var: str) -> pd.DataFrame:
    table = pd.DataFrame(index=ROW_ORDER, columns=["PSS"], dtype=float)
    available = loaded.data.get(var, {})

    for baseline in ROW_ORDER:
        pred = _resolve_baseline_da(available, baseline)
        if pred is None:
            continue

        pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
        if pred_eval.sizes.get("time", 0) == 0:
            continue

        table.loc[baseline, "PSS"] = pss_gridwise_spatial_mean(
            pred_eval,
            ref_eval,
            nbins=args.nbins,
            mode="pooled",
        )

    return table


def _model_style(i: int) -> dict:
    """Unique (colour, linestyle, marker) triple per model index."""
    return dict(
        color=_PALETTE[i % len(_PALETTE)],
        ls=_LINESTYLES[i % len(_LINESTYLES)],
        marker=_MARKERS[i % len(_MARKERS)],
        markersize=3.5,
        # Integer stride, not a fraction: the broken axis clips each subplot to
        # a narrow window, and fractional markevery is computed over the FULL
        # data extent, which can leave a panel with zero markers.
        markevery=max(1, 3 + i),
        lw=1.6,
        alpha=0.9,
    )


def _add_x_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Diagonal tick marks indicating the axis break."""
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


def _to_wavelength(
    k: np.ndarray,
    ps: np.ndarray,
    shape: tuple[int, int],
    grid_spacing_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """k -> wavelength (km), Nyquist-clipped and sorted ascending."""
    wl = wavenumber_to_wavelength_km(k, shape, grid_spacing_km)
    nyquist = 2.0 * grid_spacing_km
    keep = np.isfinite(wl) & (wl >= nyquist)
    wl, ps = wl[keep], ps[keep]
    order = np.argsort(wl)
    return wl[order], ps[order]


def _plot_rapsd_summary(
    out: Path,
    tas_ref: tuple[np.ndarray, np.ndarray],
    tas_spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
    pr_ref: tuple[np.ndarray, np.ndarray],
    pr_spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
    tas_shape: tuple[int, int],
    pr_shape: tuple[int, int],
    grid_spacing_km: float = 1.0,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.5, 1.0, 1.5],
        height_ratios=[2.2, 1.0],
        wspace=0.06,
        hspace=0.10,
    )

    # Top row: share y only WITHIN a variable — tas power (K^2) and pr power
    # (mm^2/day^2) differ by orders of magnitude, so a global share would
    # flatten one panel into a line.
    # Bottom row: share y across everything (all ratios are 0-1.75).
    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_t1 = fig.add_subplot(gs[0, 1], sharey=ax_t0)
    ax_b1 = fig.add_subplot(gs[1, 1], sharey=ax_b0)
    ax_t2 = fig.add_subplot(gs[0, 2])                       # own y-scale
    ax_b2 = fig.add_subplot(gs[1, 2], sharey=ax_b0)
    ax_t3 = fig.add_subplot(gs[0, 3], sharey=ax_t2)
    ax_b3 = fig.add_subplot(gs[1, 3], sharey=ax_b0)

    # (var, label, ref, spectra, shape, axes, show_power_ylabel, show_ratio_ylabel)
    panels = [
        ("tas", "Temperature", tas_ref, tas_spectra, tas_shape,
         (ax_t0, ax_t1, ax_b0, ax_b1), True, True),
        ("pr", "Precipitation", pr_ref, pr_spectra, pr_shape,
         (ax_t2, ax_t3, ax_b2, ax_b3), True, False),
    ]

    # Consistent style per model NAME across both variables.
    all_names = list(dict.fromkeys([*tas_spectra, *pr_spectra]))
    styles = {name: _model_style(i) for i, name in enumerate(all_names)}
    handles: dict[str, plt.Line2D] = {}

    for (_var, var_label, (k_ref, ps_ref), spectra, shape, axs,
         show_power_ylabel, show_ratio_ylabel) in panels:
        ax_tl, ax_tr, ax_bl, ax_br = axs

        wl_ref, ps_ref_wl = _to_wavelength(k_ref, ps_ref, shape, grid_spacing_km)

        # ── top row: spectra ────────────────────────────────────────────────
        for ax in (ax_tl, ax_tr):
            if wl_ref.size:
                (h_obs,) = ax.semilogy(
                    wl_ref, ps_ref_wl, color="black", lw=1.8, ls="-",
                    zorder=5, label="OBS",
                )
                handles.setdefault("OBS", h_obs)

            for baseline, (k_mod, ps_mod, _) in spectra.items():
                wl_m, ps_m = _to_wavelength(k_mod, ps_mod, shape, grid_spacing_km)
                if not wl_m.size:
                    continue
                (h,) = ax.semilogy(wl_m, ps_m, **styles[baseline])
                handles.setdefault(baseline, h)

            ax.grid(True, which="major", alpha=0.25, ls=":")
            # Descending limits put large scales on the left; do NOT also call
            # invert_xaxis() or the two cancel out.
            ax.set_xlim(
                (WL_LEFT_MAX_KM, WL_BREAK_KM) if ax is ax_tl
                else (WL_BREAK_KM, WL_MIN_KM)
            )

        # ── bottom row: model / obs ratio ───────────────────────────────────
        for ax in (ax_bl, ax_br):
            ax.axhline(1.0, color="black", lw=1.0, zorder=3)

            for baseline, (k_mod, ps_mod, _) in spectra.items():
                if not (k_ref.size and k_mod.size):
                    continue
                # interpolate onto the obs k-grid, then relabel to wavelength
                n = min(k_ref.size, k_mod.size)
                k_grid = k_ref[:n]
                ps_r = np.interp(k_grid, k_ref, ps_ref)
                ps_m = np.interp(k_grid, k_mod, ps_mod)
                wl_g, ratio = _to_wavelength(
                    k_grid, ps_m / np.maximum(ps_r, EPS), shape, grid_spacing_km
                )
                ax.plot(wl_g, ratio, **styles[baseline])

            ax.grid(True, which="major", alpha=0.25, ls=":")
            ax.set_xlim(
                (WL_LEFT_MAX_KM, WL_BREAK_KM) if ax is ax_bl
                else (WL_BREAK_KM, WL_MIN_KM)
            )
            ax.set_ylim(*RATIO_YLIM)          # identical, fixed 0–1.75

        ax_tl.set_title(var_label, fontsize=13, fontweight="bold", loc="left")

        if show_power_ylabel:
            ax_tl.set_ylabel("Power", fontsize=11)
        if show_ratio_ylabel:
            ax_bl.set_ylabel("Model / Obs", fontsize=11)

        for ax, ticks in (
            (ax_tl, LEFT_TICKS_KM), (ax_tr, RIGHT_TICKS_KM),
            (ax_bl, LEFT_TICKS_KM), (ax_br, RIGHT_TICKS_KM),
        ):
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(v)) for v in ticks], fontsize=9)
            ax.tick_params(axis="both", labelsize=10)

        ax_bl.set_xlabel("Wavelength (km)", fontsize=11, loc="right")

        _format_broken_pair(ax_tl, ax_tr, show_ylabel=show_power_ylabel,
                            show_xticklabels=False)
        _format_broken_pair(ax_bl, ax_br, show_ylabel=show_ratio_ylabel,
                            show_xticklabels=True)
        _add_x_break_marks(ax_tl, ax_tr)
        _add_x_break_marks(ax_bl, ax_br)

        # RALSD scores as a compact monospace table.
        if spectra:
            txt = "RALSD (dB)\n" + "\n".join(
                f"{n:<12s}{s:6.3f}"
                for n, (_, _, s) in spectra.items() if np.isfinite(s)
            )
            ax_tr.text(
                0.97, 0.97, txt, transform=ax_tr.transAxes, fontsize=8,
                family="monospace", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.85, ec="0.85"),
            )

    fig.legend(
        list(handles.values()), list(handles.keys()),
        loc="lower center", ncol=min(len(handles), 6),
        fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=3.0, columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.13)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved RAPSD summary → {out}")


def _compute_ralsd_for_var(args, loaded, obs_eval: xr.DataArray, var: str):
    table = pd.DataFrame(index=ROW_ORDER, columns=["RALSD"], dtype=float)
    available = loaded.data.get(var, {})

    # precip only: drop near-empty frames (unpaired, per-dataset filter)
    min_mean = PRECIP_MIN_MEAN if var == "pr" else None

    # spatial shape (H, W) — needed to map wavenumber -> wavelength in km
    spatial_dims = [d for d in obs_eval.dims if d != "time"]
    field_shape = tuple(int(obs_eval.sizes[d]) for d in spatial_dims[-2:])

    k_ref, ps_ref = rapsd(
        obs_eval,
        time_dim="time",
        min_frame_mean=min_mean,
        desc=f"RAPSD obs {var}",
    )

    spectra: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for baseline in ROW_ORDER:
        pred = _resolve_baseline_da(available, baseline)
        if pred is None:
            continue

        pred_eval, _ = xr.align(pred, obs_eval, join="inner")
        if pred_eval.sizes.get("time", 0) == 0:
            continue

        sample_dim = _resolve_sample_dim(pred_eval)
        k_mod, ps_mod = rapsd(
            pred_eval,
            time_dim="time",
            sample_dim=sample_dim,
            min_frame_mean=min_mean,
            desc=f"RAPSD {baseline} {var}",
        )
        score = ralsd(k_ref, ps_ref, k_mod, ps_mod)

        table.loc[baseline, "RALSD"] = score
        spectra[baseline] = (k_mod, ps_mod, score)

    return table, (k_ref, ps_ref), spectra, field_shape


def _write_table(table: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    table = table.reindex(ROW_ORDER)
    table.to_csv(out, float_format="%.3f")


def _resolve_means_root(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    alt = p.parent / "Ensmeans"
    if alt.exists():
        return alt
    alt2 = p.parent / "EnsMeans"
    if alt2.exists():
        return alt2
    return p


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mbe", "pss", "rapsd"], required=True)
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"))
    ap.add_argument("--ens_means_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/Ensmeans"))
    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--mbe_csv", default="Analysis/BCSR_Stats/Tables/metrics_mbe.csv")
    ap.add_argument("--pss_csv", default="Analysis/BCSR_Stats/Tables/metrics_pss.csv")
    ap.add_argument("--ralsd_csv", default="Analysis/BCSR_Stats/Tables/metrics_rapsd.csv")
    ap.add_argument("--rapsd_plot_dir", default="Analysis/BCSR_Stats/Figures")
    ap.add_argument("--grid_spacing_km", type=float, default=1.0,
                    help="HR grid spacing in km (used for the wavelength axis)")
    ap.add_argument("--verbose_loader", action="store_true")
    args = ap.parse_args()

    if args.eval_start > args.eval_end:
        raise ValueError("eval_start must be <= eval_end")

    means_root = _resolve_means_root(args.ens_means_root)

    loaded_pooled = load_all_pooled_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
        verbose=args.verbose_loader,
    )

    loaded_means = None
    if args.mode == "mbe":
        loaded_means = load_all_means_masked(
            ens_root=means_root,
            mask_hr_file=args.mask_hr_file,
            mask_hr_var=args.mask_hr_var,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            variables=["tas", "pr"],
            verbose=args.verbose_loader,
        )

    obs_map = _load_obs(args, loaded_pooled)

    if args.mode == "mbe":
        tas_mbe = _compute_mbe_for_var(loaded_means, obs_map["tas"], "tas")
        pr_mbe = _compute_mbe_for_var(loaded_means, obs_map["pr"], "pr")

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_MBE"] = tas_mbe["MBE"]
        out["pr_MBE"] = pr_mbe["MBE"]

        _write_table(out, Path(args.mbe_csv))
        print(f"[ok] wrote {args.mbe_csv}")
        return

    if args.mode == "pss":
        tas_pss = _compute_pss_for_var(args, loaded_pooled, obs_map["tas"], "tas")
        pr_pss = _compute_pss_for_var(args, loaded_pooled, obs_map["pr"], "pr")

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_PSS"] = tas_pss["PSS"]
        out["pr_PSS"] = pr_pss["PSS"]

        _write_table(out, Path(args.pss_csv))
        print(f"[ok] wrote {args.pss_csv}")
        return

    if args.mode == "rapsd":
        tas_ralsd, tas_ref, tas_spectra, tas_shape = _compute_ralsd_for_var(
            args, loaded_pooled, obs_map["tas"], "tas"
        )
        pr_ralsd, pr_ref, pr_spectra, pr_shape = _compute_ralsd_for_var(
            args, loaded_pooled, obs_map["pr"], "pr"
        )

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_RALSD"] = tas_ralsd["RALSD"]
        out["pr_RALSD"] = pr_ralsd["RALSD"]

        _write_table(out, Path(args.ralsd_csv))
        print(f"[ok] wrote {args.ralsd_csv}")

        _plot_rapsd_summary(
            out=Path(args.rapsd_plot_dir) / "rapsd_summary.png",
            tas_ref=tas_ref,
            tas_spectra=tas_spectra,
            pr_ref=pr_ref,
            pr_spectra=pr_spectra,
            tas_shape=tas_shape,
            pr_shape=pr_shape,
            grid_spacing_km=args.grid_spacing_km,
        )
        return


if __name__ == "__main__":
    main()