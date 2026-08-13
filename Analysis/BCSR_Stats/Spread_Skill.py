"""
Spread-skill diagrams for pooled ensembles — standalone.

Top panel    : tas (K), all baselines
Bottom panel : pr  (mm/day, wet days only), all baselines

x = binned ensemble spread, y = RMSE of the ensemble mean in that bin.
A perfectly calibrated ensemble lies on the 1:1 line.
Above the line = underdispersive (overconfident), below = overdispersive.

Run:
    python Analysis/BCSR_Stats/Spread_Skill.py --verbose_loader
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config

from load_pooled import (
    ROW_ORDER,
    _sanitize,
    _subset_years,
    apply_mask_exact,
    load_all_pooled_masked,
)

SAMPLE_DIMS = ("member", "sample", "samples")

EPS = 1e-12

# Wet-day threshold (mm/day). On dry days every member is ~0, so the spread
# collapses and SSR is dragged toward 0 for reasons that have nothing to do
# with calibration. Precip SSR must be conditioned on wet days.
WET_DAY_THRESHOLD = 1.0

# Black is reserved for the 1:1 calibration line.
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9a6324", "#808000", "#008080",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style(i: int) -> dict:
    return dict(
        color=_PALETTE[i % len(_PALETTE)],
        marker=_MARKERS[i % len(_MARKERS)],
        markersize=5,
        lw=1.6,
        alpha=0.9,
    )


def _norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def _resolve(data_map, baseline):
    if baseline in data_map:
        return data_map[baseline]
    t = _norm(baseline)
    for k, v in data_map.items():
        if _norm(k) == t:
            return v
    return None


def _sample_dim(da: xr.DataArray) -> str | None:
    for d in SAMPLE_DIMS:
        if d in da.dims:
            return d
    return None


def _wet_mask(obs: xr.DataArray, threshold: float | None):
    return None if threshold is None else obs >= threshold


# ---------------------------------------------------------------------------
# Spread-skill
#
#   SSR = sqrt((m+1)/m) * sigma_bar / RMSE(ens_mean)
#
# The sqrt((m+1)/m) factor (Fortin et al. 2014) corrects the finite-ensemble
# low bias in sigma. m differs by ~10x between deterministic rows (GCM members
# only) and generative rows (members x samples), so an uncorrected SSR would
# penalise the small-m rows purely for being small.
#
#   SSR ~ 1   calibrated
#   SSR < 1   underdispersive (overconfident)
#   SSR > 1   overdispersive
# ---------------------------------------------------------------------------

def ensemble_size(da: xr.DataArray, member_dim: str | None = None) -> int:
    d = member_dim if member_dim in da.dims else _sample_dim(da)
    return int(da.sizes[d]) if d is not None else 1


def spread_skill(
    pred: xr.DataArray,
    obs: xr.DataArray,
    member_dim: str | None = None,
    wet_threshold: float | None = None,
    return_components: bool = False,
):
    """
    Aggregate SSR over all (time, lat, lon) points.

    pred : (member, time, lat, lon) or (time, lat, lon)
    obs  : (time, lat, lon)
    wet_threshold : restrict to obs >= threshold (WET_DAY_THRESHOLD for pr,
                    None for tas)

    Returns SSR, or (SSR, spread, rmse, m) if return_components.
    """
    dim = member_dim if member_dim in pred.dims else _sample_dim(pred)
    m = int(pred.sizes[dim]) if dim is not None else 1

    if dim is not None and m > 1:
        # ddof=1: sample variance, matching the (m+1)/m correction below
        var_ens = pred.var(dim=dim, ddof=1)
        ens_mean = pred.mean(dim=dim)
    else:
        var_ens = xr.zeros_like(obs)
        ens_mean = pred.mean(dim=dim) if dim is not None else pred

    err_sq = (ens_mean - obs) ** 2

    mask = _wet_mask(obs, wet_threshold)
    if mask is not None:
        var_ens = var_ens.where(mask)
        err_sq = err_sq.where(mask)

    # aggregate variances first, one sqrt at the end — averaging sigma
    # pointwise is biased low (Jensen)
    spread = float(np.sqrt(var_ens.mean(skipna=True).values))
    rmse = float(np.sqrt(err_sq.mean(skipna=True).values))

    if m > 1:
        spread *= np.sqrt((m + 1.0) / m)

    ssr = spread / max(rmse, EPS) if rmse > EPS else np.nan

    if return_components:
        return ssr, spread, rmse, m
    return ssr


def spread_skill_binned(
    pred: xr.DataArray,
    obs: xr.DataArray,
    nbins: int = 10,
    member_dim: str | None = None,
    wet_threshold: float | None = None,
):
    """
    Spread-skill *diagram*: bin points by ensemble spread, return RMS spread
    and RMSE per bin. A calibrated ensemble sits on the 1:1 line.

    Returns (bin_spread, bin_rmse, bin_count).
    """
    dim = member_dim if member_dim in pred.dims else _sample_dim(pred)
    m = int(pred.sizes[dim]) if dim is not None else 1
    if dim is None or m < 2:
        return np.array([]), np.array([]), np.array([])

    sigma = pred.std(dim=dim, ddof=1) * np.sqrt((m + 1.0) / m)
    err_sq = (pred.mean(dim=dim) - obs) ** 2

    mask = _wet_mask(obs, wet_threshold)
    if mask is not None:
        sigma = sigma.where(mask)
        err_sq = err_sq.where(mask)

    # float32: arrays are (time, lat, lon) after the member reduction, but the
    # full-domain ravel is still large and this runs once per baseline.
    s = np.asarray(sigma.values, dtype=np.float32).ravel()
    e = np.asarray(err_sq.values, dtype=np.float32).ravel()
    good = np.isfinite(s) & np.isfinite(e)
    s, e = s[good], e[good]
    if s.size == 0:
        return np.array([]), np.array([]), np.array([])

    # equal-population bins: robust to the skewed precip spread distribution
    edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size < 2:
        return np.array([]), np.array([]), np.array([])

    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)

    nb = len(edges) - 1
    bin_spread = np.full(nb, np.nan)
    bin_rmse = np.full(nb, np.nan)
    bin_count = np.zeros(nb, dtype=int)

    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        bin_count[b] = n
        if n:
            bin_spread[b] = np.sqrt(np.mean(s[sel] ** 2))
            bin_rmse[b] = np.sqrt(np.mean(e[sel]))

    return bin_spread, bin_rmse, bin_count


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_obs(args, hr_mask):
    out = {}
    for var, f, vn in (
        ("tas", args.obs_tas_file, args.obs_tas_var),
        ("pr", args.obs_pr_file, args.obs_pr_var),
    ):
        with xr.open_dataset(f, chunks={"time": 365}) as ds:
            da = _sanitize(ds[vn], var).load()
        out[var] = apply_mask_exact(
            _subset_years(da, args.eval_start, args.eval_end), hr_mask
        )
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(out_png: Path, results: dict, summary: pd.DataFrame) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Two rows, one column: tas on top, pr below.
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 14))

    panels = [
        ("tas", axes[0], "(a) Temperature", "K"),
        ("pr", axes[1], "(b) Precipitation (wet days)", "mm day$^{-1}$"),
    ]

    styles = {b: _style(i) for i, b in enumerate(ROW_ORDER)}
    handles: dict[str, plt.Line2D] = {}

    for var, ax, label, unit in panels:
        lo_hi = [np.inf, -np.inf]

        for baseline in ROW_ORDER:
            entry = results[var].get(baseline)
            if entry is None:
                continue
            spread, rmse, count, m = entry

            if m < 2 or spread.size == 0:
                # single-member: no spread axis, show its RMSE as a guide line
                r = summary.loc[baseline, f"{var}_rmse"]
                if np.isfinite(r):
                    ax.axhline(r, ls=":", lw=1.0, alpha=0.55, zorder=1,
                               color=styles[baseline]["color"])
                    lo_hi[0] = min(lo_hi[0], r)
                    lo_hi[1] = max(lo_hi[1], r)
                continue

            good = np.isfinite(spread) & np.isfinite(rmse) & (count > 0)
            if not good.any():
                continue

            (h,) = ax.plot(spread[good], rmse[good], **styles[baseline])
            handles.setdefault(baseline, h)

            lo_hi[0] = min(lo_hi[0], np.nanmin(spread[good]), np.nanmin(rmse[good]))
            lo_hi[1] = max(lo_hi[1], np.nanmax(spread[good]), np.nanmax(rmse[good]))

        # 1:1 calibration line, drawn across the full data range
        if np.isfinite(lo_hi[0]) and lo_hi[1] > lo_hi[0]:
            pad = 0.05 * (lo_hi[1] - lo_hi[0])
            lo, hi = max(lo_hi[0] - pad, 0.0), lo_hi[1] + pad
            ax.plot([lo, hi], [lo, hi], color="black", lw=1.3, ls="--", zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.text(0.97, 0.03, "1:1", transform=ax.transAxes,
                    fontsize=9, ha="right", va="bottom", color="0.3")

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel(f"Ensemble spread [{unit}]", fontsize=11)
        ax.set_ylabel(f"RMSE of ensemble mean [{unit}]", fontsize=11)
        ax.grid(True, alpha=0.25, ls=":")

        rows = [
            f"{b[:28]:<28s}{summary.loc[b, f'{var}_SSR']:5.2f}"
            for b in ROW_ORDER
            if np.isfinite(summary.loc[b, f"{var}_SSR"])
        ]
        if rows:
            ax.text(
                0.02, 0.98, "SSR (aggregate)\n" + "\n".join(rows),
                transform=ax.transAxes, fontsize=7, family="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          alpha=0.85, ec="0.8"),
            )

    fig.legend(
        list(handles.values()), list(handles.keys()),
        loc="lower center", ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01), handlelength=2.8,
    )
    fig.suptitle(
        "Spread–skill diagram · above 1:1 = underdispersive, "
        "below = overdispersive",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved → {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"))
    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--wet_threshold", type=float, default=WET_DAY_THRESHOLD)
    ap.add_argument("--out_png", default="Analysis/BCSR_Stats/Figures/spread_skill.png")
    ap.add_argument("--out_csv", default="Analysis/BCSR_Stats/Tables/spread_skill.csv")
    ap.add_argument("--verbose_loader", action="store_true")
    args = ap.parse_args()

    if args.eval_start > args.eval_end:
        raise ValueError("eval_start must be <= eval_end")

    loaded = load_all_pooled_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
        verbose=args.verbose_loader,
    )
    obs = _load_obs(args, loaded.hr_mask)

    cols = []
    for v in ("tas", "pr"):
        cols += [f"{v}_SSR", f"{v}_spread", f"{v}_rmse", f"{v}_m"]
    summary = pd.DataFrame(index=ROW_ORDER, columns=cols, dtype=float)

    results = {"tas": {}, "pr": {}}

    for var in ("tas", "pr"):
        wet = args.wet_threshold if var == "pr" else None
        available = loaded.data.get(var, {})

        for baseline in ROW_ORDER:
            pred = _resolve(available, baseline)
            if pred is None:
                print(f"[skip] {var:4s} | {baseline} (not found)")
                continue

            pred_e, obs_e = xr.align(pred, obs[var], join="inner")
            if pred_e.sizes.get("time", 0) == 0:
                print(f"[skip] {var:4s} | {baseline} (no overlapping time)")
                continue

            mdim = _sample_dim(pred_e)
            m = ensemble_size(pred_e, mdim)

            ssr, spread, rmse, _ = spread_skill(
                pred_e, obs_e, member_dim=mdim,
                wet_threshold=wet, return_components=True,
            )
            summary.loc[baseline, [f"{var}_SSR", f"{var}_spread",
                                   f"{var}_rmse", f"{var}_m"]] = [ssr, spread, rmse, m]

            bs, br, bc = spread_skill_binned(
                pred_e, obs_e, nbins=args.nbins,
                member_dim=mdim, wet_threshold=wet,
            )
            results[var][baseline] = (bs, br, bc, m)

            print(f"[ok] {var:4s} | {baseline:34s} | m={m:4d} | "
                  f"SSR={ssr:.3f} | spread={spread:.3f} | rmse={rmse:.3f}")
            if bc.size:
                print(f"       bin counts: {bc.tolist()}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, float_format="%.4f")
    print(f"[ok] wrote {out_csv}")

    _plot(Path(args.out_png), results, summary)


if __name__ == "__main__":
    main()