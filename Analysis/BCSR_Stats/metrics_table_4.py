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
from PSS import pss_gridwise_spatial_mean
from rmse import rmse_gridwise_spatial_mean
from RAPSD import ralsd, rapsd, EPS


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

    cands = [(k, v) for k, v in data_map.items() if target in _norm_name(k) or _norm_name(k) in target]
    if not cands:
        return None
    cands.sort(key=lambda kv: len(_norm_name(kv[0])), reverse=True)
    return cands[0][1]


def _build_table() -> pd.DataFrame:
    return pd.DataFrame(
        index=ROW_ORDER,
        columns=["RMSE_tas", "PSS_tas", "RALSD_tas", "RMSE_pr", "PSS_pr", "RALSD_pr"],
        dtype=float,
    )


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


def _compute_rmse_pss(args, loaded, obs_map) -> pd.DataFrame:
    table = _build_table()

    for var in ("tas", "pr"):
        obs_eval = obs_map[var]
        available = loaded.data.get(var, {})
        rmse_col = f"RMSE_{var}"
        pss_col = f"PSS_{var}"

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                continue

            pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                continue

            table.loc[baseline, rmse_col] = rmse_gridwise_spatial_mean(
                pred_eval, ref_eval, sample_dim="auto", mode="per_sample_mean"
            )
            table.loc[baseline, pss_col] = pss_gridwise_spatial_mean(
                pred_eval, ref_eval, nbins=args.nbins, sample_dim="auto", mode="pooled"
            )

    return table


def _plot_rapsd(
    var: str,
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if k_ref.size:
        ax.loglog(k_ref, ps_ref, color="black", lw=2.0, label="OBS")

    for baseline, (k_mod, ps_mod, score) in spectra.items():
        label = f"{baseline} | RALSD={score:.4f}" if np.isfinite(score) else baseline
        if k_mod.size:
            ax.loglog(k_mod, ps_mod, lw=1.2, label=label)

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"RAPSD comparison — {var}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved RAPSD plot → {out}")


def _plot_spectral_ratio(
    var: str,
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out: Path,
) -> None:
    """
    Plot spectral ratio (pred / obs) vs wavenumber for all baselines.
    ratio > 1  →  pred has more energy at that scale (over-energetic)
    ratio < 1  →  pred is smoother than obs at that scale
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="ratio = 1 (perfect)", zorder=3)

    for baseline, (k_mod, ps_mod, _) in spectra.items():
        if k_mod.size == 0 or k_ref.size == 0:
            continue

        # Interpolate both onto the shorter common k grid
        n = min(k_ref.size, k_mod.size)
        k_grid   = k_ref[:n]
        ps_ref_i = np.interp(k_grid, k_ref, ps_ref)
        ps_mod_i = np.interp(k_grid, k_mod, ps_mod)

        ratio = ps_mod_i / np.maximum(ps_ref_i, EPS)
        ax.plot(k_grid, ratio, lw=1.5, label=baseline)

    ax.set_xscale("log")
    ax.set_xlabel("Wavenumber k (pixel units)")
    ax.set_ylabel("Spectral ratio (pred / obs)")
    ax.set_title(f"Time-averaged Spectral Ratio — {var}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved spectral ratio plot → {out}")



def _compute_ralsd(args, loaded, obs_map) -> pd.DataFrame:
    table = _build_table()

    for var in ("tas", "pr"):
        obs_eval = obs_map[var]
        available = loaded.data.get(var, {})
        ralsd_col = f"RALSD_{var}"

        k_ref, ps_ref = rapsd(obs_eval, time_dim="time")
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                continue

            pred_eval, _ = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                continue

            # Use member dim if present (pooled ensemble)
            sample_dim = "member" if "member" in pred_eval.dims else None
            k_mod, ps_mod = rapsd(pred_eval, time_dim="time", sample_dim=sample_dim)
            score = ralsd(k_ref, ps_ref, k_mod, ps_mod)

            table.loc[baseline, ralsd_col] = score
            spectra[baseline] = (k_mod, ps_mod, score)

        if spectra:
            plot_dir = Path(args.rapsd_plot_dir)

            # 1 — power spectra
            _plot_rapsd(
                var=var,
                k_ref=k_ref,
                ps_ref=ps_ref,
                spectra=spectra,
                out=plot_dir / f"rapsd_{var}.png",
            )

            # 2 — spectral ratio
            _plot_spectral_ratio(
                var=var,
                k_ref=k_ref,
                ps_ref=ps_ref,
                spectra=spectra,
                out=plot_dir / f"spectral_ratio_{var}.png",
            )

    return table


def _write_table(table: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    table = table.reindex(ROW_ORDER)
    table.to_csv(out, float_format="%.6f")


def _load_partial_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _build_table()
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    table = table.reindex(ROW_ORDER)
    return table


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rmse_pss", "ralsd", "merge"], required=True)
    ap.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"))
    ap.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    ap.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    ap.add_argument("--obs_tas_var", default="TabsD")
    ap.add_argument("--obs_pr_var", default="RhiresD")
    ap.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    ap.add_argument("--mask_hr_var", default="TabsD")
    ap.add_argument("--eval_start", type=int, default=2015)
    ap.add_argument("--eval_end", type=int, default=2023)
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument(
        "--rmse_pss_csv",
        default="Analysis/BCSR_Stats/Tables/metrics_rmse_pss_partial.csv",
    )
    ap.add_argument(
        "--ralsd_csv",
        default="Analysis/BCSR_Stats/Tables/metrics_ralsd_partial.csv",
    )
    ap.add_argument(
        "--metrics_csv",
        default="Analysis/BCSR_Stats/Tables/metrics_rmse_pss_ralsd_pooled_2015_2023.csv",
    )
    ap.add_argument(
        "--rapsd_plot_dir",
        default="Analysis/BCSR_Stats/Plots",
    )
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

    if loaded.missing.get("tas"):
        print(f"[warn] missing pooled tas baselines: {loaded.missing['tas']}")
    if loaded.missing.get("pr"):
        print(f"[warn] missing pooled pr baselines: {loaded.missing['pr']}")

    obs_map = _load_obs(args, loaded)

    if args.mode == "rmse_pss":
        table = _compute_rmse_pss(args, loaded, obs_map)
        out = Path(args.rmse_pss_csv)
        _write_table(table, out)
        print(f"[ok] wrote {out}")
        return

    if args.mode == "ralsd":
        table = _compute_ralsd(args, loaded, obs_map)
        out = Path(args.ralsd_csv)
        _write_table(table, out)
        print(f"[ok] wrote {out}")
        return

    rmse_pss_table = _load_partial_table(Path(args.rmse_pss_csv))
    ralsd_table = _load_partial_table(Path(args.ralsd_csv))

    merged = _build_table()
    for col in merged.columns:
        if col.startswith("RMSE") or col.startswith("PSS"):
            merged[col] = rmse_pss_table[col]
        elif col.startswith("RALSD"):
            merged[col] = ralsd_table[col]

    out = Path(args.metrics_csv)
    _write_table(merged, out)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()