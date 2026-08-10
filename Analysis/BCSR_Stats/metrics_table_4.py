from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config
from PSS import pss_gridwise_spatial_mean
from RAPSD import EPS, ralsd, rapsd
from load_means import load_all_means_masked
from load_pooled import (
    ROW_ORDER,
    _sanitize,
    _subset_years,
    apply_mask_exact,
    load_all_pooled_masked,
)
from rmse import rmse_gridwise_spatial_mean

SAMPLE_DIMS = ("member", "sample", "samples")


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


def _compute_rmse_for_var(args, loaded, obs_eval: xr.DataArray, var: str) -> pd.DataFrame:
    table = pd.DataFrame(index=ROW_ORDER, columns=["RMSE_pooled", "RMSE_mean"], dtype=float)
    available = loaded.data.get(var, {})

    for suffix, use_sample_dim in (("pooled", True), ("mean", False)):
        rmse_col = f"RMSE_{suffix}"

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                continue

            pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                continue

            sample_dim = _resolve_sample_dim(pred_eval) if use_sample_dim else None
            table.loc[baseline, rmse_col] = rmse_gridwise_spatial_mean(
                pred_eval,
                ref_eval,
                sample_dim=sample_dim if sample_dim is not None else None,
                mode="per_sample_mean",
            )

    return table


def _compute_pss_for_var(args, loaded, obs_eval: xr.DataArray, var: str) -> pd.DataFrame:
    table = pd.DataFrame(index=ROW_ORDER, columns=["PSS_pooled", "PSS_mean"], dtype=float)
    available = loaded.data.get(var, {})

    for suffix, use_sample_dim in (("pooled", True), ("mean", False)):
        pss_col = f"PSS_{suffix}"

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                continue

            pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                continue

            sample_dim = _resolve_sample_dim(pred_eval) if use_sample_dim else None
            table.loc[baseline, pss_col] = pss_gridwise_spatial_mean(
                pred_eval,
                ref_eval,
                nbins=args.nbins,
                sample_dim=sample_dim if sample_dim is not None else None,
                mode="pooled",
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
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="ratio = 1 (perfect)", zorder=3)

    for baseline, (k_mod, ps_mod, _) in spectra.items():
        if k_mod.size == 0 or k_ref.size == 0:
            continue

        n = min(k_ref.size, k_mod.size)
        k_grid = k_ref[:n]
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


def _compute_ralsd_for_var(args, loaded, obs_eval: xr.DataArray, var: str) -> pd.DataFrame:
    table = pd.DataFrame(index=ROW_ORDER, columns=["RALSD_pooled", "RALSD_mean"], dtype=float)
    available = loaded.data.get(var, {})
    k_ref, ps_ref = rapsd(obs_eval, time_dim="time")

    for suffix, use_sample_dim in (("pooled", True), ("mean", False)):
        ralsd_col = f"RALSD_{suffix}"
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                continue

            pred_eval, _ = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                continue

            sample_dim = _resolve_sample_dim(pred_eval) if use_sample_dim else None
            k_mod, ps_mod = rapsd(pred_eval, time_dim="time", sample_dim=sample_dim)
            score = ralsd(k_ref, ps_ref, k_mod, ps_mod)

            table.loc[baseline, ralsd_col] = score
            spectra[baseline] = (k_mod, ps_mod, score)

        if spectra:
            plot_dir = Path(args.rapsd_plot_dir)
            _plot_rapsd(
                var=f"{var}_{suffix}",
                k_ref=k_ref,
                ps_ref=ps_ref,
                spectra=spectra,
                out=plot_dir / f"rapsd_{var}_{suffix}.png",
            )
            _plot_spectral_ratio(
                var=f"{var}_{suffix}",
                k_ref=k_ref,
                ps_ref=ps_ref,
                spectra=spectra,
                out=plot_dir / f"spectral_ratio_{var}_{suffix}.png",
            )

    return table


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
    ap.add_argument("--mode", choices=["rmse", "pss", "rapsd"], required=True)
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
    ap.add_argument("--rmse_csv", default="Analysis/BCSR_Stats/Tables/metrics_rmse.csv")
    ap.add_argument("--pss_csv", default="Analysis/BCSR_Stats/Tables/metrics_pss.csv")
    ap.add_argument("--ralsd_csv", default="Analysis/BCSR_Stats/Tables/metrics_rapsd.csv")
    ap.add_argument("--rapsd_plot_dir", default="Analysis/BCSR_Stats/Figures")
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

    if args.mode == "rmse":
        tas_pooled = _compute_rmse_for_var(args, loaded_pooled, obs_map["tas"], "tas")
        tas_mean = _compute_rmse_for_var(args, loaded_means, obs_map["tas"], "tas")
        pr_pooled = _compute_rmse_for_var(args, loaded_pooled, obs_map["pr"], "pr")
        pr_mean = _compute_rmse_for_var(args, loaded_means, obs_map["pr"], "pr")

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_RMSE_pooled"] = tas_pooled["RMSE_pooled"]
        out["tas_RMSE_mean"] = tas_mean["RMSE_mean"]
        out["pr_RMSE_pooled"] = pr_pooled["RMSE_pooled"]
        out["pr_RMSE_mean"] = pr_mean["RMSE_mean"]
        _write_table(out, Path(args.rmse_csv))
        print(f"[ok] wrote {args.rmse_csv}")
        return

    if args.mode == "pss":
        tas_pooled = _compute_pss_for_var(args, loaded_pooled, obs_map["tas"], "tas")
        tas_mean = _compute_pss_for_var(args, loaded_means, obs_map["tas"], "tas")
        pr_pooled = _compute_pss_for_var(args, loaded_pooled, obs_map["pr"], "pr")
        pr_mean = _compute_pss_for_var(args, loaded_means, obs_map["pr"], "pr")

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_PSS_pooled"] = tas_pooled["PSS_pooled"]
        out["tas_PSS_mean"] = tas_mean["PSS_mean"]
        out["pr_PSS_pooled"] = pr_pooled["PSS_pooled"]
        out["pr_PSS_mean"] = pr_mean["PSS_mean"]
        _write_table(out, Path(args.pss_csv))
        print(f"[ok] wrote {args.pss_csv}")
        return

    if args.mode == "rapsd":
        tas_pooled = _compute_ralsd_for_var(args, loaded_pooled, obs_map["tas"], "tas")
        tas_mean = _compute_ralsd_for_var(args, loaded_means, obs_map["tas"], "tas")
        pr_pooled = _compute_ralsd_for_var(args, loaded_pooled, obs_map["pr"], "pr")
        pr_mean = _compute_ralsd_for_var(args, loaded_means, obs_map["pr"], "pr")

        out = pd.DataFrame(index=ROW_ORDER)
        out["tas_RALSD_pooled"] = tas_pooled["RALSD_pooled"]
        out["tas_RALSD_mean"] = tas_mean["RALSD_mean"]
        out["pr_RALSD_pooled"] = pr_pooled["RALSD_pooled"]
        out["pr_RALSD_mean"] = pr_mean["RALSD_mean"]
        _write_table(out, Path(args.ralsd_csv))
        print(f"[ok] wrote {args.ralsd_csv}")
        return


if __name__ == "__main__":
    main()