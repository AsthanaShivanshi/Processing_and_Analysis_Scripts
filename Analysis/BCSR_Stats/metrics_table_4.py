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


def _plot_rapsd_summary(
    out: Path,
    tas_ref: tuple[np.ndarray, np.ndarray],
    tas_spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
    pr_ref: tuple[np.ndarray, np.ndarray],
    pr_spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cmap = plt.get_cmap("tab20")
    linestyles = ["-", "--", ":", "-."]

    panel_data = [
        ("tas", axes[0, 0], axes[1, 0], tas_ref, tas_spectra),
        ("pr", axes[0, 1], axes[1, 1], pr_ref, pr_spectra),
    ]

    for var, ax_psd, ax_ratio, (k_ref, ps_ref), spectra in panel_data:
        if k_ref.size:
            ax_psd.loglog(k_ref, ps_ref, color="black", lw=2.2, label="OBS")

        ax_ratio.axhline(1.0, color="black", lw=1.3, ls="--", zorder=3)

        for i, (baseline, (k_mod, ps_mod, score)) in enumerate(spectra.items()):
            color = cmap(i % 20)
            ls = linestyles[i % len(linestyles)]
            label = f"{baseline} | RALSD={score:.4f}" if np.isfinite(score) else baseline

            if k_mod.size:
                ax_psd.loglog(k_mod, ps_mod, color=color, lw=1.4, ls=ls, label=label)

            if k_ref.size and k_mod.size:
                n = min(k_ref.size, k_mod.size)
                k_grid = k_ref[:n]
                ps_ref_i = np.interp(k_grid, k_ref, ps_ref)
                ps_mod_i = np.interp(k_grid, k_mod, ps_mod)
                ratio = ps_mod_i / np.maximum(ps_ref_i, EPS)
                ax_ratio.plot(k_grid, ratio, color=color, lw=1.4, ls=ls, label=baseline)

        ax_psd.set_title(f"RAPSD — {var}")
        ax_psd.set_xlabel("Wavenumber")
        ax_psd.set_ylabel("Power")
        ax_psd.grid(True, which="both", alpha=0.25)
        ax_psd.legend(fontsize=7, frameon=False)

        ax_ratio.set_title(f"Spectral ratio — {var}")
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("Wavenumber k (pixel units)")
        ax_ratio.set_ylabel("pred / obs")
        ax_ratio.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved RAPSD summary → {out}")


def _compute_ralsd_for_var(args, loaded, obs_eval: xr.DataArray, var: str):
    table = pd.DataFrame(index=ROW_ORDER, columns=["RALSD"], dtype=float)
    available = loaded.data.get(var, {})
    k_ref, ps_ref = rapsd(obs_eval, time_dim="time")

    spectra: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for baseline in ROW_ORDER:
        pred = _resolve_baseline_da(available, baseline)
        if pred is None:
            continue

        pred_eval, _ = xr.align(pred, obs_eval, join="inner")
        if pred_eval.sizes.get("time", 0) == 0:
            continue

        sample_dim = _resolve_sample_dim(pred_eval)
        k_mod, ps_mod = rapsd(pred_eval, time_dim="time", sample_dim=sample_dim)
        score = ralsd(k_ref, ps_ref, k_mod, ps_mod)

        table.loc[baseline, "RALSD"] = score
        spectra[baseline] = (k_mod, ps_mod, score)

    return table, (k_ref, ps_ref), spectra


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
        tas_ralsd, tas_ref, tas_spectra = _compute_ralsd_for_var(args, loaded_pooled, obs_map["tas"], "tas")
        pr_ralsd, pr_ref, pr_spectra = _compute_ralsd_for_var(args, loaded_pooled, obs_map["pr"], "pr")

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
        )
        return


if __name__ == "__main__":
    main()