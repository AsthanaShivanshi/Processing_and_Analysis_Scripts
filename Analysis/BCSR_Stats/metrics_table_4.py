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
from RAPSD import ralsd, stream_mean_isotropic_spectrum


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


def _load_obs(args, loaded):
    # eager load inside context
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


def _plot_rapsd_all_baselines(
    var: str,
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    scores: dict[str, float],
    out_file: Path,
    eval_start: int,
    eval_end: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    if k_ref.size > 0 and ps_ref.size > 0:
        ax.loglog(k_ref, ps_ref, color="black", lw=2.4, label="MCH (obs)")

    for baseline in ROW_ORDER:
        if baseline not in spectra:
            continue
        k, ps = spectra[baseline]
        sc = scores.get(baseline, np.nan)
        label = f"{baseline} | RALSD={sc:.3f}" if np.isfinite(sc) else baseline
        ax.loglog(k, ps, lw=1.2, alpha=0.95, label=label)

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"RAPSD comparison ({var}, {eval_start}-{eval_end})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--plot_ralsd", action="store_true")
    ap.add_argument("--ralsd_plot_dir", default="Analysis/BCSR_Stats/Figures/RALSD")
    ap.add_argument(
        "--metrics_csv",
        default="Analysis/BCSR_Stats/Tables/metrics_rmse_pss_ralsd_pooled_2015_2023.csv",
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

    # One combined table for tas + pr and all 3 metrics
    table = pd.DataFrame(
        index=ROW_ORDER,
        columns=["RMSE_tas", "PSS_tas", "RALSD_tas", "RMSE_pr", "PSS_pr", "RALSD_pr"],
        dtype=float,
    )

    for var in ("tas", "pr"):
        obs_eval = obs_map[var]
        available = loaded.data.get(var, {})
        print(f"[info] {var} available baselines: {list(available.keys())}")

        rmse_col = f"RMSE_{var}"
        pss_col = f"PSS_{var}"
        ralsd_col = f"RALSD_{var}"

        k_ref, ps_ref, _ = stream_mean_isotropic_spectrum(
            obs_eval, loaded.hr_mask, batch_size=args.batch_size
        )

        spectra_all: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        scores_all: dict[str, float] = {}

        for baseline in ROW_ORDER:
            pred = _resolve_baseline_da(available, baseline)
            if pred is None:
                print(f"[warn] missing {var}: {baseline}")
                continue

            pred_eval, ref_eval = xr.align(pred, obs_eval, join="inner")
            if pred_eval.sizes.get("time", 0) == 0:
                print(f"[warn] no time overlap for {var}: {baseline}")
                continue

            table.loc[baseline, rmse_col] = rmse_gridwise_spatial_mean(
                pred_eval, ref_eval, sample_dim="auto", mode="per_sample_mean"
            )
            table.loc[baseline, pss_col] = pss_gridwise_spatial_mean(
                pred_eval, ref_eval, nbins=args.nbins, sample_dim="auto", mode="pooled"
            )

            k_mod, ps_mod, _ = stream_mean_isotropic_spectrum(
                pred_eval, loaded.hr_mask, batch_size=args.batch_size
            )
            sc = ralsd(k_ref, ps_ref, k_mod, ps_mod)
            table.loc[baseline, ralsd_col] = sc

            if args.plot_ralsd and k_mod.size > 0 and ps_mod.size > 0:
                spectra_all[baseline] = (k_mod, ps_mod)
                scores_all[baseline] = sc

        if args.plot_ralsd and k_ref.size > 0 and ps_ref.size > 0:
            out_plot = Path(args.ralsd_plot_dir) / f"RAPSD_all_baselines_{var}_{args.eval_start}_{args.eval_end}.png"
            _plot_rapsd_all_baselines(
                var=var,
                k_ref=k_ref,
                ps_ref=ps_ref,
                spectra=spectra_all,
                scores=scores_all,
                out_file=out_plot,
                eval_start=args.eval_start,
                eval_end=args.eval_end,
            )
            print(f"[ok] wrote {out_plot}")

    out = Path(args.metrics_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, float_format="%.6f")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()