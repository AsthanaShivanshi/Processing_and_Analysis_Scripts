from __future__ import annotations

import argparse
from pathlib import Path

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
from spectrum import ralsd, stream_mean_isotropic_spectrum


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
    ap.add_argument(
        "--metrics_csv",
        default="Analysis/BCSR_Stats/Tables/metrics_rmse_pss_ralsd_pooled_2015_2023.csv",
    )
    args = ap.parse_args()

    loaded = load_all_pooled_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
    )

    with xr.open_dataset(args.obs_tas_file, chunks={"time": 365}) as ds_tas:
        obs_tas = _sanitize(ds_tas[args.obs_tas_var], "tas")
    with xr.open_dataset(args.obs_pr_file, chunks={"time": 365}) as ds_pr:
        obs_pr = _sanitize(ds_pr[args.obs_pr_var], "pr")

    obs_map = {"tas": obs_tas, "pr": obs_pr}

    table = pd.DataFrame(index=ROW_ORDER)

    for var in ("tas", "pr"):
        obs_eval = apply_mask_exact(
            _subset_years(obs_map[var], args.eval_start, args.eval_end),
            loaded.hr_mask,
        )

        k_ref, ps_ref, _ = stream_mean_isotropic_spectrum(
            obs_eval, loaded.hr_mask, batch_size=args.batch_size
        )

        rmse_col = f"RMSE_{var}"
        pss_col = f"PSS_{var}"
        ralsd_col = f"RALSD_{var}"

        table[rmse_col] = np.nan
        table[pss_col] = np.nan
        table[ralsd_col] = np.nan

        available = loaded.data.get(var, {})
        print(f"[info] {var} available baselines: {list(available.keys())}")

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
                pred_eval, ref_eval, sample_dim="auto"
            )
            table.loc[baseline, pss_col] = pss_gridwise_spatial_mean(
                pred_eval, ref_eval, nbins=args.nbins, sample_dim="auto"
            )




            k_mod, ps_mod, _ = stream_mean_isotropic_spectrum(
                pred, loaded.hr_mask, batch_size=args.batch_size
            )
            table.loc[baseline, ralsd_col] = ralsd(k_ref, ps_ref, k_mod, ps_mod)



    out = Path(args.metrics_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, float_format="%.6f")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()