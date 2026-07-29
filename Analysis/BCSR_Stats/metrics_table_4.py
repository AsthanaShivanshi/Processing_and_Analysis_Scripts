from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from load_medians import ROW_ORDER, MASK_LEVEL, load_all_medians_masked, apply_mask_exact
from ralsd import ralsd

from rmse import _sanitize_obs, _subset_years , rmse_spatiotemporal

from LHD import lhd


def main():
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    ds_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    p = argparse.ArgumentParser()
    p.add_argument("--ens_root", default=str(base / "GCM_pipeline/ALP-FINEv1.0/Ensmedians"))
    p.add_argument("--obs_tas_file", default=str(ds_root / "TabsD_step1_latlon.nc"))
    p.add_argument("--obs_pr_file", default=str(ds_root / "RhiresD_step1_latlon.nc"))
    p.add_argument("--obs_tas_var", default="TabsD")
    p.add_argument("--obs_pr_var", default="RhiresD")
    p.add_argument("--mask_hr_file", default=str(ds_root / "Swiss_Mask_HR.nc"))
    p.add_argument("--mask_lr_file", default=str(ds_root / "Swiss_Mask_LR.nc"))
    p.add_argument("--mask_hr_var", default="TabsD")
    p.add_argument("--mask_lr_var", default="TabsD")
    p.add_argument("--eval_start", type=int, default=2015)
    p.add_argument("--eval_end", type=int, default=2023)
    p.add_argument("--metrics_csv", default="Analysis/BCSR_Stats/Tables/distributional_dist_bcsr_Table_4.csv")
    args = p.parse_args()

    loaded = load_all_medians_masked(
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_lr_file=args.mask_lr_file,
        mask_hr_var=args.mask_hr_var,
        mask_lr_var=args.mask_lr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        variables=["tas", "pr"],
    )

    obs = {
        "tas": _sanitize_obs(xr.open_dataset(args.obs_tas_file)[args.obs_tas_var], "tas"),
        "pr": _sanitize_obs(xr.open_dataset(args.obs_pr_file)[args.obs_pr_var], "pr"),
    }

    rows = []
    for var in ["tas", "pr"]:
        obs_eval_base = _subset_years(obs[var], args.eval_start, args.eval_end)

        for baseline in ROW_ORDER:
            pred = loaded.data[var].get(baseline)
            if pred is None:
                rows.append({"Baseline": baseline, "Variable": var, "RMSE": np.nan, "LHD": np.nan, "RALSD": np.nan})
                continue

            mask = loaded.lr_mask if MASK_LEVEL[baseline] == "lr" else loaded.hr_mask
            obs_eval = apply_mask_exact(obs_eval_base, mask)

            pred, obs_eval = xr.align(pred, obs_eval, join="inner")
            if pred.sizes.get("time", 0) == 0:
                rows.append({"Baseline": baseline, "Variable": var, "RMSE": np.nan, "LHD": np.nan, "RALSD": np.nan})
                continue

            rows.append(
                {
                    "Baseline": baseline,
                    "Variable": var,
                    "RMSE": rmse_spatiotemporal(pred, obs_eval),
                    "LHD": lhd(pred, obs_eval),
                    "RALSD": ralsd(pred, obs_eval),
                }
            )

    out = Path(args.metrics_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()