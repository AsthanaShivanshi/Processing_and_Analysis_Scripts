import os
import numpy as np
import pandas as pd
import xarray as xr
from pysteps.verification.salscores import sal

SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Notes from Wernli et al. 2008 and pysteps docs:
# https://pysteps.readthedocs.io/en/latest/generated/pysteps.verification.salscores.sal.html
# thr_factor = 1/15 from the paper.

# Framewise SAL averaged over all timesteps.
# Skips totally dry frames (both obs and pred are zero).
# Returns S, A, L for valid frames only.


def subset_season(da, season):
    return da.where(da.time.dt.season == season, drop=True)


def _clip_negative_precip_da(da: xr.DataArray) -> xr.DataArray:
    # Force physically valid precipitation values.
    return da.clip(min=0).astype("float32")


def _sanitize_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(frame, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    arr[arr < 0] = 0.0
    return arr


def sal_frame(obs_frame, pred_frame, thr_factor=1 / 15, thr_quantile=0.95):
    obs_frame = _sanitize_frame(obs_frame)
    pred_frame = _sanitize_frame(pred_frame)

    if obs_frame.max() == 0 and pred_frame.max() == 0:
        return np.nan, np.nan, np.nan

    try:
        S, A, L = sal(pred_frame, obs_frame, thr_factor=thr_factor, thr_quantile=thr_quantile)

        if not (np.isfinite(S) and np.isfinite(A) and np.isfinite(L)):
            return np.nan, np.nan, np.nan

        # SAL theoretical bounds in pysteps:
        # S in [-2, 2], A in [-2, 2], L (L1+L2) in [0, 2]
        eps = 1e-6
        if not (-2 - eps <= S <= 2 + eps):
            return np.nan, np.nan, np.nan
        if not (-2 - eps <= A <= 2 + eps):
            return np.nan, np.nan, np.nan
        if not (0 - eps <= L <= 2 + eps):
            return np.nan, np.nan, np.nan

        return float(S), float(A), float(L)
    except Exception:
        return np.nan, np.nan, np.nan


def sal_timeseries(obs_arr, pred_arr, times=None, sample_id=None, thr_factor=1 / 15, thr_quantile=0.95):
    rows = []
    T = obs_arr.shape[0]

    for t in range(T):
        S, A, L = sal_frame(obs_arr[t], pred_arr[t], thr_factor=thr_factor, thr_quantile=thr_quantile)
        if np.isfinite(S) and np.isfinite(A) and np.isfinite(L):
            rows.append(
                {
                    "time": None if times is None else pd.to_datetime(times[t]),
                    "sample": sample_id,
                    "S": S,
                    "A": A,
                    "L": L,
                }
            )

    return rows


def sal_timeseries_ensemble(obs_arr, pred_arr, times=None, thr_factor=1 / 15, thr_quantile=0.95):


    raw_rows = []
    mean_rows = []
    median_rows = []

    T = obs_arr.shape[0]
    Nsamples = pred_arr.shape[1]

    for t in range(T):
        sample_values = []

        for s in range(Nsamples):
            S, A, L = sal_frame(obs_arr[t], pred_arr[t, s], thr_factor=thr_factor, thr_quantile=thr_quantile)
            if np.isfinite(S) and np.isfinite(A) and np.isfinite(L):
                raw_rows.append(
                    {
                        "time": None if times is None else pd.to_datetime(times[t]),
                        "sample": s,
                        "S": S,
                        "A": A,
                        "L": L,
                    }
                )
                sample_values.append((S, A, L))

        if sample_values:
            vals = np.asarray(sample_values, dtype=np.float32)

            mean_rows.append(
                {
                    "time": None if times is None else pd.to_datetime(times[t]),
                    "sample": "mean",
                    "S": float(np.nanmean(vals[:, 0])),
                    "A": float(np.nanmean(vals[:, 1])),
                    "L": float(np.nanmean(vals[:, 2])),
                }
            )

            median_rows.append(
                {
                    "time": None if times is None else pd.to_datetime(times[t]),
                    "sample": "median",
                    "S": float(np.nanmedian(vals[:, 0])),
                    "A": float(np.nanmedian(vals[:, 1])),
                    "L": float(np.nanmedian(vals[:, 2])),
                }
            )

    return raw_rows, mean_rows, median_rows


def _append_ensemble_rows(raw_rows, season, model_name, obs_s, ens_s):
    raw_e, mean_e, median_e = sal_timeseries_ensemble(
        obs_s.values,
        ens_s.values,
        times=obs_s.time.values,
    )

    for r in raw_e:
        r.update({"season": season, "model": model_name, "type": "ensemble_sample"})
    for r in mean_e:
        r.update({"season": season, "model": model_name, "type": "ensemble_mean"})
    for r in median_e:
        r.update({"season": season, "model": model_name, "type": "ensemble_median"})

    raw_rows.extend(raw_e)
    raw_rows.extend(mean_e)
    raw_rows.extend(median_e)


def _load_precip(path, var_name, swiss_mask):
    da = (
        xr.open_dataset(path)[var_name]
        .sel(time=slice("2015-01-01", "2023-12-31"))
        .where(swiss_mask)
        .load()
    )
    return _clip_negative_precip_da(da)


def main():
    out_dir = "Analysis/Paper_Stats"
    os.makedirs(out_dir, exist_ok=True)

    swiss_mask = xr.open_dataset(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"
    )["TabsD"].load()

    obs = _load_precip(
        "data_1971_2023/HR_files_full/RhiresD_1971_2023.nc",
        "RhiresD",
        swiss_mask,
    )

    bilinear = _load_precip(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc",
        "RhiresD",
        swiss_mask,
    )

    bicubic = _load_precip(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc",
        "RhiresD",
        swiss_mask,
    )

    unet = _load_precip(
        "../Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
        "precip",
        swiss_mask,
    )

    ddim_ens = _load_precip(
        "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
        "precip",
        swiss_mask,
    )

    cfm_ens = _load_precip(
        "../Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
        "precip",
        swiss_mask,
    )

    models_det = {
        "Bilinear": bilinear,
        "Bicubic": bicubic,
        "UNet": unet,
    }

    raw_rows = []

    for season in SEASONS:
        obs_s = subset_season(obs, season).astype("float32")

        # Deterministic models
        for name, pred in models_det.items():
            pred_s = subset_season(pred, season).astype("float32")
            rows = sal_timeseries(obs_s.values, pred_s.values, times=obs_s.time.values)
            for r in rows:
                r.update({"season": season, "model": name, "type": "deterministic"})
            raw_rows.extend(rows)

        # DDIM ensemble (sample / mean / median)
        ddim_s = subset_season(ddim_ens, season).astype("float32")
        _append_ensemble_rows(raw_rows, season, "DDIM", obs_s, ddim_s)

        # CFM ensemble (sample / mean / median) - same logic as DDIM
        cfm_s = subset_season(cfm_ens, season).astype("float32")
        _append_ensemble_rows(raw_rows, season, "CFM", obs_s, cfm_s)

    sal_raw_df = pd.DataFrame(raw_rows)
    sal_raw_df.to_csv(f"{out_dir}/SAL_precip_seasonal_daily_values.csv", index=False)

    print(sal_raw_df.head().to_string(index=False))


if __name__ == "__main__":
    main()