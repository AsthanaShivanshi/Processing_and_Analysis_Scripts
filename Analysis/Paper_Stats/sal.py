import os
import numpy as np
import pandas as pd
import xarray as xr
from pysteps.verification.salscores import sal

SEASONS = ["DJF", "MAM", "JJA", "SON"]


#Notes from Wernli et al. 2008: and pysteps lib..https://pysteps.readthedocs.io/en/latest/generated/pysteps.verification.salscores.sal.html $$$$


#thr_factor = 1/15 from the paper 

    # Framewise SAL averaged over all timesteps — deterministic.
    # Skips totally dry frames (both obs and pred are zero).
    # Returns mean S, A, L over all valid frames.
    # S: size and shape of precipitation objects
    # A:  domain-average rainfall bias
    # L: centre of mass displacement



def subset_season(da, season):
    return da.where(da.time.dt.season == season, drop=True)


def sal_frame(obs_frame, pred_frame, thr_factor=1 / 15, thr_quantile=0.95):
    obs_frame = np.nan_to_num(np.asarray(obs_frame, dtype=np.float32), nan=0.0)
    pred_frame = np.nan_to_num(np.asarray(pred_frame, dtype=np.float32), nan=0.0)

    if obs_frame.max() == 0 and pred_frame.max() == 0:
        return np.nan, np.nan, np.nan

    try:
        S, A, L = sal(pred_frame, obs_frame, thr_factor=thr_factor, thr_quantile=thr_quantile)
        if np.isfinite(S) and np.isfinite(A) and np.isfinite(L):
            return float(S), float(A), float(L)
    except Exception:
        pass

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
    N = pred_arr.shape[1]

    for t in range(T):
        sample_values = []

        for s in range(N):
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


def main():
    out_dir = "Analysis/Paper_Stats"
    os.makedirs(out_dir, exist_ok=True)

    swiss_mask = xr.open_dataset(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"
    )["TabsD"].load()

    obs = xr.open_dataset("data_1971_2023/HR_files_full/RhiresD_1971_2023.nc")["RhiresD"].sel(
        time=slice("2015-01-01", "2023-12-31")
    ).where(swiss_mask).load().astype("float32")

    bilinear = xr.open_dataset(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc"
    )["RhiresD"].sel(time=slice("2015-01-01", "2023-12-31")).where(swiss_mask).load().astype("float32")

    bicubic = xr.open_dataset(
        "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc"
    )["RhiresD"].sel(time=slice("2015-01-01", "2023-12-31")).where(swiss_mask).load().astype("float32")

    unet = xr.open_dataset(
        "../Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc"
    )["precip"].sel(time=slice("2015-01-01", "2023-12-31")).where(swiss_mask).load().astype("float32")

    ddim_ens = xr.open_dataset(
        "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc"
    )["precip"].sel(time=slice("2015-01-01", "2023-12-31")).where(swiss_mask).load().astype("float32")

    models_det = {
        "Bilinear": bilinear,
        "Bicubic": bicubic,
        "UNet": unet,
    }

    raw_rows = []

    for season in SEASONS:
        obs_s = subset_season(obs, season).astype("float32")

        for name, pred in models_det.items():
            pred_s = subset_season(pred, season).astype("float32")
            rows = sal_timeseries(obs_s.values, pred_s.values, times=obs_s.time.values)
            for r in rows:
                r.update({"season": season, "model": name, "type": "deterministic"})
            raw_rows.extend(rows)

        ddim_s = subset_season(ddim_ens, season).astype("float32")
        raw_ddim, mean_ddim, median_ddim = sal_timeseries_ensemble(
            obs_s.values,
            ddim_s.values,
            times=obs_s.time.values,
        )

        for r in raw_ddim:
            r.update({"season": season, "model": "DDIM", "type": "ensemble_sample"})
        for r in mean_ddim:
            r.update({"season": season, "model": "DDIM", "type": "ensemble_mean"})
        for r in median_ddim:
            r.update({"season": season, "model": "DDIM", "type": "ensemble_median"})

        raw_rows.extend(raw_ddim)
        raw_rows.extend(mean_ddim)
        raw_rows.extend(median_ddim)

    sal_raw_df = pd.DataFrame(raw_rows)
    sal_raw_df.to_csv(f"{out_dir}/SAL_precip_seasonal_daily_values.csv", index=False)

    print(sal_raw_df.head().to_string(index=False))


if __name__ == "__main__":
    main()