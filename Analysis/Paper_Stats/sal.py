import xarray as xr
from pysteps.verification.salscores import sal
import numpy as np
from plotstyle import apply_paper_style
apply_paper_style()
import pandas as pd
from tqdm.auto import tqdm as tqdm_auto



#Notes from Wernli et al. 2008: and pysteps lib..https://pysteps.readthedocs.io/en/latest/generated/pysteps.verification.salscores.sal.html $$$$


#thr_factor = 1/15 from the paper 

    # Framewise SAL averaged over all timesteps — deterministic.
    # Skips totally dry frames (both obs and pred are zero).
    # Returns mean S, A, L over all valid frames.
    # S: size and shape of precipitation objects
    # A:  domain-average rainfall bias
    # L: centre of mass displacement


Swiss_Mask_HR = xr.open_dataset("../Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc")["TabsD"]

test_precip_target = xr.open_dataset("data_1971_2023/HR_files_full/RhiresD_1971_2023.nc")["RhiresD"].sel(
    time=slice("2015-01-01", "2023-12-31")
).where(Swiss_Mask_HR)

test_precip_bilinear = xr.open_dataset("../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc")["RhiresD"].sel(
    time=slice("2015-01-01", "2023-12-31")
).where(Swiss_Mask_HR)

test_precip_bicubic = xr.open_dataset("../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc")["RhiresD"].sel(
    time=slice("2015-01-01", "2023-12-31")
).where(Swiss_Mask_HR)

test_precip_unet = xr.open_dataset("../Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc")["precip"].sel(
    time=slice("2015-01-01", "2023-12-31")
).where(Swiss_Mask_HR)

test_precip_ddim_median = xr.open_dataset("../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0_median.nc")["precip"].sel(
    time=slice("2015-01-01", "2023-12-31")
).where(Swiss_Mask_HR)

test_precip_ddim_ensemble = xr.open_dataset(
    "../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc"
)["precip"].sel(time=slice("2015-01-01", "2023-12-31")).where(Swiss_Mask_HR)

sample_dim = "sample"  

test_precip_ddim_mean = test_precip_ddim_ensemble.mean(dim=sample_dim)

# ------------------------------------------------------------------ #

def get_best_member(obs_da, ens_da, sample_dim="sample"):
    T = obs_da.shape[0]
    n_members = ens_da.sizes[sample_dim]
    best_frames = []
    for t in tqdm_auto(range(T), desc="Selecting best member"):
        obs_f = obs_da.isel(time=t).values
        best_rmse, best_frame = np.inf, None
        for s in range(n_members):
            pred_f = ens_da.isel(time=t, **{sample_dim: s}).values
            rmse = np.sqrt(np.nanmean((pred_f - obs_f) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_frame = pred_f
        best_frames.append(best_frame)
    best_arr = np.stack(best_frames, axis=0)
    return xr.DataArray(best_arr, dims=obs_da.dims, coords=obs_da.coords)



test_precip_ddim_best = get_best_member(test_precip_target, test_precip_ddim_ensemble, sample_dim)

# ------------------------------------------------------------------ #

def sal_timeseries(obs_da, pred_da, thr_factor=1/15, thr_quantile=0.95):


    S_list, A_list, L_list = [], [], []
    T = obs_da.shape[0]
    for t in range(T):
        obs_frame  = np.nan_to_num(obs_da.isel(time=t).values.astype(float), nan=0.0)
        pred_frame = np.nan_to_num(pred_da.isel(time=t).values.astype(float), nan=0.0)
        if obs_frame.max() == 0 and pred_frame.max() == 0:
            continue
        try:
            S, A, L = sal(pred_frame, obs_frame, thr_factor=thr_factor, thr_quantile=thr_quantile)
            if np.isfinite(S) and np.isfinite(A) and np.isfinite(L):
                S_list.append(S)
                A_list.append(A)
                L_list.append(L)
        except Exception:
            continue
    return np.nanmean(S_list), np.nanmean(A_list), np.nanmean(L_list)


# ------------------------------------------------------------------ #
def sal_probabilistic(obs_da, ens_da, sample_dim="sample", thr_factor=1/15, thr_quantile=0.95):



    S_all, A_all, L_all = [], [], []
    T = obs_da.shape[0]
    n_members = ens_da.sizes[sample_dim]


    for t in tqdm_auto(range(T), desc="Probabilistic SAL", leave=False):
        obs_frame = np.nan_to_num(obs_da.isel(time=t).values.astype(float), nan=0.0)
        if obs_frame.max() == 0:
            continue
        for s in range(n_members):
            pred_frame = np.nan_to_num(
                ens_da.isel(time=t, **{sample_dim: s}).values.astype(float), nan=0.0
            )
            if pred_frame.max() == 0:
                continue
            try:
                S, A, L = sal(pred_frame, obs_frame, thr_factor=thr_factor, thr_quantile=thr_quantile)
                if np.isfinite(S) and np.isfinite(A) and np.isfinite(L):
                    S_all.append(S)
                    A_all.append(A)
                    L_all.append(L)
            except Exception:
                continue
    return np.nanmean(S_all), np.nanmean(A_all), np.nanmean(L_all)

# ------------------------------------------------------------------ #


models_precip = {
    "Bilinear":         test_precip_bilinear,
    "Bicubic":          test_precip_bicubic,
    "UNet":             test_precip_unet,
    "DDIM_median":      test_precip_ddim_median,
    "DDIM_mean":        test_precip_ddim_mean,
    "DDIM_best_member": test_precip_ddim_best,
}

sal_results = []

for name, pred in tqdm_auto(models_precip.items(), desc="SAL deterministic"):
    print(f"\nComputing SAL for {name}...")
    S, A, L = sal_timeseries(test_precip_target, pred)
    sal_results.append({"model": name, "type": "deterministic", "S": S, "A": A, "L": L})
    print(f"  S={S:.4f}  A={A:.4f}  L={L:.4f}")



print("\nComputing Probabilistic SAL (all members × all timesteps)...")


S, A, L = sal_probabilistic(test_precip_target, test_precip_ddim_ensemble, sample_dim)
sal_results.append({"model": "DDIM_probabilistic", "type": "probabilistic", "S": S, "A": A, "L": L})
print(f"  S={S:.4f}  A={A:.4f}  L={L:.4f}")



sal_df = pd.DataFrame(sal_results)
print("\n", sal_df.to_string(index=False))
sal_df.to_csv("Analysis/Paper_Stats/SAL_precip.csv", index=False)
print("\nSaved to Analysis/Paper_Stats/SAL_precip.csv")