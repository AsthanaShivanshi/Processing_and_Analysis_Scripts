import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Output directory for plots
BASE_DIR= Path(os.environ["BASE_DIR"])
OUTDIR = BASE_DIR / "sasthana"/"Downscaling"/"Processing_and_Analysis_Scripts"/"Prelim_Stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

cases = [
    {
        "obs_path": BASE_DIR/"sasthana"/"Downscaling"/"Processing_and_Analysis_Scripts"/"data_1971_2023"/"HR_files_full"/"TmaxD_1971_2023.nc",
        "rec_path": BASE_DIR/"raw_data"/"Reconstruction_UniBern_1763_2020"/"tmax_1763_2020.nc",
        "obs_var": "TmaxD",
        "rec_var": "tmax",
        "title": "Max daily temperature"
    },
    {
        "obs_path": BASE_DIR/"sasthana"/"Downscaling"/"Processing_and_Analysis_Scripts"/"data_1971_2023"/"HR_files_full"/"RhiresD_1971_2023.nc",
        "rec_path": BASE_DIR/"raw_data"/"Reconstruction_UniBern_1763_2020"/"precip_1763_2020.nc",
        "obs_var": "RhiresD",
        "rec_var": "precip",
        "title": "Daily precip sums"
    },
    {
        "obs_path": BASE_DIR/"sasthana"/"Downscaling"/"Processing_and_Analysis_Scripts"/"data_1971_2023"/"HR_files_full"/"TminD_1971_2023.nc",
        "rec_path": BASE_DIR/"raw_data"/"Reconstruction_UniBern_1763_2020"/"tmin_1763_2020.nc",
        "obs_var": "TminD",
        "rec_var": "tmin",
        "title": "Minimum daily temperature"
    },
    {
        "obs_path": BASE_DIR/"sasthana"/"Downscaling"/"Processing_and_Analysis_Scripts"/"data_1971_2023"/"HR_files_full"/"TabsD_1971_2023.nc",
        "rec_path": BASE_DIR/"raw_data"/"Reconstruction_UniBern_1763_2020"/"temp_1763_2020.nc",
        "obs_var": "TabsD",
        "rec_var": "temp",
        "title": "Average daily temperature"
    }
]

years_range_1 = range(1971, 2011)
years_range_2 = range(1771, 2011)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, case in enumerate(cases):
    ds1 = xr.open_dataset(case["obs_path"], chunks={"time": 20})
    ds2 = xr.open_dataset(case["rec_path"], chunks={"time": 20})

    # Observed 1971–2010
    all_obs = []
    for year in years_range_1:
        d = ds1[case["obs_var"]].sel(time=slice(f'{year}-01-01', f'{year}-12-31')).values.flatten()
        d = d[~np.isnan(d)]
        all_obs.append(d)
    data_obs = np.concatenate(all_obs)

    # Recon 1971–2010
    all_rec_1971_2010 = []
    for year in years_range_1:
        d = ds2[case["rec_var"]].sel(time=slice(f'{year}-01-01', f'{year}-12-31')).values.flatten()
        d = d[~np.isnan(d)]
        all_rec_1971_2010.append(d)
    data_rec_1971_2010 = np.concatenate(all_rec_1971_2010)

    # Reconstructed 1771–2010
    all_rec_1771_2010 = []
    for year in years_range_2:
        d = ds2[case["rec_var"]].sel(time=slice(f'{year}-01-01', f'{year}-12-31')).values.flatten()
        d = d[~np.isnan(d)]
        all_rec_1771_2010.append(d)
    data_rec_1771_2010 = np.concatenate(all_rec_1771_2010)

    axs[i].hist(data_obs, bins=50, alpha=0.5, label='Observed 1971–2010', color="blue")
    axs[i].hist(data_rec_1971_2010, bins=50, alpha=0.5, label='Reconstructed 1971–2010', color="red")
    axs[i].hist(data_rec_1771_2010, bins=50, alpha=0.5, label='Reconstructed 1771–2010', color="green")
    axs[i].set_title(f"Distribution Comparison for {case['title']}")
    axs[i].set_xlabel(f"{case['title']}")
    axs[i].set_ylabel("Frequency")
    axs[i].legend()

plt.tight_layout()
plt.savefig(OUTDIR / "distribution_comparison.png", dpi=300)
plt.close()