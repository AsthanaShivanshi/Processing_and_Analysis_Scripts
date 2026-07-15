import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from plotstyle import apply_paper_style, save_figure, VARIABLE_COLORS
import os


ref_path = "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step1_latlon.nc"
etas = [0.0]
downscaled_paths = [
    f"../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc"
    for eta in etas
]



#inability of the model to predict dry days (exact zeroes, very small valuzes around zero instead. )


WET_THRESHOLD = 0.1 # mm/day

output_dir = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(output_dir, exist_ok=True)

bins = 50
apply_paper_style()



def zero_inflated_pit(obs, ref_values, rng=None, threshold=WET_THRESHOLD):
    if rng is None:
        rng = np.random.default_rng(42)

    obs_thresholded = np.where(obs < threshold, 0.0, obs)
    ref_thresholded = np.where(ref_values < threshold, 0.0, ref_values)

    ecdf = ECDF(ref_thresholded)
    p_zero = ecdf(0.0)

    pit = np.empty(len(obs_thresholded))
    zero_mask = (obs_thresholded == 0)

    pit[zero_mask] = rng.uniform(0.0, p_zero, size=zero_mask.sum())
    pit[~zero_mask] = ecdf(obs_thresholded[~zero_mask])

    return pit, p_zero



ref_ds = xr.open_dataset(ref_path).sel(time=slice("2015-01-01", "2023-12-31"))
ref_temp = ref_ds['RhiresD']
mask = ~np.isnan(ref_temp.values)
ref_temp_masked = np.where(mask, ref_temp.values, np.nan)

ref_spatial_mean = np.nanmean(ref_temp_masked, axis=(1, 2))

rng = np.random.default_rng(42)
ref_pit, p_zero_ref = zero_inflated_pit(ref_spatial_mean, ref_spatial_mean, rng=rng)

print(f"Reference zero fraction: {p_zero_ref:.3f}")

for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")

    ds = xr.open_dataset(ds_path)
    temp = ds['precip']
    temp_values = temp.values

    mask_4d = np.broadcast_to(mask[:, np.newaxis, :, :], temp_values.shape)
    temp_masked = np.where(mask_4d, temp_values, np.nan)

    temp_spatial_mean = np.nanmean(temp_masked, axis=(2, 3))

    all_pit = []
    for s in range(temp_spatial_mean.shape[1]):
        pit_s, _ = zero_inflated_pit(temp_spatial_mean[:, s], ref_spatial_mean, rng=rng)
        all_pit.append(pit_s)
    pit = np.concatenate(all_pit)

    n_samples  = temp_spatial_mean.shape[1]
    n_timesteps = ref_spatial_mean.shape[0]

    bin_edges   = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit,     bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width       = (bin_edges[1] - bin_edges[0]) * 0.4



    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(bin_centers - width/2, ref_hist, width=width,
           label="Reference (2015–2023)",
           color='#4575b4', alpha=0.85, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width,
           label=f"Diffusion (η={eta})",
           color=VARIABLE_COLORS["temp"], alpha=0.85, edgecolor='black')

    ax.axhline(1, color='gray', linestyle='-', linewidth=2, label='Uniform')


    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.annotate(
        f"Time steps: {n_timesteps}\nSamples: {n_samples}\n"
        f"Dry-day fraction: {p_zero_ref:.2f}",
        xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    ax.legend(loc="best")

    plt.tight_layout()


    save_figure(fig, os.path.join(output_dir, "Precip_PIT.pdf"))

    
    plt.close()