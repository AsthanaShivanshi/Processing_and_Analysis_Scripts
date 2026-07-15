import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import — no display needed
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from plotstyle import apply_paper_style, save_figure, VARIABLE_COLORS
import os 


ref_path = "../Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step1_latlon.nc"
etas = [0.0]
downscaled_paths = [
    f"../Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc"
    for eta in etas
]

output_dir = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(output_dir, exist_ok=True)


n_samples = 10

bins = 50

apply_paper_style()

#------------------------------------------------------------------------#

ref_ds = xr.open_dataset(ref_path).sel(time=slice("2015-01-01", "2023-12-31")) #Model calib evaluation on the test set. 
ref_temp = ref_ds['TabsD']
mask = ~np.isnan(ref_temp.values)
ref_temp_masked = np.where(mask, ref_temp.values, np.nan)

ref_spatial_mean = np.nanmean(ref_temp_masked, axis=(1, 2))

ecdf_ref_spatial = ECDF(ref_spatial_mean)
ref_pit = ecdf_ref_spatial(ref_spatial_mean)


for eta, ds_path in zip(etas, downscaled_paths):
    print(f"Processing eta={eta}...")

    ds = xr.open_dataset(ds_path)
    temp = ds['temp']  
    temp_values = temp.values  

    mask_4d = np.broadcast_to(mask[:, np.newaxis, :, :], temp_values.shape)
    temp_masked = np.where(mask_4d, temp_values, np.nan)

    temp_spatial_mean = np.nanmean(temp_masked, axis=(2, 3))

    all_pit = []
    for s in range(temp_spatial_mean.shape[1]):
        pit_s = ecdf_ref_spatial(temp_spatial_mean[:, s])
        all_pit.append(pit_s)
    pit = np.concatenate(all_pit) 

    n_samples = temp_spatial_mean.shape[1]
    n_timesteps = ref_spatial_mean.shape[0]

    bin_edges = np.linspace(0, 1, bins + 1)
    ref_hist, _ = np.histogram(ref_pit, bins=bin_edges, density=True)
    pit_hist, _ = np.histogram(pit, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4

    #------------------------------------------------------------------------#

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_centers - width/2, ref_hist, width=width, label="Reference (2011–2023)",
           color='#4575b4', alpha=0.85, edgecolor='black')
    ax.bar(bin_centers + width/2, pit_hist, width=width, label=f"Diffusion (η={eta})",
           color=VARIABLE_COLORS["temp"], alpha=0.85, edgecolor='black')
    ax.plot([0, 1], [1, 1], color='gray', linestyle='-', linewidth=2)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram of Spatial Means")
    ax.annotate(f"Time steps: {n_timesteps}\nSamples: {n_samples}",
                xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    

    ax.legend(loc="best")


    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "Temp_PIT.pdf"))    
    plt.close()