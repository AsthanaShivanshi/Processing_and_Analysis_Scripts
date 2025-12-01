import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from closest_grid_cell import select_nearest_grid_cell

import proplot as pplt

# Set global poster-worthy style
pplt.rc.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 38,
    'axes.titlesize': 44,
    'axes.titleweight': 'bold',
    'axes.labelsize': 40,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 32,
    'figure.titlesize': 44,
    'lines.linewidth': 5,
    'axes.linewidth': 2.5,
    'savefig.dpi': 1000,
    'figure.figsize': (16, 12),
    'axes.grid': True,
    'grid.linewidth': 1.5,
    'grid.alpha': 0.5,
    'legend.frameon': False,
    'legend.loc': 'best',
    'axes.spines.top': False,
    'axes.spines.right': False,
})



city_name = "Geneva"  

np.Inf=np.inf


cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417)
}
target_lat, target_lon = cities[city_name]
time_slice = slice("1981-01-01", "2010-12-31")

coarse_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Model_Runs/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc"
dotc_bc_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/temp_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc"
dotc_bicubic_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/temp_BC_bicubic_r01.nc"
unet_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
obs_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc"

coarse_ds = xr.open_dataset(coarse_path)
dotc_bc_ds = xr.open_dataset(dotc_bc_path)
dotc_bicubic_ds = xr.open_dataset(dotc_bicubic_path)
unet_ds = xr.open_dataset(unet_path)
obs_ds = xr.open_dataset(obs_path)

def daily_climatology(arr, time_coord):
    dayofyear = time_coord.dt.dayofyear
    clim = np.full((366,), np.nan)
    for doy in range(1, 367):
        mask = (dayofyear == doy)
        if np.any(mask):
            clim[doy-1] = np.nanmean(arr[mask])
        else:
            clim[doy-1] = np.nan
    return clim



def extract_climatology(ds, var, lat, lon):
    result = select_nearest_grid_cell(ds, lat, lon, var_name=var)
    data = result['data'].sel(time=time_slice)
    return daily_climatology(data.values, data['time'])



coarse_clim = extract_climatology(coarse_ds, "temp", target_lat, target_lon)
dotc_bc_clim = extract_climatology(dotc_bc_ds, "temp", target_lat, target_lon)
dotc_bicubic_clim = extract_climatology(dotc_bicubic_ds, "temp", target_lat, target_lon)
unet_clim = extract_climatology(unet_ds, "temp", target_lat, target_lon)
obs_clim = extract_climatology(obs_ds, "TabsD", target_lat, target_lon)




def perkins_skill_score(a, b, nbins=50):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.sum(mask) < 10:
        return np.nan
    a_valid = a[mask]
    b_valid = b[mask]
    combined_data = np.concatenate([a_valid, b_valid])
    bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
    hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
    hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
    hist_a = hist_a / np.sum(hist_a)
    hist_b = hist_b / np.sum(hist_b)
    return np.sum(np.minimum(hist_a, hist_b))

pss_coarse = perkins_skill_score(coarse_clim, obs_clim)
pss_dotc_bc = perkins_skill_score(dotc_bc_clim, obs_clim)
pss_dotc_bicubic = perkins_skill_score(dotc_bicubic_clim, obs_clim)
pss_unet = perkins_skill_score(unet_clim, obs_clim)



print(f"Perkins Skill Score({city_name}):")

print(f"  Coarse non Bias Corrected:           {pss_coarse:.3f}")
print(f"  EQM Bias Corrected: {pss_dotc_bc:.3f}")
print(f"  EQM Bias Corrected + Bicubically Interpolated:   {pss_dotc_bicubic:.3f}")
print(f"  EQM Bias Corrected + Bicubically Interpolated+ UNet Super Resolved:  {pss_unet:.3f}")


cb_colors = ['black', 'red', 'green', 'blue']
days = np.arange(1, 367)

fig, ax = pplt.subplots(figsize=(16, 10))
ax.plot(days, obs_clim, label="Ground Truth", color="black", linewidth=5)
ax.plot(days, coarse_clim, label=f"Coarse non BC (PSS={pss_coarse:.2f})", color=cb_colors[0], linewidth=5)
ax.plot(days, dotc_bc_clim, label=f"EQM BC (PSS={pss_dotc_bc:.2f})", color=cb_colors[1], linewidth=5)
ax.plot(days, dotc_bicubic_clim, label=f"EQM + Bicubic Interpolation (PSS={pss_dotc_bicubic:.2f})", color=cb_colors[2], linewidth=5)
ax.plot(days, unet_clim, label=f"EQM + Bicubic Interpolation + UNet Super Resolved (PSS={pss_unet:.2f})", color=cb_colors[3], linewidth=5)

ax.format(
    xlabel="Day of Year",
    ylabel="Temperature (°C)",
    title=f"{city_name} PSS",
    legend='r',  # upper right side
    legendfontsize=32,
    xlabelsize=40,
    ylabelsize=40,
    titlesize=44,
    xticklabelsize=32,
    yticklabelsize=32,
    grid=True,
    gridlinewidth=1.5,
    gridalpha=0.5,
)
fig.save(f"{city_name}_daily_climatology_1981_2010.png", dpi=1000)
