from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask

dask.config.set({"distributed.worker.memory.target": 0.8})

ProgressBar().register()

# Helper function to process a block
@delayed
def process_block(temp, mean, std, i_start, i_end, j_start, j_end):
    # Load only the block into memory
    block_temp = temp[:, i_start:i_end, j_start:j_end].compute()
    block_mean = mean[i_start:i_end, j_start:j_end].compute()
    block_std = std[i_start:i_end, j_start:j_end].compute()

    block_KS = np.full((i_end - i_start, j_end - j_start), np.nan)
    block_pval = np.full((i_end - i_start, j_end - j_start), np.nan)

    for ii in range(block_temp.shape[1]):  # Local block indices
        for jj in range(block_temp.shape[2]):
            data = block_temp[:, ii, jj]
            mu = block_mean[ii, jj]
            sigma = block_std[ii, jj]
            data = data[~np.isnan(data)]
            if len(data) > 0 and sigma > 0:
                stat, pval = kstest(data, 'norm', args=(mu, sigma))
                block_KS[ii, jj] = stat
                block_pval[ii, jj] = pval

    return block_KS, block_pval

def Kalmogorov_Smirnov_gridded(temp, mean, std, data_path, alpha=0.10, block_size=20, season="Season"):
    """Performs KS test for each grid cell and plots the result."""

    n_lat, n_lon = temp.sizes['N'], temp.sizes['E']
    KS_Stat = np.full((n_lat, n_lon), np.nan)
    p_val_ks_stat = np.full((n_lat, n_lon), np.nan)

    tasks = []

    for i in range(0, n_lat, block_size):
        for j in range(0, n_lon, block_size):
            i_end = min(i + block_size, n_lat)
            j_end = min(j + block_size, n_lon)
            task = process_block(temp, mean, std, i, i_end, j, j_end) 
            tasks.append((i, j, task))

    with ProgressBar():
        results = compute(*[t[2] for t in tasks], scheduler="processes")

    for idx, (i, j, _) in enumerate(tasks):
        block_KS, block_pval = results[idx]
        KS_Stat[i:i+block_KS.shape[0], j:j+block_KS.shape[1]] = block_KS
        p_val_ks_stat[i:i+block_pval.shape[0], j:j+block_pval.shape[1]] = block_pval

    E = data_path["E"].values
    N = data_path["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Binary mask: 1 if null hypothesis accepted, 0 if rejected. Null h<ypothesis is that the data follows the parametric distribution
    accept_h0 = (p_val_ks_stat > alpha).astype(int)
    # the test gets harsher for big sample sizes this also seems legit, since there IS a difference in the distributions, even if it is not that big.
    #Source : https://stats.stackexchange.com/questions/570430/kolmogorov-smirnov-p-value-and-alpha-value-in-python

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    cmap = plt.get_cmap('bwr_r', 2) 

    plot = ax.pcolormesh(lon, lat, accept_h0, cmap=cmap, shading="auto", vmin=0, vmax=1, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax=ax, shrink=0.7, orientation='horizontal', ticks=[0, 1])
    cbar.ax.set_xticklabels(['Reject H₀', 'Accept H₀'])
    cbar.set_label(f'KS Test Hypothesis Test (α={alpha})')
    plt.title(f'KS Test: Normality of TabsD - {season} with 90 pc confidence')
    plt.tight_layout()

    plt.savefig(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Outputs/TabsD/90_pc_confidence_TabsD_KS_Test_training_set_{season}.png", dpi=300)
    plt.close()

    return KS_Stat, p_val_ks_stat
