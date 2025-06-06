import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gamma, kstest
from scipy.optimize import minimize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from dask import delayed, compute
from dask.diagnostics import ProgressBar

ProgressBar().register()

def NLL(params, data):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    return -np.sum(gamma.logpdf(data, a=alpha, scale=beta))

def fit_gamma_mle(data):
    """ Fit Gamma distribution to data via MLE """

    data = data[np.isfinite(data) & (data > 0)]  # Gamma only for positive values

    if len(data) == 0:
        return np.nan, np.nan

    mean_data = np.mean(data)
    var_data = np.var(data)

    if mean_data <= 0 or var_data <= 0:
        return np.nan, np.nan

    alpha_0 = mean_data**2 / var_data
    beta_0 = var_data / mean_data
    guess_0 = [alpha_0, beta_0]

    def safe_nll(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        logpdf = gamma.logpdf(data, a=alpha, scale=beta)
        if not np.all(np.isfinite(logpdf)):
            return np.inf
        return -np.sum(logpdf)

    result = minimize(
        safe_nll,
        guess_0,
        method='L-BFGS-B',
        bounds=[(1e-6, None), (1e-6, None)]
    )

    if result.success and np.all(np.isfinite(result.x)):
        return result.x  # alpha_mle, beta_mle
    else:
        return np.nan, np.nan

    
@delayed
def process_block_precip(temp_block, i_start, j_start):
    block_data = temp_block.compute()  

    block_KS = np.full(block_data.shape[1:], np.nan)
    block_pval = np.full(block_data.shape[1:], np.nan)

    for ii in range(block_data.shape[1]):
        for jj in range(block_data.shape[2]):
            data = block_data[:, ii, jj]
            data = data[~np.isnan(data)]  # removing NaNs

            if len(data) == 0:
                continue  # Skip if all values are NaN

            alpha_mle, beta_mle = fit_gamma_mle(data)
            if np.isnan(alpha_mle) or np.isnan(beta_mle):
                continue  # Skip if MLE failed

            stat, pval = kstest(data, 'gamma', args=(alpha_mle, 0, beta_mle))
            block_KS[ii, jj] = stat
            block_pval[ii, jj] = pval

    return block_KS, block_pval


def Gamma_KS_gridded(temp, data_path, alpha=0.05, block_size=20, season="Season"):
    """Performs KS test for each grid cell and plot gridwise for precipitation Gamma fitting"""
    
    n_lat, n_lon = temp.sizes['N'], temp.sizes['E']
    KS_Stat = np.full((n_lat, n_lon), np.nan)
    p_val_ks_stat = np.full((n_lat, n_lon), np.nan)

    tasks = []
    for i in range(0, n_lat, block_size):
        for j in range(0, n_lon, block_size):
            i_end = min(i + block_size, n_lat)
            j_end = min(j + block_size, n_lon)
            task = process_block_precip(temp[:, i:i_end, j:j_end], i, j)
            tasks.append((i, j, task))

    with ProgressBar():
        results = compute(*[t[2] for t in tasks], scheduler="threads")

    for idx, (i, j, _) in enumerate(tasks):
        block_KS, block_pval = results[idx]
        KS_Stat[i:i+block_KS.shape[0], j:j+block_KS.shape[1]] = block_KS
        p_val_ks_stat[i:i+block_pval.shape[0], j:j+block_pval.shape[1]] = block_pval

    # Prepare grid for plotting
    E = data_path["E"].values
    N = data_path["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Binary mask: 1 if accepted, 0 if rejected
    # Null hypothesis is that the data follows the Gamma distribution
    accept_h0 = (p_val_ks_stat > alpha).astype(int)

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
    cbar.set_label(f'KS Test for Gamma Fit (α={alpha})')
    plt.title(f'Gamma KS Test: Wet Day Precipitation - {season}')
    plt.tight_layout()

    plt.savefig(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/Outputs/90pc_alpha_Gamma_KS_Test_training_set_{season}.png", dpi=300)
    plt.close()

    return KS_Stat, p_val_ks_stat
