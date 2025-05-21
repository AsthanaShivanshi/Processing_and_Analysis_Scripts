#!/usr/bin/env python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from pyproj import Transformer
import os
from joblib import Parallel, delayed  # for parallelism


def calculate_correlation(ds1_path, ds2_path, n_jobs=4):
    # Load datasets with chunking for memory efficiency
    ds1 = xr.open_dataset(ds1_path, chunks={"time": 100})
    ds2 = xr.open_dataset(ds2_path, chunks={"time": 100})
    
    TabsD = ds1['TabsD']
    RhiresD = ds2['RhiresD']

    # Precompute grid coordinates
    E = ds1["E"].values
    N = ds1["N"].values
    E_grid, N_grid = np.meshgrid(E, N)
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }

    spearman_maps = {}
    kendall_maps = {}

    for season, months in seasons.items():
        print(f"Processing season: {season}")
        
        TabsD_season = TabsD.sel(time=TabsD['time.month'].isin(months))
        RhiresD_season = RhiresD.sel(time=RhiresD['time.month'].isin(months))
        
        TabsD_vals = TabsD_season.values
        RhiresD_vals = RhiresD_season.values
        
        spearman_grid = np.full((len(N), len(E)), np.nan)
        kendall_grid = np.full((len(N), len(E)), np.nan)

        def compute_cell(i, j):
            x = TabsD_vals[:, i, j]
            y = RhiresD_vals[:, i, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) > 2:
                spearman_corr, _ = spearmanr(x[mask], y[mask])
                kendall_corr, _ = kendalltau(x[mask], y[mask])
                return (i, j, spearman_corr, kendall_corr)
            else:
                return (i, j, np.nan, np.nan)

        # Parallel computation
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_cell)(i, j) for i in range(len(N)) for j in range(len(E))
        )

        for i, j, spearman_corr, kendall_corr in results:
            spearman_grid[i, j] = spearman_corr
            kendall_grid[i, j] = kendall_corr

        # Save season results
        spearman_maps[season] = spearman_grid
        kendall_maps[season] = kendall_grid

    return lon, lat, spearman_maps, kendall_maps

def plot_correlations(lon, lat, correlation_maps, title_prefix, output_dir):
    fig_count = 0
    for season, corr in correlation_maps.items():
        plt.figure(figsize=(8,6))
        plt.pcolormesh(lon, lat, corr, shading='auto')
        plt.colorbar(label='Correlation')
        plt.title(f'{title_prefix} {season}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{title_prefix}_{season}.png")
        plt.savefig(filename, dpi=150)  # Slightly lower DPI to speed up
        plt.close()
        fig_count += 1
    print(f"Saved {fig_count} plots for {title_prefix}")

def main():
    ds1_path = "../../data/targets_tas_masked_train.nc"
    ds2_path = "../../data/targets_precip_masked_train.nc"
    output_dir = "../../Outputs/plots"
    os.makedirs(output_dir, exist_ok=True)

    lon, lat, spearman_maps, kendall_maps = calculate_correlation(ds1_path, ds2_path, n_jobs=4)
    
    plot_correlations(lon, lat, spearman_maps, 'Spearman', output_dir)
    plot_correlations(lon, lat, kendall_maps, 'Kendall', output_dir)

if __name__ == "__main__":
    main()
