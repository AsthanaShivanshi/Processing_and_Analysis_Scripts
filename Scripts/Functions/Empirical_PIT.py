
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

def Empirical_PIT(tabsd_wet, rhiresd_wet, city_name="City", season_label="JJA"):
    """
    Applies Empirical PIT for grid specific tabsd_wet and rhiresd_wet,
    visualizes uniform empirically transformed marginals
    """

    # Empirical CDF of tabsd_wet and rhiresd_wet respectively
    ecdf_tabsd = ECDF(tabsd_wet)
    ecdf_rhiresd = ECDF(rhiresd_wet)

    # PIT
    tabs_U_empirical = ecdf_tabsd(tabsd_wet)
    rhires_U_empirical = ecdf_rhiresd(rhiresd_wet)

    # Marginal uniform PDFs for both : empirical uniform histograms
    plt.figure()
    plt.hist(tabs_U_empirical, bins=30, density=True, alpha=0.6, color='red', edgecolor='k')
    plt.title(f'Uniform Marginal: tabsd_U_empirical - {city_name} ({season_label})')
    plt.xlabel('PIT Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(rhires_U_empirical, bins=30, density=True, alpha=0.6, color='blue', edgecolor='k')
    plt.title(f'Uniform Marginal: rhiresd_U_empirical - {city_name} ({season_label})')
    plt.xlabel('PIT Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Clip for inverse transform to avoid infinite values at the margins
    epsilon = 1e-6
    tabs_U_clipped = np.clip(tabs_U_empirical, epsilon, 1 - epsilon)
    rhires_U_clipped = np.clip(rhires_U_empirical, epsilon, 1 - epsilon)

    # Transform to standard normal using inverse CDF (percent-point function)
    tabs_Z = norm.ppf(tabs_U_clipped)
    rhires_Z = norm.ppf(rhires_U_clipped)

    # Step 6: Stack and compute correlation matrix
    Z_values_stacked = np.column_stack((tabs_Z, rhires_Z))
    copula_corr_matrix = np.corrcoef(Z_values_stacked, rowvar=False)

    print(f"\nCopula Correlation Matrix ({city_name}, {season_label}):")
    print(copula_corr_matrix)

    return tabs_U_empirical, rhires_U_empirical, copula_corr_matrix
