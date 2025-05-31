
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

def Parametric_PIT(tabsd_wet, rhiresd_wet,
                   mu_tabs_cell, sigma_tabs_cell,
                   alpha_mle, beta_mle,
                   city_name="City", season_label="JJA"):
    """
    Applies Parametric PIT for grid-specific tabsd_wet and rhiresd_wet.
    Assumes tabsd_wet follows Normal(mu, sigma)
    and rhiresd_wet follows Gamma(alpha, scale=beta).
    Visualizes uniform marginals after transformation.
    """

    # Ensure positive standard deviation and shape parameters
    sigma_tabs_cell = np.maximum(sigma_tabs_cell, 1e-6)
    alpha_mle = np.maximum(alpha_mle, 1e-6)
    beta_mle = np.maximum(beta_mle, 1e-6)

    # Parametric PIT using CDFs
    tabs_U_parametric = norm.cdf(tabsd_wet, loc=mu_tabs_cell, scale=sigma_tabs_cell)
    rhires_U_parametric = gamma.cdf(rhiresd_wet, a=alpha_mle, scale=beta_mle)

    # Plot histograms of the PIT values
    plt.figure()
    plt.hist(tabs_U_parametric, bins=30, density=True, alpha=0.6, color='red', edgecolor='k')
    plt.title(f'Parametric PIT: tabsd_U - {city_name} ({season_label})')
    plt.xlabel('PIT Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(rhires_U_parametric, bins=30, density=True, alpha=0.6, color='blue', edgecolor='k')
    plt.title(f'Parametric PIT: rhiresd_U - {city_name} ({season_label})')
    plt.xlabel('PIT Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Clip for inverse transform to avoid infinite values at the margins
    epsilon = 1e-6
    tabs_U_clipped = np.clip(tabs_U_parametric, epsilon, 1 - epsilon)
    rhires_U_clipped = np.clip(rhires_U_parametric, epsilon, 1 - epsilon)

    # Transform to standard normal using inverse CDF (percent-point function)
    tabs_Z = norm.ppf(tabs_U_clipped)
    rhires_Z = norm.ppf(rhires_U_clipped)

    # Step 6: Stack and compute correlation matrix
    Z_values_stacked = np.column_stack((tabs_Z, rhires_Z))
    copula_corr_matrix = np.corrcoef(Z_values_stacked, rowvar=False)

    print(f"\nCopula Correlation Matrix ({city_name}, {season_label}):")
    print(copula_corr_matrix)

    return tabs_U_parametric, rhires_U_parametric, copula_corr_matrix
