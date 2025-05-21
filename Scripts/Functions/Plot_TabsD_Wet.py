import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot

def plot_tabsd_wet(tabsd_wet, mu_tabs_cell, sigma_tabs_cell, city_name="City", season_label="JJA"):
    """
    Plots histogram, QQ plot, and Empirical vs Parametric CDF for wet days of the closest grid cell of the specified lat lon value"""

    # Histogram 
    plt.figure(figsize=(10, 6))
    sns.histplot(tabsd_wet, kde=False, color="red", stat="density", bins=50)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu_tabs_cell, sigma_tabs_cell)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f"Average {season_label} daily temperatures for rainy days in {city_name} (1961–2023)")
    plt.xlabel("TabsD (°C)")
    plt.ylabel("Probability")
    plt.legend([f"Mean = {mu_tabs_cell:.2f}, Std Dev = {sigma_tabs_cell:.2f}"], loc="upper left")
    plt.grid(True)
    plt.show()

    # QQ Plot
    plt.figure(figsize=(8, 6))
    probplot(tabsd_wet, dist="norm", sparams=(mu_tabs_cell, sigma_tabs_cell), plot=plt)
    plt.title(f"QQ Plot for TabsD {season_label} (Rainy Days) in {city_name} (1961–2023)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Parametric Quantiles")
    plt.grid(True)
    plt.show()

    # Empirical vs Parametric CDF
    sorted_data = np.sort(tabsd_wet)
    empirical_CDF = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fitted_CDF = norm.cdf(sorted_data, loc=mu_tabs_cell, scale=sigma_tabs_cell)

    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, empirical_CDF, color="red", label="Empirical CDF", where='post')
    plt.plot(sorted_data, fitted_CDF, color="black", label="Parametric CDF")
    plt.xlabel("TabsD (°C)")
    plt.ylabel("CDF")
    plt.title(f"Empirical vs Parametric CDF for Rainy Days in {city_name}")
    plt.grid(True)
    plt.legend()
    plt.show()
