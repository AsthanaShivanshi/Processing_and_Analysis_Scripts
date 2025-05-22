import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot
import pandas as pd

def plot_temperature_gridwise(temp_wet, mu_cell, sigma_cell, city_name="City",label="TabsD"):
    """
    Plots seasonal histogram, QQ plot, and CDF comparison for a temperature series on wet days,
    separated by meteorological seasons: DJF, MAM, JJA, SON.
    
    Parameters:
        temp_wet (xarray.DataArray or pandas.Series): Time series with datetime index.
        mu_cell (float): Mean temperature across all wet days.
        sigma_cell (float): Standard deviation of temperature across all wet days.
        city_name (str): Name of the city.
    """

    # Convert to pandas Series
    if hasattr(temp_wet, 'to_series'):
        temp_wet = temp_wet.to_series()
    
    if not isinstance(temp_wet.index, pd.DatetimeIndex):
        raise ValueError("temp_wet must have a datetime index.")
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'DJF'
        elif month in [3, 4, 5]:
            return 'MAM'
        elif month in [6, 7, 8]:
            return 'JJA'
        elif month in [9, 10, 11]:
            return 'SON'

    seasons = temp_wet.index.month.map(get_season)
    temp_wet_by_season = temp_wet.groupby(seasons)
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for season,data in temp_wet_by_season:
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=False, color="red", stat="density", bins=50)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, mu_cell, sigma_cell)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f"{label} on Wet Days in {city_name} ({season}) 1971–2023")
        plt.xlabel(f"{label}(°C)")
        plt.ylabel("Density")
        plt.legend([f"Mean = {mu_cell:.2f}, Std Dev = {sigma_cell:.2f}"], loc="upper left")
        plt.grid(True)
        plt.show()

        # QQ Plot
        plt.figure(figsize=(8, 6))
        probplot(data, dist="norm", sparams=(mu_cell, sigma_cell), plot=plt)
        plt.title(f"QQ Plot for {label} ({season}) Rainy Days in {city_name}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Observed Quantiles")
        plt.grid(True)
        plt.show()

        # Empirical vs Parametric CDF
        sorted_data = np.sort(data)
        empirical_CDF = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fitted_CDF = norm.cdf(sorted_data, loc=mu_cell, scale=sigma_cell)

        plt.figure(figsize=(10, 6))
        plt.step(sorted_data, empirical_CDF, color="red", label="Empirical CDF", where='post')
        plt.plot(sorted_data, fitted_CDF, color="black", label="Parametric CDF")
        plt.xlabel(f"{label} (°C)")
        plt.ylabel("CDF")
        plt.title(f"Empirical vs Parametric CDF ({season}) Rainy Days in {city_name}")
        plt.grid(True)
        plt.legend()
        plt.show()
