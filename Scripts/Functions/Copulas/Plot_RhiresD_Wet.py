
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, probplot

def plot_rhiresd_wet(rhiresd_wet, alpha_mle, beta_mle, city_name="City", season_label="JJA"):
    """
    Plots histogram, QQ plot, and Empirical vs Parametric CDF for rhiresd_wet for the pixel specified using fitted gamma distribution.
    """
    rhiresd_wet = np.ravel(rhiresd_wet)

    # Histogram + parameteric Gamma PDF
    plt.figure(figsize=(10, 6))
    sns.histplot(rhiresd_wet, kde=False, color="blue", stat="density", bins=50)
    x = np.linspace(min(rhiresd_wet), max(rhiresd_wet), 1000)
    pdf_gamma = gamma.pdf(x, a=alpha_mle, scale=beta_mle)
    plt.plot(x, pdf_gamma, 'k-', label=f"Gamma(shape={alpha_mle:.3f}, scale={beta_mle:.3f})")
    plt.title(f"{season_label} Daily Precipitation for Wet Days in {city_name} (1961–2023)")
    plt.xlabel("RhiresD (mm/day)")
    plt.ylabel("Probability ")
    plt.legend()
    plt.grid(True)
    plt.show()

    # QQ Plot
    plt.figure(figsize=(8, 6))
    probplot(rhiresd_wet, dist="gamma", sparams=(alpha_mle, 0, beta_mle), plot=plt)
    plt.title(f"QQ Plot for RhiresD on Wet Days ({season_label}) in {city_name}")
    plt.xlabel("Parametric Quantiles")
    plt.ylabel("Empirical Quantiles")
    plt.grid(True)
    plt.show()

    # Empirical vs Parametric CDF
    sorted_data = np.sort(rhiresd_wet)
    empirical_CDF = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fitted_CDF = gamma.cdf(sorted_data, a=alpha_mle, scale=beta_mle)

    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, empirical_CDF, color="blue", where='post', label="Empirical CDF")
    plt.plot(sorted_data, fitted_CDF, color="black", label="Fitted Gamma CDF")
    plt.xlabel("Daily Precipitation (mm/day)")
    plt.ylabel("CDF")
    plt.title(f"Empirical vs Parametric CDF for Precipitation on Wet Days ({season_label}) in {city_name}")
    plt.grid(True)
    plt.legend()
    plt.show()
