
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

def GaussianCopula_Sim(copula_corr_matrix, 
                                       rhires_U_parametric, 
                                       tabs_U_parametric, 
                                       n_samples, 
                                       city_name="City", 
                                       season_label="JJA"):
    """
    Plots the Gaussian copula density and simulates pseudo-observations from it along side the empirical pseudo obsevrations in the Gaussian Copula Space.
    """

    # Gaussian copula density
    u = np.linspace(0, 1, 100)
    U1, U2 = np.meshgrid(u, u)
    points = np.stack([U1.ravel(), U2.ravel()], axis=1)
    Z = norm.ppf(points)

    rv = multivariate_normal(mean=[0, 0], cov=copula_corr_matrix)
    pdf = rv.pdf(Z).reshape(U1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(U1, U2, pdf, levels=20, cmap='viridis')
    plt.colorbar(cp)
    corr_val = copula_corr_matrix[0][1]
    plt.title(f'Gaussian Copula Density ({city_name}, {season_label}, ρ={corr_val:.3f})')
    plt.xlabel('rhiresd_U')
    plt.ylabel('tabsd_U')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Simulate N samples from the Gaussian copula
    simulated_Z = np.random.multivariate_normal(mean=[0, 0], cov=copula_corr_matrix, size=n_samples)
    simulated_U = norm.cdf(simulated_Z)
    rhires_U_sim = simulated_U[:, 0] #x axis
    tabs_U_sim = simulated_U[:, 1] #y axis

    # Plot empirical vs simulated pseudo obs
    plt.figure(figsize=(8, 6))
    plt.scatter(rhires_U_parametric, tabs_U_parametric, alpha=0.3, color='red', edgecolor='k', label='Parametric Pseudo Obs')
    plt.scatter(rhires_U_sim, tabs_U_sim, alpha=0.3, color='blue', edgecolor='none', label=f'Simulated ({n_samples} samples)')
    plt.title(f'Empirical vs Simulated Uniform Marginals using Empirical PIT({city_name}, {season_label})')
    plt.xlabel('rhiresd_U')
    plt.ylabel('tabsd_U')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
