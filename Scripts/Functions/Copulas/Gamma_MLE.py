
import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize

def gamma_mle(data):
    """
    Estimates the shape (alpha) and scale (beta) parameters of a Gamma distribution
    using MLE method using initial guess based on the method of moments"""

    def NLL(params, data):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        return -np.sum(np.log(gamma.pdf(data, a=alpha, loc=0, scale=beta)))


    # Initial guesses 
    mean_data = np.mean(data)
    var_data = np.var(data)
    alpha_0 = (mean_data ** 2) / var_data
    beta_0 = var_data / mean_data
    guess_0 = [alpha_0, beta_0]
    result = minimize(NLL, guess_0, args=(data,))

    
    alpha_mle, beta_mle = result.x
    return alpha_mle, beta_mle
