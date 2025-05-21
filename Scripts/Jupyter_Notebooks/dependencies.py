# dependencies for the data exploration part of the project 
#How to load : load in your notebook using from dependencies import * or if it is a different subfolder , use the following code block to import 
# import sys
#sys.path.append('utils')  # give the full/relative path to the utils folder within ""
#from my_imports import *
import numpy as np
import xarray as xr
import pyproj
import scipy.stats as stats
from scipy.stats import spearmanr, linregress, norm, gamma, lognorm, pearsonr
from scipy.optimize import minimize
from scipy.special import gammaln
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from statsmodels.graphics.gofplots import qqplot_2samples
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
from scipy.stats import kstest

__all__ = [
    'np', 'xr', 'pyproj', 'stats', 'spearmanr', 'linregress', 'norm', 'gamma',
    'lognorm', 'pearsonr', 'minimize', 'gammaln', 'math', 'sys', 'plt', 'pd',
    'sns', 'mcolors', 'qqplot_2samples', 'Axes3D', 'ccrs', 'cfeature', 'gpd',
    'Transformer', 'ECDF', 'multivariate_normal'
]
