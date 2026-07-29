from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config


def lhd(pred: xr.DataArray, ref: xr.DataArray, bins=50, min_count=10, eps=1e-12) -> float:
    p = np.asarray(pred.values).ravel()
    r = np.asarray(ref.values).ravel()
    m = np.isfinite(p) & np.isfinite(r)
    p, r = p[m], r[m]
    if p.size == 0:
        return np.nan

    vmin, vmax = np.nanmin(np.r_[p, r]), np.nanmax(np.r_[p, r])
    if vmax <= vmin:
        return np.nan

    edges = np.linspace(vmin, vmax, bins + 1)
    hp, _ = np.histogram(p, bins=edges)
    hr, _ = np.histogram(r, bins=edges)

    valid = (hp > min_count) & (hr > min_count)
    if not np.any(valid):
        return np.nan

    hp = hp[valid] / hp[valid].sum()
    hr = hr[valid] / hr[valid].sum()
    return float(np.sqrt(np.mean((np.log(hp + eps) - np.log(hr + eps)) ** 2)))

