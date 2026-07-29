from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config

def _sanitize_obs(da: xr.DataArray, var: str) -> xr.DataArray:
    da = da.squeeze(drop=True)
    return da.clip(min=0) if var == "pr" else da


def _subset_years(da: xr.DataArray, y1: int, y2: int) -> xr.DataArray:
    yy = da["time"].dt.year
    return da.where((yy >= y1) & (yy <= y2), drop=True)


def _spatial_dims(da: xr.DataArray):
    return [d for d in da.dims if d != "time"]


def rmse_spatiotemporal(pred: xr.DataArray, ref: xr.DataArray) -> float:
    pred_ts = pred.mean(dim=_spatial_dims(pred), skipna=True)
    ref_ts = ref.mean(dim=_spatial_dims(ref), skipna=True)
    pred_ts, ref_ts = xr.align(pred_ts, ref_ts, join="inner")
    return float(np.sqrt(((pred_ts - ref_ts) ** 2).mean(skipna=True).values))
