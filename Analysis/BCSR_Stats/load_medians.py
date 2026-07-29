from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr

import config


ROW_ORDER = [
    "Coarse (uncorrected)",
    "EQM Coarse",
    "CDF-t Coarse",
    "dOTC Coarse",
    "EQM + Bilinear",
    "CDF-t + Bilinear",
    "dOTC + Bilinear",
    "EQM + Bilinear + U-Net",
    "CDF-t + Bilinear + U-Net",
    "dOTC + Bilinear + U-Net",
    "EQM + Bilinear + U-Net + DDIM",
    "CDF-t + Bilinear + U-Net + DDIM",
    "dOTC + Bilinear + U-Net + DDIM",
    "CH2025 methodological baseline",
]

MASK_LEVEL = {
    "Coarse (uncorrected)": "lr",
    "EQM Coarse": "lr",
    "CDF-t Coarse": "lr",
    "dOTC Coarse": "lr",
    "EQM + Bilinear": "hr",
    "CDF-t + Bilinear": "hr",
    "dOTC + Bilinear": "hr",
    "EQM + Bilinear + U-Net": "hr",
    "CDF-t + Bilinear + U-Net": "hr",
    "dOTC + Bilinear + U-Net": "hr",
    "EQM + Bilinear + U-Net + DDIM": "hr",
    "CDF-t + Bilinear + U-Net + DDIM": "hr",
    "dOTC + Bilinear + U-Net + DDIM": "hr",
    "CH2025 methodological baseline": "lr",
}


@dataclass
class LoadedMedians:
    data: Dict[str, Dict[str, xr.DataArray]]  # data[var][baseline]
    hr_mask: xr.DataArray
    lr_mask: xr.DataArray
    missing: Dict[str, List[str]]


def _default_base_dir() -> Path:
    return Path(config.BASE_DIR) / "sasthana/Downscaling"


def _sanitize(da: xr.DataArray, var: str) -> xr.DataArray:
    da = da.squeeze(drop=True)
    return da.clip(min=0) if var == "pr" else da


def _subset_years(da: xr.DataArray, y1: int, y2: int) -> xr.DataArray:
    yy = da["time"].dt.year
    return da.where((yy >= y1) & (yy <= y2), drop=True)


def _first_var(ds: xr.Dataset, var: str) -> str:
    for name in [var, "tas", "pr", "temp", "precip", "TabsD", "RhiresD"]:
        if name in ds.data_vars:
            return name
    return list(ds.data_vars)[0]


def _load_mask(mask_file: str | Path, mask_var: str) -> xr.DataArray:
    ds = xr.open_dataset(mask_file)
    da = ds[mask_var] if mask_var in ds else ds[list(ds.data_vars)[0]]
    return da.load()



def _apply_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    m = mask
    if "time" in m.dims:
        m = m.isel(time=0, drop=True)

    da2 = da.isel(time=0, drop=True) if "time" in da.dims else da

    # exact same shape -> direct mask
    if da2.shape == m.shape:
        m2 = xr.DataArray(m.values, coords=da2.coords, dims=da2.dims)
        return da.where((m2 > 0) & np.isfinite(m2))

    # plot_frame_model_run style fallback: rename + nearest interp
    mk = m
    if "latitude" in mk.coords and "lat" not in mk.coords:
        mk = mk.rename({"latitude": "lat"})
    if "longitude" in mk.coords and "lon" not in mk.coords:
        mk = mk.rename({"longitude": "lon"})

    ren_da = {}
    if "latitude" in da.dims:
        ren_da["latitude"] = "lat"
    if "longitude" in da.dims:
        ren_da["longitude"] = "lon"

    dw = da.rename(ren_da) if ren_da else da
    mi = mk.interp(lat=dw["lat"], lon=dw["lon"], method="nearest")
    out = dw.where(mi > 0)

    if ren_da:
        out = out.rename({v: k for k, v in ren_da.items()})
    return out


def apply_mask_exact(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return _apply_mask(da, mask)


def _build_case_templates() -> Dict[str, List[str]]:
    return {
        "Coarse (uncorrected)": [
            "Swiss/ssp370/day/{var}/v20250415/MME_median_Swiss_ssp370_{var}_all.nc",
        ],
        "EQM Coarse": [
            "BC/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_all.nc",
        ],
        "CDF-t Coarse": [
            "BC/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_all.nc",
        ],
        "dOTC Coarse": [
            "BC/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_all.nc",
        ],
        "CDF-t + Bilinear": [
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "EQM + Bilinear": [
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear": [
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "EQM + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "CDF-t + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "CDF-t + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_median_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "EQM + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_median_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_median_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "CH2025 methodological baseline": [
            "BC/EQM/ssp370/day/{var}/v20250415/MME_median_EQM_ssp370_{var}_all.nc",
        ],
    }


def _open_baseline_var(ens_root: Path, rels: List[str], var: str) -> xr.DataArray | None:
    files = [ens_root / r.format(var=var) for r in rels]
    files = [f for f in files if f.exists()]
    if not files:
        return None

    ds = xr.open_dataset(files[0]) if len(files) == 1 else xr.open_mfdataset([str(f) for f in sorted(files)], combine="by_coords")
    vn = _first_var(ds, var)
    da = _sanitize(ds[vn], var).load()
    ds.close()
    return da


def load_all_medians_masked(
    ens_root: str | Path,
    mask_hr_file: str | Path,
    mask_lr_file: str | Path,
    *,
    mask_hr_var: str = "TabsD",
    mask_lr_var: str = "TabsD",
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
) -> LoadedMedians:
    ens_root = Path(ens_root)
    variables = variables or ["pr", "tas"]

    hr_mask = _load_mask(mask_hr_file, mask_hr_var)
    lr_mask = _load_mask(mask_lr_file, mask_lr_var)
    templates = _build_case_templates()

    data = {v: {} for v in variables}
    missing = {v: [] for v in variables}

    for var in variables:
        for baseline in ROW_ORDER:
            da = _open_baseline_var(ens_root, templates[baseline], var)
            if da is None:
                missing[var].append(baseline)
                continue

            da = _subset_years(da, eval_start, eval_end)
            mask = lr_mask if MASK_LEVEL[baseline] == "lr" else hr_mask
            data[var][baseline] = _apply_mask(da, mask)

    return LoadedMedians(data=data, hr_mask=hr_mask, lr_mask=lr_mask, missing=missing)


def load_all_medians_masked_defaults(
    *,
    eval_start: int = 2015,
    eval_end: int = 2023,
    
    variables: List[str] | None = None,
) -> LoadedMedians:
    base = _default_base_dir()
    return load_all_medians_masked(
        ens_root=base / "GCM_pipeline/ALP-FINEv1.0/Ensmedians",
        mask_hr_file=base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        mask_lr_file=base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc",
        eval_start=eval_start,
        eval_end=eval_end,
        variables=variables,
    )