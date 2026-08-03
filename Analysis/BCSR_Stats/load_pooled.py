from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import xarray as xr

import config


ROW_ORDER = [
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

MASK_LEVEL = {k: "hr" for k in ROW_ORDER}


@dataclass
class LoadedPooled:
    data: Dict[str, Dict[str, xr.DataArray]]
    hr_mask: xr.DataArray
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
    m = mask.isel(time=0, drop=True) if "time" in mask.dims else mask
    spatial_dims = tuple(d for d in da.dims if d != "time" and d != "member")
    m2 = xr.DataArray(m.values, coords={d: da.coords[d] for d in spatial_dims}, dims=spatial_dims)
    return da.where(m2 > 0)


def apply_mask_exact(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return _apply_mask(da, mask)


def _build_case_templates() -> Dict[str, List[str]]:
    return {
        "EQM + Bilinear": [
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "CDF-t + Bilinear": [
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear": [
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "EQM + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "CDF-t + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "EQM + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_2021-2030.nc",
        ],
        "CDF-t + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_2021-2030.nc",
        ],
        "dOTC + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2015-2020.nc",
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_2021-2030.nc",
        ],
        "CH2025 methodological baseline": [
            "BC/EQM/ssp370/day/{var}/v20250415/MME_pooled_EQM_ssp370_{var}_all.nc",
        ],
    }


def _open_baseline_var(ens_root: Path, rels: List[str], var: str) -> xr.DataArray | None:
    files = [ens_root / r.format(var=var) for r in rels]
    files = [f for f in files if f.exists()]
    if not files:
        return None

    if len(files) == 1:
        ds = xr.open_dataset(files[0], chunks={"time": 365}, cache=False)
    else:
        ds = xr.open_mfdataset(
            [str(f) for f in sorted(files)],
            combine="by_coords",
            chunks={"time": 365},
            cache=False,
        )

    vn = _first_var(ds, var)
    da = _sanitize(ds[vn], var)
    return da


def load_all_pooled_masked(
    ens_root: str | Path,
    mask_hr_file: str | Path,
    *,
    mask_hr_var: str = "TabsD",
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
) -> LoadedPooled:
    ens_root = Path(ens_root)
    variables = variables or ["pr", "tas"]

    hr_mask = _load_mask(mask_hr_file, mask_hr_var)
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
            data[var][baseline] = _apply_mask(da, hr_mask)

    return LoadedPooled(data=data, hr_mask=hr_mask, missing=missing)


def load_all_pooled_masked_defaults(
    *,
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
) -> LoadedPooled:
    base = _default_base_dir()
    return load_all_pooled_masked(
        ens_root=base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled",
        mask_hr_file=base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        eval_start=eval_start,
        eval_end=eval_end,
        variables=variables,
    )