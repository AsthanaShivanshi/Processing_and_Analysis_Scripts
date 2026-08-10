from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import re

import numpy as np
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

COORD_ALIASES = {
    "lat": ("lat", "latitude", "nav_lat"),
    "lon": ("lon", "longitude", "nav_lon"),
}


@dataclass
class LoadedMeans:
    data: Dict[str, Dict[str, xr.DataArray]]
    hr_mask: xr.DataArray
    missing: Dict[str, List[str]]


def _default_base_dir() -> Path:
    return Path(config.BASE_DIR) / "sasthana/Downscaling"


def _sanitize(da: xr.DataArray, var: str) -> xr.DataArray:
    da = da.squeeze(drop=True)
    return da.clip(min=0) if var == "pr" else da


def _subset_years(da: xr.DataArray, y1: int, y2: int) -> xr.DataArray:
    if "time" not in da.coords:
        return da
    yy = da["time"].dt.year
    return da.where((yy >= y1) & (yy <= y2), drop=True)


def _first_var(ds: xr.Dataset, var: str) -> str:
    for name in [var, "tas", "pr", "temp", "precip", "TabsD", "RhiresD"]:
        if name in ds.data_vars:
            return name
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables.")
    return list(ds.data_vars)[0]


def _load_mask(mask_file: str | Path, mask_var: str) -> xr.DataArray:
    with xr.open_dataset(mask_file) as ds:
        da = ds[mask_var] if mask_var in ds else ds[list(ds.data_vars)[0]]
        return da.load()


def _apply_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    m = mask.isel(time=0, drop=True) if "time" in mask.dims else mask

    anchor = da
    for d in ("time", "member", "sample"):
        if d in anchor.dims:
            anchor = anchor.isel({d: 0}, drop=True)

    m_al, a_al = xr.align(m, anchor, join="inner")
    if m_al.size == 0 or a_al.size == 0:
        raise ValueError("Mask/data alignment produced empty overlap.")
    m2 = m_al.broadcast_like(a_al)
    return da.where(m2 > 0)


def apply_mask_exact(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return _apply_mask(da, mask)


def _build_case_templates() -> Dict[str, List[str]]:
    return {
        "EQM + Bilinear": [
            "BC+SR/Bilinear/EQM_C/historical/day/{var}/v20250415/MME_mean_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_mean_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear": [
            "BC+SR/Bilinear/CDF-t/historical/day/{var}/v20250415/MME_mean_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_mean_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear": [
            "BC+SR/Bilinear/dOTC/historical/day/{var}/v20250415/MME_mean_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_mean_dOTC_ssp370_{var}_*.nc",
        ],
        "EQM + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/EQM_C/historical/day/{var}/v20250415/MME_mean_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_mean_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/CDF-t/historical/day/{var}/v20250415/MME_mean_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_mean_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/dOTC/historical/day/{var}/v20250415/MME_mean_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_mean_dOTC_ssp370_{var}_*.nc",
        ],
        "EQM + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/historical/day/{var}/v20250415/MME_mean_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_mean_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/historical/day/{var}/v20250415/MME_mean_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_mean_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/dOTC/historical/day/{var}/v20250415/MME_mean_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_mean_dOTC_ssp370_{var}_*.nc",
        ],
        "CH2025 methodological baseline": [
            "BC/EQM/historical/day/{var}/v20250415/MME_mean_EQM_historical_{var}_*.nc",
            "BC/EQM/ssp370/day/{var}/v20250415/MME_mean_EQM_ssp370_{var}_all.nc",
            "BC/EQM/ssp370/day/{var}/v20250415/MME_mean_EQM_ssp370_{var}_*.nc",
        ],
    }


def _expand_relpaths(ens_root: Path, rels: List[str], var: str) -> List[Path]:
    out: List[Path] = []
    for rel in rels:
        patt = rel.format(var=var)
        if any(ch in patt for ch in ["*", "?", "["]):
            out.extend(list(ens_root.glob(patt)))
        else:
            p = ens_root / patt
            if p.exists():
                out.append(p)
    return sorted({p.resolve() for p in out}, key=lambda x: str(x))


def _extract_year_range_from_name(name: str) -> tuple[int, int] | None:
    m = re.search(r"_(\d{4})-(\d{4})\.nc$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _select_files_for_window(files: List[Path], eval_start: int, eval_end: int) -> List[Path]:
    if not files:
        return []

    ssp_all = [f for f in files if f.name.endswith("_all.nc")]
    if ssp_all:
        non_ssp370 = [f for f in files if "_ssp370_" not in f.name]
        files = non_ssp370 + ssp_all

    picked: List[Path] = []
    for f in files:
        yr = _extract_year_range_from_name(f.name)
        if yr is None:
            picked.append(f)
            continue
        y0, y1 = yr
        if not (y1 < eval_start or y0 > eval_end):
            picked.append(f)

    return sorted({p.resolve() for p in picked}, key=lambda x: str(x))


def _promote_and_standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    for canon, aliases in COORD_ALIASES.items():
        found = None
        for a in aliases:
            if a in ds.dims or a in ds.coords or a in ds.data_vars:
                found = a
                break
        if found is not None and found != canon:
            rename_map[found] = canon
    if rename_map:
        ds = ds.rename(rename_map)

    for c in ("lat", "lon"):
        if c in ds.data_vars and c not in ds.coords:
            ds = ds.set_coords(c)

    return ds


def _get_reference_spatial_coords(files: List[Path]) -> dict[str, xr.DataArray]:
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = _promote_and_standardize_coords(ds)
            out: dict[str, xr.DataArray] = {}
            if "lat" in ds.coords:
                out["lat"] = ds["lat"].load()
            if "lon" in ds.coords:
                out["lon"] = ds["lon"].load()
            if out:
                return out
    return {}


def _assign_missing_spatial_from_reference(
    ds: xr.Dataset, ref_coords: dict[str, xr.DataArray]
) -> xr.Dataset:
    for c in ("lat", "lon"):
        if c in ds.coords or c not in ref_coords:
            continue
        ref = ref_coords[c]

        # assign only if all ref dims exist in ds
        if all(d in ds.dims for d in ref.dims):
            ds = ds.assign_coords({c: ref})
        elif c in ds.dims and ref.ndim == 1 and ref.dims == (c,):
            ds = ds.assign_coords({c: ref})
    return ds


def _sort_and_unique_time(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.coords or da.sizes.get("time", 0) == 0:
        return da
    da = da.sortby("time")
    t = da["time"].values
    _, idx = np.unique(t, return_index=True)
    if len(idx) < len(t):
        da = da.isel(time=np.sort(idx))
    return da


def _open_baseline_var(
    ens_root: Path,
    rels: List[str],
    var: str,
    eval_start: int,
    eval_end: int,
) -> xr.DataArray | None:
    files = _expand_relpaths(ens_root, rels, var)
    files = _select_files_for_window(files, eval_start=eval_start, eval_end=eval_end)
    if not files:
        return None

    ref_coords = _get_reference_spatial_coords(files)

    def _pre(ds: xr.Dataset) -> xr.Dataset:
        ds = _promote_and_standardize_coords(ds)
        ds = _assign_missing_spatial_from_reference(ds, ref_coords)
        return ds

    if len(files) == 1:
        ds = xr.open_dataset(files[0], chunks={"time": 180}, cache=False)
        ds = _pre(ds)
    else:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            preprocess=_pre,
            coords="minimal",
            data_vars="minimal",
            compat="override",
            join="outer",
            chunks={"time": 180},
            cache=False,
        )

    vn = _first_var(ds, var)
    da = _sanitize(ds[vn], var)
    da = _sort_and_unique_time(da)
    return da


def load_all_means_masked(
    ens_root: str | Path,
    mask_hr_file: str | Path,
    *,
    mask_hr_var: str = "TabsD",
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
    verbose: bool = False,
) -> LoadedMeans:
    ens_root = Path(ens_root)
    variables = variables or ["pr", "tas"]

    hr_mask = _load_mask(mask_hr_file, mask_hr_var)
    templates = _build_case_templates()

    data = {v: {} for v in variables}
    missing = {v: [] for v in variables}

    for var in variables:
        for baseline in ROW_ORDER:
            da = _open_baseline_var(
                ens_root=ens_root,
                rels=templates[baseline],
                var=var,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            if da is None:
                missing[var].append(baseline)
                continue

            da = _subset_years(da, eval_start, eval_end)
            if "time" in da.coords and da.sizes.get("time", 0) == 0:
                missing[var].append(baseline)
                continue

            data[var][baseline] = _apply_mask(da, hr_mask)

    return LoadedMeans(data=data, hr_mask=hr_mask, missing=missing)


def load_all_means_masked_defaults(
    *,
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
) -> LoadedMeans:
    base = _default_base_dir()
    return load_all_means_masked(
        ens_root=base / "GCM_pipeline/ALP-FINEv1.0/Ensmeans",
        mask_hr_file=base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        eval_start=eval_start,
        eval_end=eval_end,
        variables=variables,
    )