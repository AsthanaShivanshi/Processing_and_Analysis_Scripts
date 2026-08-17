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
    "Fine-scale EQM",
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

    m_aligned, anchor_aligned = xr.align(m, anchor, join="inner")
    if m_aligned.size == 0 or anchor_aligned.size == 0:
        raise ValueError("Mask/data alignment produced empty overlap. Check grid/coords.")

    m2 = m_aligned.broadcast_like(anchor_aligned)
    return da.where(m2 > 0)


def apply_mask_exact(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return _apply_mask(da, mask)


def _build_case_templates() -> Dict[str, List[str]]:
    return {
        "EQM + Bilinear": [
            "BC+SR/Bilinear/EQM_C/historical/day/{var}/v20250415/MME_pooled_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear": [
            "BC+SR/Bilinear/CDF-t/historical/day/{var}/v20250415/MME_pooled_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear": [
            "BC+SR/Bilinear/dOTC/historical/day/{var}/v20250415/MME_pooled_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_*.nc",
        ],
        "EQM + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/EQM_C/historical/day/{var}/v20250415/MME_pooled_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/CDF-t/historical/day/{var}/v20250415/MME_pooled_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear + U-Net": [
            "BC+SR/Bilinear_UNet/dOTC/historical/day/{var}/v20250415/MME_pooled_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_*.nc",
        ],
        "EQM + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/historical/day/{var}/v20250415/MME_pooled_EQM_C_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/EQM_C/ssp370/day/{var}/v20250415/MME_pooled_EQM_C_ssp370_{var}_*.nc",
        ],
        "CDF-t + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/historical/day/{var}/v20250415/MME_pooled_CDF-t_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/CDF-t/ssp370/day/{var}/v20250415/MME_pooled_CDF-t_ssp370_{var}_*.nc",
        ],
        "dOTC + Bilinear + U-Net + DDIM": [
            "BC+SR/Bilinear_UNet_DDIM/dOTC/historical/day/{var}/v20250415/MME_pooled_dOTC_historical_{var}_*.nc",
            "BC+SR/Bilinear_UNet_DDIM/dOTC/ssp370/day/{var}/v20250415/MME_pooled_dOTC_ssp370_{var}_*.nc",
        ],
        "Fine-scale EQM": [
            "BC/EQM/historical/day/{var}/v20250415/MME_pooled_EQM_historical_{var}_*.nc",
            "BC/EQM/ssp370/day/{var}/v20250415/MME_pooled_EQM_ssp370_{var}_all.nc",
            "BC/EQM/ssp370/day/{var}/v20250415/MME_pooled_EQM_ssp370_{var}_*.nc",
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

    # Prefer *_all.nc for ssp370 if present; keep historical files too.
    ssp_all = [f for f in files if f.name.endswith("_all.nc")]
    if ssp_all:
        non_ssp370 = [f for f in files if "_ssp370_" not in f.name]
        files = non_ssp370 + ssp_all

    picked: List[Path] = []
    for f in files:
        yrs = _extract_year_range_from_name(f.name)
        if yrs is None:
            picked.append(f)  # keep *_all.nc
            continue
        y0, y1 = yrs
        if not (y1 < eval_start or y0 > eval_end):
            picked.append(f)

    return sorted({p.resolve() for p in picked}, key=lambda x: str(x))


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
    files = _select_files_for_window(files, eval_start, eval_end)
    if not files:
        return None

    if len(files) == 1:
        ds = xr.open_dataset(files[0], chunks={"time": 365}, cache=False)
    else:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            chunks={"time": 365},
            cache=False,
        )

    vn = _first_var(ds, var)
    da = _sanitize(ds[vn], var)
    da = _sort_and_unique_time(da)
    return da



def load_all_pooled_masked(
    ens_root: str | Path,
    mask_hr_file: str | Path,
    *,
    mask_hr_var: str = "TabsD",
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
    labels: List[str] | None = None,
    verbose: bool = False,
) -> LoadedPooled:
    ens_root = Path(ens_root)
    variables = variables or ["pr", "tas"]

    if labels is not None:
        unknown = set(labels) - set(ROW_ORDER)
        if unknown:
            raise ValueError(f"Unknown baseline labels: {sorted(unknown)}")
        selected_baselines = [b for b in ROW_ORDER if b in labels]
    else:
        selected_baselines = ROW_ORDER

    hr_mask = _load_mask(mask_hr_file, mask_hr_var)
    templates = _build_case_templates()

    data = {v: {} for v in variables}
    missing = {v: [] for v in variables}

    for var in variables:
        for baseline in selected_baselines:
            rels = templates[baseline]

            files = _select_files_for_window(
                _expand_relpaths(ens_root, rels, var),
                eval_start=eval_start,
                eval_end=eval_end,
            )

            if verbose:
                print(f"[info] {var} | {baseline} | files={len(files)}")
                for file in files:
                    print(f"       - {file}")

            da = _open_baseline_var(
                ens_root=ens_root,
                rels=rels,
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

    return LoadedPooled(
        data=data,
        hr_mask=hr_mask,
        missing=missing,
    )


def load_all_pooled_masked_defaults(
    *,
    eval_start: int = 2015,
    eval_end: int = 2023,
    variables: List[str] | None = None,
    verbose: bool = False,
) -> LoadedPooled:
    base = _default_base_dir()
    return load_all_pooled_masked(
        ens_root=base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled",
        mask_hr_file=base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        eval_start=eval_start,
        eval_end=eval_end,
        variables=variables,
        verbose=verbose,
    )