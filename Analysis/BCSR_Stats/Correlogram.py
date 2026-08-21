from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from time import perf_counter

from typing import Sequence

import argparse
import csv
import gc
import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

import config
from load_pooled import load_all_pooled_masked


#Conceptual source : https://github.com/LouiseIII/extreme-precip-sr 

#Many samples per location. : mmeber wise correlogram,  then averaging . 
# one normalized field per location
# pooling
# One correlogram per timestep. 



MIN_TIME_SAMPLES = 2
MIN_SPATIAL_POINTS = 3

REPLICATE_DIMS = {
    "time",
    "member",
    "sample",
    "samples",
}

PAIR_CHUNK_SIZE = 1_000_000
REPLICATE_CHUNK_SIZE = 1000


@dataclass
class CorrelogramResult:
    label: str
    bin_edges_km: np.ndarray
    bin_centers_km: np.ndarray
    correlation: np.ndarray
    pair_counts: np.ndarray


@dataclass
class DatasetPayload:
    label: str
    values: np.ndarray
    kind: str
    c1: np.ndarray
    c2: np.ndarray


def _spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    dims = [d for d in da.dims if d.lower() not in REPLICATE_DIMS]
    if not dims:
        raise ValueError("No spatial dimensions found.")
    return tuple(dims)


def _replicate_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d.lower() in REPLICATE_DIMS]


def _member_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d.lower() in {"member", "sample", "samples"}]


def _coord_name(da: xr.DataArray, names: Sequence[str]) -> str | None:
    for name in names:
        if name in da.coords:
            return name
    return None


def _haversine_km(lon1, lat1, lon2, lat2):
    radius_km = 6371.0

    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def _extract_coords(stacked: xr.DataArray) -> tuple[str, np.ndarray, np.ndarray]:
    lat_name = _coord_name(stacked, ("lat", "latitude", "nav_lat"))
    lon_name = _coord_name(stacked, ("lon", "longitude", "nav_lon"))



    x_name = _coord_name(stacked, ("x", "easting"))
    y_name = _coord_name(stacked, ("y", "northing"))

    if lat_name and lon_name:
        return (
            "latlon",
            np.asarray(stacked[lon_name].values, dtype=float),
            np.asarray(stacked[lat_name].values, dtype=float),
        )

    if x_name and y_name:
        return (
            "xy",
            np.asarray(stacked[x_name].values, dtype=float),
            np.asarray(stacked[y_name].values, dtype=float),
        )

    raise ValueError("Could not find lat/lon or x/y coordinates for distance calculation.")


def _standardize_over_time(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da



#Averaging over replicates., 
    # Multiple samples belong to the same timestep.

    valid_time = da.count("time") >= MIN_TIME_SAMPLES
    da = da.where(valid_time)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = da.mean("time", skipna=True)
        std = da.std("time", skipna=True, ddof=1)

    std = xr.where((std == 0) | ~np.isfinite(std), np.nan, std)
    return (da - mean) / std


def _prepare_payload(label: str, da: xr.DataArray) -> DatasetPayload:
    work = _standardize_over_time(da)

    if hasattr(work.data, "compute"):
        work = work.compute()

    spatial_dims = _spatial_dims(work)
    point_work = work.stack(point=spatial_dims).reset_index("point")

    kind, c1, c2 = _extract_coords(point_work)

    # Correlogram per member inside each timestep, then average over time.


    if "time" in point_work.dims:
        member_dims = _member_dims(point_work)
        if member_dims:
            stacked = point_work.stack(member=member_dims).transpose("time", "member", "point")
            values = np.asarray(stacked.values, dtype=float)
        else:
            values = np.asarray(point_work.transpose("time", "point").values, dtype=float)[:, None, :]
    else:
        member_dims = _member_dims(point_work)
        if member_dims:
            stacked = point_work.stack(member=member_dims).transpose("member", "point")
            values = np.asarray(stacked.values, dtype=float)[None, :, :]
        else:
            values = np.asarray(point_work.values, dtype=float)[None, None, :]



    return DatasetPayload(
        label=label,
        values=np.ascontiguousarray(values),
        kind=kind,
        c1=np.ascontiguousarray(c1),
        c2=np.ascontiguousarray(c2),
    )


def _precompute_pair_bins(
    kind: str,
    c1: np.ndarray,
    c2: np.ndarray,
    bin_edges_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = c1.size

    if n_points < 2:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    pair_i_all, pair_j_all = np.triu_indices(n_points, k=1)

    pair_i_chunks = []
    pair_j_chunks = []
    pair_bin_chunks = []

    for start in range(0, pair_i_all.size, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, pair_i_all.size)

        pair_i = pair_i_all[start:end]
        pair_j = pair_j_all[start:end]

        if kind == "latlon":
            distance = _haversine_km(
                c1[pair_i],
                c2[pair_i],
                c1[pair_j],
                c2[pair_j],
            )
        else:
            dx = c1[pair_j] - c1[pair_i]
            dy = c2[pair_j] - c2[pair_i]
            distance = np.sqrt(dx * dx + dy * dy)

        good_distance = np.isfinite(distance)
        if not np.any(good_distance):
            continue

        pair_bin = np.searchsorted(bin_edges_km, distance, side="right") - 1

        good = (
            good_distance
            & (pair_bin >= 0)
            & (pair_bin < bin_edges_km.size - 1)
        )

        if not np.any(good):
            continue

        pair_i_chunks.append(pair_i[good].astype(np.int32, copy=False))
        pair_j_chunks.append(pair_j[good].astype(np.int32, copy=False))
        pair_bin_chunks.append(pair_bin[good].astype(np.int32, copy=False))

    if not pair_i_chunks:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    return (
        np.concatenate(pair_i_chunks),
        np.concatenate(pair_j_chunks),
        np.concatenate(pair_bin_chunks),
    )


def _bin_stats_precomputed(
    values: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    bin_pair_idx: list[np.ndarray],
    nbins: int,
    min_points: int = MIN_SPATIAL_POINTS,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if valid.sum() < min_points:
        return np.full(nbins, np.nan)

    mean = np.nanmean(values)
    variance = np.nanvar(values)

    if not np.isfinite(variance) or variance == 0:
        return np.full(nbins, np.nan)

    correlation = np.full(nbins, np.nan, dtype=float)

    for bin_index, pair_indices in enumerate(bin_pair_idx):
        if pair_indices.size == 0:
            continue

        value_i = values[pair_i[pair_indices]]
        value_j = values[pair_j[pair_indices]]

        good = np.isfinite(value_i) & np.isfinite(value_j)
        if not np.any(good):
            continue

        products = (value_i[good] - mean) * (value_j[good] - mean)
        correlation[bin_index] = (products.mean()) / variance

    return correlation


def compute_correlogram_payload(
    payload: DatasetPayload,
    bin_edges_km: np.ndarray,
    *,
    replicate_chunk_size: int = REPLICATE_CHUNK_SIZE,
) -> CorrelogramResult:
    nbins = bin_edges_km.size - 1
    start_time = perf_counter()

    pair_i, pair_j, pair_bin = _precompute_pair_bins(
        payload.kind,
        payload.c1,
        payload.c2,
        bin_edges_km,
    )

    bin_pair_idx = [
        np.flatnonzero(pair_bin == bin_index)
        for bin_index in range(nbins)
    ]

    pair_counts = np.bincount(pair_bin, minlength=nbins).astype(int)

    print(
        f"[{payload.label}] pair precompute: {perf_counter() - start_time:.2f}s",
        flush=True,
    )

    # Final average: first over members inside each timestep, then over time.

    correlation_sum = np.zeros(nbins, dtype=float)
    time_count = np.zeros(nbins, dtype=int)

    ntime, nmember, _ = payload.values.shape

    for start in tqdm(
        range(0, ntime, replicate_chunk_size),
        desc=payload.label,
        leave=False,
    ):
        end = min(start + replicate_chunk_size, ntime)
        values_chunk = payload.values[start:end]

        for timestep_values in values_chunk:
            member_sum = np.zeros(nbins, dtype=float)
            member_count = np.zeros(nbins, dtype=int)

            for member_values in timestep_values:
                correlation = _bin_stats_precomputed(
                    member_values,
                    pair_i,
                    pair_j,
                    bin_pair_idx,
                    nbins,
                )

                good = np.isfinite(correlation)
                member_sum[good] += correlation[good]
                member_count[good] += 1

            timestep_correlation = np.full(nbins, np.nan, dtype=float)
            good_member = member_count > 0
            timestep_correlation[good_member] = member_sum[good_member] / member_count[good_member]

            good_time = np.isfinite(timestep_correlation)
            correlation_sum[good_time] += timestep_correlation[good_time]
            time_count[good_time] += 1

    result_correlation = np.full(nbins, np.nan, dtype=float)
    good = time_count > 0
    result_correlation[good] = correlation_sum[good] / time_count[good]

    bin_centers = 0.5 * (bin_edges_km[:-1] + bin_edges_km[1:])

    return CorrelogramResult(
        label=payload.label,
        bin_edges_km=bin_edges_km,
        bin_centers_km=bin_centers,
        correlation=result_correlation,
        pair_counts=pair_counts,
    )


def _safe_label(label: str) -> str:
    return (
        label.replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def save_correlogram_result_csv(
    result: CorrelogramResult,
    *,
    panel_name: str,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"{panel_name}_{_safe_label(result.label)}_correlogram.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "label",
                "panel",
                "bin_left_km",
                "bin_right_km",
                "bin_center_km",
                "correlation",
                "pair_count",
            ]
        )

        for left, right, center, correlation, count in zip(
            result.bin_edges_km[:-1],
            result.bin_edges_km[1:],
            result.bin_centers_km,
            result.correlation,
            result.pair_counts,
        ):
            writer.writerow(
                [
                    result.label,
                    panel_name,
                    left,
                    right,
                    center,
                    correlation,
                    int(count),
                ]
            )


def _load_data_array(
    *,
    var_name: str,
    ens_root: str,
    mask_hr_file: str,
    mask_hr_var: str,
    eval_start: int,
    eval_end: int,
    obs_file: str,
    obs_var: str,
    requested_label: str,
) -> xr.DataArray:
    if requested_label == "OBS":
        with xr.open_dataset(obs_file) as dataset:
            return dataset[obs_var].load()

    loaded = load_all_pooled_masked(
        ens_root=ens_root,
        mask_hr_file=mask_hr_file,
        mask_hr_var=mask_hr_var,
        eval_start=eval_start,
        eval_end=eval_end,
        variables=[var_name],
        labels=[requested_label],
    )

    loaded_models = loaded.data.get(var_name, {})
    if requested_label not in loaded_models:
        raise KeyError(
            f"Baseline {requested_label!r} was not found for {var_name!r}. "
            f"Returned labels: {list(loaded_models)}"
        )

    result = loaded_models[requested_label]
    del loaded
    gc.collect()
    return result


def _run_panel(
    *,
    panel_name: str,
    var_name: str,
    obs_file: str,
    obs_var: str,
    args,
    bins: np.ndarray,
) -> CorrelogramResult:
    start_time = perf_counter()

    data_array = _load_data_array(
        var_name=var_name,
        ens_root=args.ens_root,
        mask_hr_file=args.mask_hr_file,
        mask_hr_var=args.mask_hr_var,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        obs_file=obs_file,
        obs_var=obs_var,
        requested_label=args.label,
    )

    print(
        f"[{panel_name}] loaded {args.label} in {perf_counter() - start_time:.2f}s",
        flush=True,
    )

    payload = _prepare_payload(args.label, data_array)
    result = compute_correlogram_payload(
        payload,
        bins,
        replicate_chunk_size=args.replicate_chunk_size,
    )

    save_correlogram_result_csv(
        result,
        panel_name=panel_name,
        out_dir=args.csv_dir,
    )

    del data_array, payload
    gc.collect()

    print(
        f"[{panel_name}] total: {perf_counter() - start_time:.2f}s",
        flush=True,
    )

    return result


def main() -> None:
    base = Path(config.BASE_DIR) / "sasthana/Downscaling"
    dataset_root = base / "Downscaling_Models/Dataset_Setup_I_Chronological_12km"

    parser = argparse.ArgumentParser(
        description="Compute spatial correlograms for pr and tas."
    )
    parser.add_argument(
        "--ens_root",
        default=str(base / "GCM_pipeline/ALP-FINEv1.0/EnsPooled"),
    )
    parser.add_argument(
        "--obs_tas_file",
        default=str(dataset_root / "TabsD_step1_latlon.nc"),
    )
    parser.add_argument(
        "--obs_pr_file",
        default=str(dataset_root / "RhiresD_step1_latlon.nc"),
    )
    parser.add_argument("--obs_tas_var", default="TabsD")
    parser.add_argument("--obs_pr_var", default="RhiresD")
    parser.add_argument(
        "--mask_hr_file",
        default=str(dataset_root / "Swiss_Mask_HR.nc"),
    )
    parser.add_argument("--mask_hr_var", default="TabsD")

    parser.add_argument("--eval_start", type=int, default=2015)
    parser.add_argument("--eval_end", type=int, default=2023)
    parser.add_argument(
        "--csv_dir",
        default="Correlogram_bcsr_csv",
    )
    parser.add_argument(
        "--replicate_chunk_size",
        type=int,
        default=REPLICATE_CHUNK_SIZE,
    )
    parser.add_argument(
        "--label",
        required=True,
        help="One exact baseline label. Array jobs use one label per task.",
    )

    args = parser.parse_args()

    bins = np.arange(0.0, 12.0, 1.0, dtype=float)



    _run_panel(
        panel_name="pr",
        var_name="pr",
        obs_file=args.obs_pr_file,
        obs_var=args.obs_pr_var,
        args=args,
        bins=bins,
    )



    _run_panel(
        panel_name="tas",
        var_name="tas",
        obs_file=args.obs_tas_file,
        obs_var=args.obs_tas_var,
        args=args,
        bins=bins,
    )


if __name__ == "__main__":
    main()