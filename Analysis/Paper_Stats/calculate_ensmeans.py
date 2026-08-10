from __future__ import annotations

from pathlib import Path
import xarray as xr

PROJECT_ROOT = Path("/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/")


def _find_sample_dim(da: xr.DataArray) -> str | None:
    for d in ("sample", "samples", "member"):
        if d in da.dims:
            return d
    return None


def save_mean_file(infile: Path, outfile: Path, vars_to_keep=("temp", "precip")) -> None:
    ds = xr.open_dataset(infile)

    out_vars = {}
    for v in vars_to_keep:
        if v not in ds:
            continue
        da = ds[v]
        sdim = _find_sample_dim(da)
        if sdim is None:
            out_vars[v] = da
        else:
            out_vars[v] = da.mean(dim=sdim, skipna=True)

    out_ds = xr.Dataset(out_vars)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(outfile)
    print(f"[ok] wrote {outfile}")


def main() -> None:
    files = [
        (
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0_mean.nc",
        ),
        (
            PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
            PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10_mean.nc",
        ),
    ]

    for infile, outfile in files:
        save_mean_file(infile, outfile)


if __name__ == "__main__":
    main()