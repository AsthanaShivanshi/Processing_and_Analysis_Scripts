import argparse
import re
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plotstyle import apply_paper_style, get_variable_cmap

from config import (
    CH2025_DIR,
    DATASETS_TRAINING_DIR,
    BC_COARSE_EQM_DIR,
    BC_COARSE_CDFT_DIR,
    BC_COARSE_dOTC_DIR,
    BCSR_EQM_Bilinear_DIR,
    BCSR_CDFT_Bilinear_DIR,
    BCSR_dOTC_Bilinear_DIR,
    BCSR_EQM_Bilinear_UNet_DIR,
    BCSR_CDFT_Bilinear_UNet_DIR,
    BCSR_dOTC_Bilinear_UNet_DIR,
    BCSR_EQM_Bilinear_UNet_DDIM_DIR,
    BCSR_CDFT_Bilinear_UNet_DDIM_DIR,
    BCSR_dOTC_Bilinear_UNet_DDIM_DIR,
)

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
matplotlib.rcParams["pdf.compression"] = 4

DDIM_SAMPLES_TO_PLOT = 4
PLOT_EXTENT = [5.8, 10.6, 45.7, 47.9]

METHOD_ALIASES = {
    "eqm_c": {"eqm_c"},
    "cdft": {"cdft", "cdf-t", "cdft-t", "cdftt", "cdf_t", "cdft_t"},
    "dotc": {"dotc", "d-otc", "d_otc"},
}
METHOD_DIR_TOKENS = {
    "eqm_c": ["EQM_C"],  
    "cdft": ["CDF-t", "CDFT-t", "CDFt", "CDFT", "cdft", "cdf-t"],
    "dotc": ["dOTC", "DOTC", "dotc"],
}
ALL_METHOD_TOKENS = sorted({t for v in METHOD_DIR_TOKENS.values() for t in v})

METHOD_DIRS = {
    "eqm_c": {
        "coarse": BC_COARSE_EQM_DIR,
        "bilinear": BCSR_EQM_Bilinear_DIR,
        "unet": BCSR_EQM_Bilinear_UNet_DIR,
        "ddim": BCSR_EQM_Bilinear_UNet_DDIM_DIR,
        "label": "EQM",
    },
    "cdft": {
        "coarse": BC_COARSE_CDFT_DIR,
        "bilinear": BCSR_CDFT_Bilinear_DIR,
        "unet": BCSR_CDFT_Bilinear_UNet_DIR,
        "ddim": BCSR_CDFT_Bilinear_UNet_DDIM_DIR,
        "label": "CDFt",
    },
    "dotc": {
        "coarse": BC_COARSE_dOTC_DIR,
        "bilinear": BCSR_dOTC_Bilinear_DIR,
        "unet": BCSR_dOTC_Bilinear_UNet_DIR,
        "ddim": BCSR_dOTC_Bilinear_UNet_DDIM_DIR,
        "label": "dOTC",
    },
}


def normalize_method(method: str) -> str:
    m = method.strip().lower().replace(" ", "")
    for k, vals in METHOD_ALIASES.items():
        if m in vals:
            return k
    raise ValueError(f"Unsupported method: {method}")


def scenario_candidates(scenario: str):
    return list(dict.fromkeys([scenario, "historical"]))


def extract_year_span(name: str):
    m = re.findall(r"(\d{4})-(\d{4})", name)
    return (int(m[0][0]), int(m[0][1])) if m else None


def year_compatible(path: Path, year: int) -> bool:
    ys = extract_year_span(path.name)
    return True if ys is None else ys[0] <= year <= ys[1]


def year_score(path: Path, year: int) -> float:
    ys = extract_year_span(path.name)
    if ys is None:
        return 3.0
    a, b = ys
    if a <= year <= b:
        return 25.0 - 0.02 * (b - a)
    return -abs(year - 0.5 * (a + b))


def is_output_file(path: Path) -> bool:
    n = path.name.lower()
    bad = ("corrfx", "corr_fx", "correction_function", "transfer_function", "quantile_map", "qmap_fx")
    return not any(b in n for b in bad)


def base_score(path: Path, var: str = None, prefer_unet=False, prefer_ddim=False):
    n = path.name.lower()
    s = 1.0 if n.endswith(".nc") else 0.0
    if var and f"_{var}_" in n:
        s += 12.0
    if "swiss" in n:
        s += 2.0
    if prefer_unet and "unet" in n:
        s += 10.0
    if prefer_ddim and "ddim" in n:
        s += 10.0
    return s


def expand_base_dirs(base_dir, method_key: str):
    p = Path(base_dir)
    cands = [p] + [p.parent / tok for tok in METHOD_DIR_TOKENS[method_key]]
    s = str(p)
    for old in ALL_METHOD_TOKENS:
        if old in s:
            for new in METHOD_DIR_TOKENS[method_key]:
                cands.append(Path(s.replace(old, new)))
    out, seen = [], set()
    for c in cands:
        sc = str(c)
        if sc not in seen:
            seen.add(sc)
            out.append(c)
    return out


def collect_files(base_dir, method_key, model, scenario, member, rcm, version, var):
    files = []
    for b in expand_base_dirs(base_dir, method_key):
        for sc in scenario_candidates(scenario):
            root = b / model / sc / member / rcm / version / "day" / var
            if root.exists():
                files.extend(root.glob("v*/*.nc"))
                files.extend(root.glob("*.nc"))
            fb = b / model / sc
            if fb.exists():
                files.extend(fb.rglob("*.nc"))
    if not files:
        for b in expand_base_dirs(base_dir, method_key):
            fb = b / model
            if fb.exists():
                files.extend(fb.rglob("*.nc"))
    return list(dict.fromkeys(files))


def supports_var_quick(ds: xr.Dataset, var: str) -> bool:
    if var in ds.data_vars:
        return True
    aliases = {"pr": ["pr", "precip", "precipitation", "rhiresd"], "tas": ["tas", "temp", "temperature", "tabsd"]}
    wanted = aliases.get(var, [var])
    for k in ds.data_vars:
        lk = k.lower()
        if any(w == lk or w in lk for w in wanted):
            return True
    return False


def pick_file(base_dir, method_key, model, scenario, member, rcm, version, var, year, kind):
    files = [f for f in collect_files(base_dir, method_key, model, scenario, member, rcm, version, var) if is_output_file(f)]
    if kind == "unet":
        u = [f for f in files if "unet" in f.name.lower()]
        files = u if u else files
    if kind == "ddim":
        files = [f for f in files if "ddim" in f.name.lower()]
    if not files:
        return None

    compat = [f for f in files if year_compatible(f, year)]
    pool = compat if compat else files

    if kind == "ddim":
        ok = []
        for f in pool:
            try:
                with xr.open_dataset(f) as ds:
                    if supports_var_quick(ds, var):
                        ok.append(f)
            except Exception:
                continue
        pool = ok if ok else pool

    pool.sort(
        key=lambda f: base_score(f, var=var, prefer_unet=(kind == "unet"), prefer_ddim=(kind == "ddim")) + year_score(f, year),
        reverse=True,
    )
    return pool[0] if pool else None


def find_ch2025_file(model, scenario, member, rcm, version, var, year):
    files = []
    for sc in scenario_candidates(scenario):
        root = Path(CH2025_DIR) / model / sc / member / rcm / version / "day" / var
        if root.exists():
            files.extend(root.glob("v*/*.nc"))
            files.extend(root.glob("*.nc"))
        fb = Path(CH2025_DIR) / model / sc
        if fb.exists():
            files.extend(fb.rglob("*.nc"))
    if not files:
        fb = Path(CH2025_DIR) / model
        if fb.exists():
            files.extend(fb.rglob("*.nc"))

    files = [f for f in files if is_output_file(f)]
    if not files:
        return None

    compat = [f for f in files if year_compatible(f, year)]
    pool = compat if compat else files

    def _score(f: Path):
        n = f.name.lower()
        s = base_score(f, var=var) + year_score(f, year)
        if "eqm" in n:
            s += 30.0
        if "eqm_c" in n:
            s -= 10.0
        if "cdft" in n or "cdf-t" in n or "dotc" in n:
            s -= 15.0
        return s

    pool.sort(key=lambda f: _score(f), reverse=True)
    return pool[0]


def pick_dataarray_for_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var in ds.data_vars and np.issubdtype(ds[var].dtype, np.number):
        return ds[var]
    aliases = {"pr": ["pr", "precip", "precipitation", "rhiresd"], "tas": ["tas", "temp", "temperature", "tabsd"]}
    wanted = aliases.get(var, [var])
    for k, da in ds.data_vars.items():
        lk = k.lower()
        if np.issubdtype(da.dtype, np.number) and any(w == lk or w in lk for w in wanted):
            return da
    for k, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.number) and "bnd" not in k.lower():
            return da
    raise ValueError(f"No numeric variable found for '{var}'")


def get_latlon(ds: xr.Dataset, da: xr.DataArray):
    if "lat" in da.coords and "lon" in da.coords:
        return da["lat"].values, da["lon"].values
    if "latitude" in da.coords and "longitude" in da.coords:
        return da["latitude"].values, da["longitude"].values
    for lon_name, lat_name in [("lon", "lat"), ("longitude", "latitude")]:
        if lon_name in ds and lat_name in ds:
            return ds[lat_name].values, ds[lon_name].values
    ydim, xdim = da.dims[-2], da.dims[-1]
    return np.arange(da.sizes[ydim]), np.arange(da.sizes[xdim])


def load_mask(base_dir: Path, hr=True):
    names = ["Swiss_Mask_HR.nc", "Mask_HR.nc"] if hr else ["Swiss_Mask_LR.nc", "Mask_LR.nc"]
    for n in names:
        p = base_dir / n
        if p.exists():
            with xr.open_dataset(p) as ds:
                v = "TabsD" if "TabsD" in ds.data_vars else next(iter(ds.data_vars))
                m = ds[v].load()
            return xr.where(np.isfinite(m), m > 0, False)
    raise FileNotFoundError(f"Mask not found in {base_dir}")


def apply_mask(frame: np.ndarray, lat, lon, mask: xr.DataArray):
    if frame.shape == mask.shape:
        return np.ma.masked_invalid(np.where(mask.values, frame, np.nan))
    mk = mask
    if "latitude" in mk.coords and "lat" not in mk.coords:
        mk = mk.rename({"latitude": "lat"})
    if "longitude" in mk.coords and "lon" not in mk.coords:
        mk = mk.rename({"longitude": "lon"})
    if "lat" in mk.coords and "lon" in mk.coords and np.ndim(lat) == 1 and np.ndim(lon) == 1:
        da = xr.DataArray(frame, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        mi = mk.interp(lat=da["lat"], lon=da["lon"], method="nearest")
        return np.ma.masked_invalid(da.where(mi > 0).values)
    return np.ma.masked_invalid(frame)


def resolve_common_time(ref_file, var, date_str):
    t = np.datetime64(date_str)
    if ref_file is None:
        return t
    with xr.open_dataset(ref_file) as ds:
        da = pick_dataarray_for_var(ds, var)
        if "time" not in da.dims:
            return t
        return np.datetime64(da.sel(time=t, method="nearest")["time"].values)


def load_field(file_path, var, target_time, keep_samples=False, mask=None):
    if file_path is None:
        return None, None, None
    with xr.open_dataset(file_path) as ds:
        da = pick_dataarray_for_var(ds, var)
        if "time" in da.dims:
            da = da.sel(time=target_time, method="nearest")

        sd = [d for d in ["sample", "samples", "member", "realization"] if d in da.dims]
        if keep_samples:
            if sd:
                da = da.transpose(sd[0], *[d for d in da.dims if d != sd[0]])
            else:
                da = da.expand_dims(sample=[0])
        else:
            if sd:
                da = da.mean(sd[0])
            da = da.expand_dims(sample=[0])

        lat, lon = get_latlon(ds, da)
        vals = da.values.astype(float, copy=False)
        if var == "pr":
            vals = np.clip(vals, 0.0, None)
        data = np.ma.masked_invalid(vals)

    if mask is not None:
        data = np.ma.stack([apply_mask(np.asarray(data[i]), lat, lon, mask) for i in range(data.shape[0])], axis=0)
    return data, lat, lon


def main():
    p = argparse.ArgumentParser(description="4x2 frame plot: method chain + DDIM samples")
    p.add_argument("--date", required=True)
    p.add_argument("--var", default="pr", choices=["pr", "tas"])
    p.add_argument("--method", default="EQM_C")
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", default="ssp370")
    p.add_argument("--member", default="r1i1p1f1")
    p.add_argument("--rcm", default="RegCM5-0")
    p.add_argument("--version", default="v1-r1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    apply_paper_style()
    year = int(args.date[:4])

    mkey = normalize_method(args.method)
    mlabel = METHOD_DIRS[mkey]["label"]
    md = METHOD_DIRS[mkey]

    mask_base = Path(DATASETS_TRAINING_DIR)
    mask_hr = load_mask(mask_base, hr=True)
    mask_lr = load_mask(mask_base, hr=False)

    coarse_file = pick_file(md["coarse"], mkey, args.model, args.scenario, args.member, args.rcm, args.version, args.var, year, "coarse")
    bilinear_file = pick_file(md["bilinear"], mkey, args.model, args.scenario, args.member, args.rcm, args.version, args.var, year, "bilinear")
    unet_file = pick_file(md["unet"], mkey, args.model, args.scenario, args.member, args.rcm, args.version, args.var, year, "unet")
    ddim_file = pick_file(md["ddim"], mkey, args.model, args.scenario, args.member, args.rcm, args.version, args.var, year, "ddim")
    ch2025_file = find_ch2025_file(args.model, args.scenario, args.member, args.rcm, args.version, args.var, year)

    print("coarse:", coarse_file)
    print("bilinear:", bilinear_file)
    print("unet:", unet_file)
    print("ddim:", ddim_file)
    print("ch2025:", ch2025_file)

    ref = coarse_file or bilinear_file or unet_file or ddim_file or ch2025_file
    target_time = resolve_common_time(ref, args.var, args.date)

    c, c_lat, c_lon = load_field(coarse_file, args.var, target_time, keep_samples=False, mask=mask_lr)
    b, b_lat, b_lon = load_field(bilinear_file, args.var, target_time, keep_samples=False, mask=mask_hr)
    u, u_lat, u_lon = load_field(unet_file, args.var, target_time, keep_samples=True, mask=mask_hr)
    d, d_lat, d_lon = load_field(ddim_file, args.var, target_time, keep_samples=True, mask=mask_hr)
    h, h_lat, h_lon = load_field(ch2025_file, args.var, target_time, keep_samples=False, mask=mask_hr)

    left = [
        (f"{mlabel} coarse BC (12 km)", c[0] if c is not None else None, c_lat, c_lon),
        (f"{mlabel} coarse + Bilinear (1 km)", b[0] if b is not None else None, b_lat, b_lon),
        (f"{mlabel} coarse + Bilinear + UNet mean", np.mean(u, axis=0) if u is not None else None, u_lat, u_lon),
        ("Fine-scale EQM", h[0] if h is not None else None, h_lat, h_lon),
    ]

    right = []
    
    if d is not None:
        for i in range(min(DDIM_SAMPLES_TO_PLOT, d.shape[0], 4)):
            right.append((f"{mlabel} DDIM sample {i+1}", d[i], d_lat, d_lon))
    while len(right) < 4:
        right.append((f"{mlabel} DDIM sample {len(right)+1}", None, None, None))

    if not any(fr is not None for _, fr, _, _ in left + right):
        raise FileNotFoundError("No plottable data found")

    vmin, vmax = (0.0, 20.0) if args.var == "pr" else (-20.0, 30.0)
    cmap = get_variable_cmap("precip" if args.var == "pr" else "temp")

    fig, axes = plt.subplots(
        4, 2, figsize=(10.2, 12.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor="white", constrained_layout=True
    )

    tags_l = ["(a)", "(b)", "(c)", "(d)"]
    tags_r = ["(e)", "(f)", "(g)", "(h)"]
    mesh = None

    for r in range(4):
        for col, panel, tag in [(0, left[r], tags_l[r]), (1, right[r], tags_r[r])]:
            ax = axes[r, col]
            title, frame, lat, lon = panel
            if frame is not None:
                mesh = ax.pcolormesh(
                    lon, lat, frame, cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="auto", transform=ccrs.PlateCarree(), rasterized=True
                )
                ax.coastlines(resolution="10m", linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{tag} {title}", fontsize=11, fontweight="bold", loc="left")
            ax.axis("off")

    if mesh is not None:
        units = "mm day$^{-1}$" if args.var == "pr" else "°C"
        label = "Precipitation" if args.var == "pr" else "Temperature"
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.01, shrink=0.97)
        cbar.set_label(f"{label} [{units}] ({str(target_time)[:10]})", fontsize=11)

    fig.suptitle(f"{mlabel} | {args.model} | {args.scenario} | {str(target_time)[:10]}", fontsize=13, fontweight="bold")

    out = Path(args.out) if args.out else (
        Path("Analysis") / "BCSR_Stats" / "Figures" / f"frames_2col_{mlabel}_{args.model}_{args.scenario}_{args.var}_{str(target_time)[:10]}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()