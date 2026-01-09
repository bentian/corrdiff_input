"""
CORDEX data loader utilities for CorrDiff training.

This module builds standardized *high-resolution (HR)* and *low-resolution (LR)* xarray datasets
from preprocessed CORDEX NetCDF files, aligning variable names, dimensions, and coordinates to
the schema expected by CorrDiff training pipelines.

Key outputs
- HR: (time, south_north, west_east) with 2D XLAT/XLONG coordinates and renamed HR variables.
- LR: (time, level, south_north, west_east) upper-air predictors regridded onto the orography grid,
      plus a static (time, south_north, west_east) orography field.
- ORO: original static fields dataset used as the LR target grid (contains lat/lon/orog).

Notes
- Time is normalized to daily timestamps (floor to day).
- LR regridding uses `regrid_dataset` (xesmf bilinear + nearest extrapolation).
- Domain/config determine which files are loaded and whether a future-period suffix is used.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import xarray as xr

from .util import is_local_testing, regrid_dataset


DEBUG = True  # Set to True to enable debugging

TRAIN_SET: Dict[str, str] = {
    "ALPS": "CNRM-CM5",
    "NZ": "ACCESS-CM2"
}
IMPERFECT_SET: Dict[str, str] = {
    "ALPS": "MPI-ESM-LR",
    "NZ": "EC-Earth3"
}

CORDEX_HR_CHANNELS: Dict[str, dict] = {
    "pr": "precipitation",
    "tasmax": "max_surface_temperature"
}
CORDEX_LR_CHANNELS: List[Dict[str, dict]] = [
    *[
        {"name": name, "pressure": pressure, "variable": variable}
        for pressure in (500, 700, 850)
        for name, variable in (
            ("z", "geopotential_height"),
            ("q", "specific_humidity"),
            ("t", "temperature"),
            ("u", "eastward_wind"),
            ("v", "northward_wind"),
        )
    ],
    {'name': 'orog', 'variable': 'orography'}
]


def get_hr_channels() -> dict:
    """Returns Cordex HR channel list."""
    return CORDEX_HR_CHANNELS

def get_lr_channels() -> dict:
    """Returns Cordex LR channel list."""
    return CORDEX_LR_CHANNELS


# -------------------------------------------------------------------
# Directory / File paths
# -------------------------------------------------------------------

def data_root() -> Path:
    """Root directory for CORDEX train/test data."""
    return Path("../data/cordex" if is_local_testing() else "/lfs/home/corrdiff/data/40-CORDEX")


def train_dir(exp_domain: str, train_config: str) -> Path:
    """Train directory for a given domain/config."""
    return data_root() / f"{exp_domain}_domain" / "train" / train_config


def get_static_dataset(exp_domain: str, train_config: str) -> xr.Dataset:
    """Get static fields dataset (lat/lon/orog grid)."""
    return xr.open_mfdataset(
        train_dir(exp_domain, train_config) / "predictors" / "Static_fields.nc"
    )


def get_train_paths(exp_domain: str, train_config: str) -> list[Path]:
    """Get HR target + LR predictors paths for training."""
    base = train_dir(exp_domain, train_config)
    suffix = "_2080-2099" if train_config == "Emulator_hist_future" else ""
    model = TRAIN_SET[exp_domain]

    return [
        base / "target" / f"pr_tasmax_{model}_1961-1980{suffix}.nc",
        base / "predictors" / f"{model}_1961-1980{suffix}.nc",
    ]


def get_test_paths(exp_domain: str, test_config: str, perfect: bool) -> list[Path]:
    """Get LR predictors paths for test periods (historical/mid/end century)."""
    base = Path(data_root()) / f"{exp_domain}_domain" / "test"
    prefix = TRAIN_SET[exp_domain] if test_config == "TG" else IMPERFECT_SET[exp_domain]

    return [
        base / p / "predictors" / ("perfect" if perfect else "imperfect") / f"{prefix}_{y}.nc"
        for p, y in [
            ("historical","1981-2000"), ("mid_century","2041-2060"), ("end_century","2080-2099")
        ]
    ]


# -------------------------------------------------------------------
# HR / LR Datasets
# -------------------------------------------------------------------

def load_ds(path: Path, debug_slice: slice = slice("1961-01-01", "1961-01-31")) -> xr.Dataset:
    """Load a NetCDF dataset and normalize its time coordinate to daily resolution."""
    ds = xr.open_mfdataset(path).assign_coords(time=lambda d: d.time.dt.floor("D"))
    return ds.sel(time=debug_slice) if DEBUG else ds


def align_static_grid(
    statid_ds: xr.Dataset
) -> Tuple[xr.Dataset, xr.DataArray, xr.DataArray, Dict[str, str]]:
    """
    Align static_fields to a consistent grid interface.

    Returns
    -------
    grid : xr.Dataset
        Dataset with keys {"lat","lon"} for xesmf regridding (1D or 2D).
    xlat : xr.DataArray
        2D XLAT on (south_north, west_east)
    xlong : xr.DataArray
        2D XLONG on (south_north, west_east)
    dim_rename : dict
        Mapping to rename spatial dims to {"south_north","west_east"} for this grid.
        (Either {"y":"south_north","x":"west_east"} or {"lat":"south_north","lon":"west_east"})
    """
    DIM_YX_RENAME = {"y": "south_north", "x": "west_east"}
    DIM_LATLON_RENAME = {"lat": "south_north", "lon": "west_east"}
    
    # Choose which dim rename to use:
    # - curvilinear grid (ALPS): lat/lon are 2D on (y,x)
    # - regular grid (NZ): lat/lon are 1D dims (lat,lon)
    dim_rename = DIM_YX_RENAME if {"y", "x"}.issubset(statid_ds.dims) else DIM_LATLON_RENAME
    if not set(dim_rename).issubset(statid_ds.dims):
        raise ValueError("static_fields must contain dims (y,x) or (lat,lon)")

    grid = statid_ds[["lat", "lon"]]

    # make 2D XLAT/XLONG on (south_north, west_east)
    xlat, xlon = (
        (grid["lat"], grid["lon"]) if dim_rename is DIM_YX_RENAME
        else xr.broadcast(grid["lat"], grid["lon"])
    )

    return (
        grid,
        xlat.astype("float32").rename(dim_rename).rename("XLAT"),
        xlon.astype("float32").rename(dim_rename).rename("XLONG"),
        dim_rename,
    )


def get_train_datasets(
    exp_domain: str,
    train_config: str
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load and construct standardized CORDEX training datasets for CorrDiff.

    This function prepares both high-resolution (HR) and low-resolution (LR)
    training data for a given experiment domain and training configuration.
    It aligns all datasets to a common static grid, standardizes variable and
    dimension names, stacks LR pressure levels, regrids LR fields to the HR
    orography grid, and attaches static orography and grid coordinates.

    Parameters
    ----------
    exp_domain : str
        Experiment domain identifier (e.g., "ALPS", "NZ").
    train_config : str
        Training configuration identifier
        (e.g., "ESD_pseudo_reality", "Emulator_hist_future").

    Returns
    -------
    hr_out : xr.Dataset
        High-resolution target dataset with standardized variables and dimensions,
        on (time, south_north, west_east), including 2D XLAT/XLONG coordinates.
    lr_pre : xr.Dataset
        Low-resolution predictor dataset before regridding, stacked by pressure
        level on (time, level, lat/lon or y/x).
    lr_out : xr.Dataset
        Low-resolution dataset regridded onto the static (orography) grid, with
        standardized variable names and dimensions
        (time, level, south_north, west_east), including orography and XLAT/XLONG.
    grid_coords : xr.Dataset
        Dataset containing only the 2D grid coordinates (XLAT, XLONG) on
        (south_north, west_east), suitable for downstream reuse.
    """
    static_ds = get_static_dataset(exp_domain, train_config)
    grid, xlat_2d, xlong_2d, dim_rename = align_static_grid(static_ds)
    
    # HR
    target_path, predictor_path = get_train_paths(exp_domain, train_config)
    hr_out = (
        load_ds(target_path).drop_attrs()
        .rename({**dim_rename, **CORDEX_HR_CHANNELS})
        .assign_coords(XLAT=xlat_2d, XLONG=xlong_2d)
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
        [["XLAT", "XLONG", *CORDEX_HR_CHANNELS.values()]]
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )
    print(f"\nCordex HR [train] =>\n {hr_out}")

    # LR
    pred = load_ds(predictor_path)

    ch = [c for c in CORDEX_LR_CHANNELS if "pressure" in c]
    pressures = sorted({c["pressure"] for c in ch})
    rename_vars = {c["name"]: c["variable"] for c in ch}

    # Stack by level
    lr_pre = xr.Dataset({
        s: xr.concat(
            [pred[f"{s}_{p}"] for p in ps],
            dim=xr.IndexVariable("level", [float(p) for p in ps]),
        )
        for s in rename_vars
        for ps in ([p for p in pressures if f"{s}_{p}" in pred],)
    })

    # Regrid
    lr_rg = (
        regrid_dataset(lr_pre, grid)
        .rename({**dim_rename, **rename_vars})
        .transpose("time", "level", "south_north", "west_east")
        .chunk(time=1, level=1)
    )

    # Preprocess orography
    orography = (
        static_ds["orog"]
        .rename(dim_rename).astype("float32")
        .expand_dims(time=lr_rg.time)
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )

    # Final LR with orography
    lr_out = (
        xr.Dataset(
            coords={"time": lr_rg.time, "level": lr_rg.level, "XLAT": xlat_2d, "XLONG": xlong_2d},
            data_vars={**lr_rg.data_vars, "orography": orography},
            attrs={"regrid_method": "bilinear"},
        )
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
    )
    print(f"\nCordex LR [train] =>\n {lr_out}")

    # Grid coords
    grid_coords = xr.Dataset(
        coords={
            "XLAT": (("south_north", "west_east"), xlat_2d.data),
            "XLONG": (("south_north", "west_east"), xlong_2d.data),
        }
    )

    return hr_out, lr_pre, lr_out, grid_coords


def get_test_datasets(exp_domain: str, train_config: str, test_config: str, perfect: bool
                      ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    print(f"get_test_datasets: {exp_domain} {train_config} {test_config} {perfect}")
    print(get_test_paths(exp_domain, test_config, perfect))

    static_ds = get_static_dataset(exp_domain, train_config)
    grid, xlat_2d, xlong_2d, dim_rename = align_static_grid(static_ds)

    # LR
    pred = load_ds(get_test_paths(exp_domain, test_config, perfect))

    ch = [c for c in CORDEX_LR_CHANNELS if "pressure" in c]
    pressures = sorted({c["pressure"] for c in ch})
    rename_vars = {c["name"]: c["variable"] for c in ch}

    # Stack by level
    lr_pre = xr.Dataset({
        s: xr.concat(
            [pred[f"{s}_{p}"] for p in ps],
            dim=xr.IndexVariable("level", [float(p) for p in ps]),
        )
        for s in rename_vars
        for ps in ([p for p in pressures if f"{s}_{p}" in pred],)
    })

    # Regrid
    lr_rg = (
        regrid_dataset(lr_pre, grid)
        .rename({**dim_rename, **rename_vars})
        .transpose("time", "level", "south_north", "west_east")
        .chunk(time=1, level=1)
    )

    # Preprocess orography
    orography = (
        static_ds["orog"]
        .rename(dim_rename).astype("float32")
        .expand_dims(time=lr_rg.time)
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )

    # Final LR with orography
    lr_out = (
        xr.Dataset(
            coords={"time": lr_rg.time, "level": lr_rg.level, "XLAT": xlat_2d, "XLONG": xlong_2d},
            data_vars={**lr_rg.data_vars, "orography": orography},
            attrs={"regrid_method": "bilinear"},
        )
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
    )
    print(f"\nCordex LR [train] =>\n {lr_out}")

    # Grid coords
    grid_coords = xr.Dataset(
        coords={
            "XLAT": (("south_north", "west_east"), xlat_2d.data),
            "XLONG": (("south_north", "west_east"), xlong_2d.data),
        }
    )

    return None, lr_pre, lr_out, grid_coords
