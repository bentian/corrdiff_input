"""
CORDEX data loaders and preprocessing utilities for CorrDiff.

This module provides a unified interface for loading, standardizing, and
preprocessing CORDEX high-resolution (HR) and low-resolution (LR) datasets
for use in the CorrDiff training and evaluation pipelines.

Core responsibilities
---------------------
- Load CORDEX HR target datasets and LR predictor datasets from NetCDF files
- Normalize temporal coordinates to daily resolution
- Align all datasets to a shared static grid (lat/lon/orography), supporting
  both curvilinear (y, x) and regular (lat, lon) grids
- Stack LR pressure-level variables into a unified `level` dimension
- Regrid LR fields onto the static target grid using xESMF
- Standardize variable names, dimension names, and coordinate conventions
  to the CorrDiff schema
- Attach static fields (orography) and 2D grid coordinates (XLAT, XLONG)
- Construct synthetic HR datasets for test configurations where true HR
  targets are unavailable

Supported workflows
-------------------
- Training data preparation (HR targets + LR predictors)
- Test-time data preparation (LR predictors + synthetic HR placeholders)
- Multiple experiment domains (e.g., ALPS, NZ)
- Multiple training and test configurations (e.g., pseudo-reality, TG, OOSG)
- Perfect and imperfect model predictors

Design notes
------------
- All large arrays are kept Dask-backed to support scalable, lazy computation
- Grid alignment and regridding logic is factored into reusable helper functions
- Returned datasets conform to the CorrDiff input schema and are ready for
  downstream normalization, batching, and Zarr serialization

This module is intended to be used by higher-level dataset assembly and
serialization code (e.g., CorrDiff Zarr generators), rather than directly
by model training code.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import xarray as xr
import dask.array as da

from .util import is_local_testing, regrid_dataset


DEBUG = False  # Set to True to enable debugging

GCM_SET: Dict[str, Dict[str, str]] = {
    "ALPS": {"TG": "CNRM-CM5", "OOSG": "MPI-ESM-LR"},
    "NZ": {"TG": "ACCESS-CM2", "OOSG": "EC-Earth3"},
}
DIM_YX_RENAME = {"y": "south_north", "x": "west_east"}
DIM_LATLON_RENAME = {"lat": "south_north", "lon": "west_east"}

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

# -------------------------------------------------------------------
# Channel list getters
# -------------------------------------------------------------------

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
    model = GCM_SET[exp_domain]["TG"]

    return [
        base / "target" / f"pr_tasmax_{model}_1961-1980{suffix}.nc",
        base / "predictors" / f"{model}_1961-1980{suffix}.nc",
    ]


def get_test_paths(exp_domain: str, test_config: str, perfect: bool) -> list[Path]:
    """Get LR predictors paths for test periods (historical/mid/end century)."""
    base = Path(data_root()) / f"{exp_domain}_domain" / "test"
    prefix = GCM_SET[exp_domain][test_config]

    return [
        base / p / "predictors" / ("perfect" if perfect else "imperfect") / f"{prefix}_{y}.nc"
        for p, y in [
            ("historical","1981-2000"), ("mid_century","2041-2060"), ("end_century","2080-2099")
        ]
    ]


# -------------------------------------------------------------------
# Helpers to create HR / LR datasets
# -------------------------------------------------------------------

def _load_train_ds(path: Path) -> xr.Dataset:
    """Load a NetCDF dataset and normalize its time coordinate to daily resolution."""
    ds = xr.open_mfdataset(path).assign_coords(time=lambda d: d.time.dt.floor("D"))
    return ds.sel(time=slice("1961-01-01", "1961-01-31")) if DEBUG else ds


def _align_static_grid(
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


def _stack_levels(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    """Stack pressure-level predictor variables into a single `level` dimension."""
    ch = [c for c in CORDEX_LR_CHANNELS if "pressure" in c]
    pressures = sorted({c["pressure"] for c in ch})
    rename_vars = {c["name"]: c["variable"] for c in ch}

    lr_pre = xr.Dataset({
        s: xr.concat(
            [ds[f"{s}_{p}"] for p in ps],
            dim=xr.IndexVariable("level", [float(p) for p in ps]),
        )
        for s in rename_vars
        for ps in ([p for p in pressures if f"{s}_{p}" in ds],)
    })

    return lr_pre, rename_vars


def _finalize_lr(
    lr_pre: xr.Dataset,
    grid: xr.Dataset,
    static_ds: xr.Dataset,
    XLAT: xr.DataArray,
    XLONG: xr.DataArray,
    dim_rename: dict[str, str],
    rename_vars: dict[str, str],
) -> xr.Dataset:
    """
    Regrid, standardize, and finalize a low-resolution (LR) dataset.

    This function regrids the stacked LR predictors to the static target grid,
    standardizes dimension and variable names, attaches static orography, and
    adds 2D grid coordinates (XLAT, XLONG).

    Parameters
    ----------
    lr_pre : xr.Dataset
        LR dataset stacked by pressure level, before regridding.
    grid : xr.Dataset
        Target grid dataset containing `lat` and `lon` for xESMF regridding.
    static_ds : xr.Dataset
        Static fields dataset containing orography.
    XLAT, XLONG : xr.DataArray
        2D latitude and longitude arrays on (south_north, west_east).
    dim_rename : dict[str, str]
        Mapping from source spatial dimensions to
        ("south_north", "west_east").
    rename_vars : dict[str, str]
        Mapping from short variable names to standardized CorrDiff names.

    Returns
    -------
    xr.Dataset
        Final LR dataset on the CorrDiff grid with standardized variables,
        dimensions, orography, and grid coordinates.
    """
    lr_rg = (
        regrid_dataset(lr_pre, grid)
        .rename({**dim_rename, **rename_vars})
        .transpose("time", "level", "south_north", "west_east")
        .chunk(time=1, level=1)
    )

    orography = (
        static_ds["orog"]
        .rename(dim_rename).astype("float32")
        .expand_dims(time=lr_rg.time)
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )

    return xr.Dataset(
        coords={"time": lr_rg.time, "level": lr_rg.level, "XLAT": XLAT, "XLONG": XLONG},
        data_vars={**lr_rg.data_vars, "orography": orography},
        attrs={"regrid_method": "bilinear"},
    ).drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")


def _grid_coords_only(XLAT: xr.DataArray, XLONG: xr.DataArray) -> xr.Dataset:
    """
    Construct a minimal grid-coordinates dataset with
    2D latitude and longitude arrays on (south_north, west_east)
    """
    return xr.Dataset(coords={
        "XLAT": (("south_north", "west_east"), XLAT.data),
        "XLONG": (("south_north", "west_east"), XLONG.data),
    })


def _fake_hr_from_lr(lr: xr.Dataset) -> xr.Dataset:
    """Create a synthetic HR dataset on the LR grid with all values set to zero."""
    shape = (lr.sizes["time"], lr.sizes["south_north"], lr.sizes["west_east"])
    chunks = (1, shape[1], shape[2])
    z = da.zeros(shape, chunks=chunks, dtype="float32")

    return xr.Dataset(
        data_vars={
            var: (("time", "south_north", "west_east"), z)
            for var in CORDEX_HR_CHANNELS.values()
        },
        coords={"time": lr.time, "XLAT": lr.XLAT, "XLONG": lr.XLONG},
    )


# -------------------------------------------------------------------
# Entry for HR & LR train / test datasets
# -------------------------------------------------------------------

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
    grid, XLAT, XLONG, dim_rename = _align_static_grid(static_ds)
    target_path, predictor_path = get_train_paths(exp_domain, train_config)

    hr_out = (
        _load_train_ds(target_path).drop_attrs()
        .rename({**dim_rename, **CORDEX_HR_CHANNELS})
        .assign_coords(XLAT=XLAT, XLONG=XLONG)
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
        [["XLAT", "XLONG", *CORDEX_HR_CHANNELS.values()]]
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )

    lr_pre, rename_vars = _stack_levels(_load_train_ds(predictor_path))
    lr_out = _finalize_lr(lr_pre, grid, static_ds, XLAT, XLONG, dim_rename, rename_vars)

    return hr_out, lr_pre, lr_out, _grid_coords_only(XLAT, XLONG)


def get_test_datasets(exp_domain: str, train_config: str, test_config: str, perfect: bool
                      ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load and construct standardized CORDEX test datasets for CorrDiff.

    This function prepares CORDEX test-time inputs by loading low-resolution (LR)
    predictor datasets for the specified experiment domain and test configuration,
    regridding them to the static target grid, and attaching orography and grid
    coordinates. Since no high-resolution (HR) targets are available at test time,
    a synthetic (zero-filled) HR dataset is generated on the same grid to satisfy
    the CorrDiff interface.

    Parameters
    ----------
    exp_domain : str
        Experiment domain identifier (e.g., "ALPS", "NZ").
    train_config : str
        Training configuration used to select the static grid and reference model.
    test_config : str
        Test configuration identifier (e.g., "TG", "OOSG").
    perfect : bool
        If True, load “perfect-model” LR predictors; otherwise load imperfect predictors.

    Returns
    -------
    hr_fake : xr.Dataset
        Synthetic HR dataset on (time, south_north, west_east) with all values set
        to zero, matching the expected CorrDiff HR schema.
    lr_pre : xr.Dataset
        LR predictor dataset before regridding, stacked by pressure level on
        (time, level, lat/lon or y/x).
    lr_out : xr.Dataset
        LR dataset regridded onto the static grid, with standardized variables,
        dimensions, orography, and 2D grid coordinates.
    grid_coords : xr.Dataset
        Dataset containing only the grid coordinates (`XLAT`, `XLONG`) on
        (south_north, west_east), suitable for downstream reuse.
    """
    static_ds = get_static_dataset(exp_domain, train_config)
    grid, XLAT, XLONG, dim_rename = _align_static_grid(static_ds)

    pred = xr.open_mfdataset(
        get_test_paths(exp_domain, test_config, perfect),
        combine="by_coords",
        compat="no_conflicts",
    ).assign_coords(time=lambda d: d.time.dt.floor("D"))
    if DEBUG:
        pred = pred.sel(time=slice("1981-01-01", "1981-01-31"))

    lr_pre, rename_vars = _stack_levels(pred)
    lr_out = _finalize_lr(lr_pre, grid, static_ds, XLAT, XLONG, dim_rename, rename_vars)

    return _fake_hr_from_lr(lr_out), lr_pre, lr_out, _grid_coords_only(XLAT, XLONG)
