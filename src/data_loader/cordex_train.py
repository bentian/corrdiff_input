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
from typing import Dict, List, Tuple, Literal

import xarray as xr

from .util import is_local_testing, regrid_dataset

ExpDomain = Literal["ALPS", "SA", "NZ"]
TrainConfig = Literal["ESD_pseudo_reality", "Emulator_hist_future"]
TRAIN_SET: Dict[ExpDomain, str] = {
    "ALPS": "CNRM-CM5",
    "NZ": "ACCESS-CM2"
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


def get_hr_channels() -> dict:
    """Returns Cordex HR channel list."""
    return CORDEX_HR_CHANNELS

def get_lr_channels() -> dict:
    """Returns Cordex LR channel list."""
    return CORDEX_LR_CHANNELS


def get_data_dir() -> str:
    """Get data directory including train and test datasets."""
    return "../data/cordex/" if is_local_testing() else "/lfs/home/corrdiff/data/40-CORDEX"

def get_train_file_paths(
    exp_domain: ExpDomain,
    train_config: TrainConfig
) -> List[str]:
    """Get file paths to load HR target, LR predictors, and static fields."""
    folder_path = Path(get_data_dir()) / f"{exp_domain}_domain" / "train" / train_config
    extra = "_2080-2099" if train_config == "Emulator_hist_future" else ""

    return (
        folder_path / "target" / f"pr_tasmax_{TRAIN_SET[exp_domain]}_1961-1980{extra}.nc",
        folder_path / "predictors" / f"{TRAIN_SET[exp_domain]}_1961-1980{extra}.nc",
        folder_path / "predictors" / "Static_fields.nc"
    )


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
    # Choose which dim rename to use:
    # - curvilinear grid (ALPS): lat/lon are 2D on (y,x)
    # - regular grid (NZ): lat/lon are 1D dims (lat,lon)
    dim_rename = DIM_YX_RENAME if {"y", "x"}.issubset(statid_ds.dims) else DIM_LATLON_RENAME
    if not set(dim_rename).issubset(statid_ds.dims):
        raise ValueError("static_fields must contain dims (y,x) or (lat,lon)")

    grid = statid_ds[["lat", "lon"]]

    # make 2D XLAT/XLONG on (south_north, west_east)
    xlat, xlon = (
        (grid["lat"], grid["lon"])
        if dim_rename is DIM_YX_RENAME
        else xr.broadcast(grid["lat"], grid["lon"])
    )

    return (
        grid,
        xlat.astype("float32").rename(dim_rename).rename("XLAT"),
        xlon.astype("float32").rename(dim_rename).rename("XLONG"),
        dim_rename,
    )


def get_hr_dataset(target_path: Path, static_ds: xr.Dataset) -> xr.Dataset:
    """Load and standardize the CORDEX HR target dataset."""
    # Load target data
    target_ds = xr.open_mfdataset(target_path)
    _, XLAT, XLONG, dim_rename = align_static_grid(static_ds)

    return (
        target_ds.drop_attrs()                              # remove all attributes
        .assign_coords(time=target_ds.time.dt.floor("D"))   # normalize to daily timestamps
        # .sel(time=slice("1961-01-01", "1961-01-31"))        # DEBUG

        # Rename spatial dimensions and data variables to the standardized CorrDiff schema
        # (e.g. lat/lon or y/x → south_north/west_east, and HR variable names)
        .rename({**dim_rename, **CORDEX_HR_CHANNELS})

        .assign_coords(XLAT=XLAT, XLONG=XLONG)              # attach 2D coords to the grid
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
        [["XLAT", "XLONG", *CORDEX_HR_CHANNELS.values()]]   # keep needed coords & vars
        .transpose("time", "south_north", "west_east")      # enforce consistent dimension order
        .chunk(time=1)                                      # one timestep per chunk
    )


def get_lr_datasets(
    predictor_path: Path,
    static_ds: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load LR predictors + static fields, stack pressure levels,
    regrid to orography grid, and attach orography data.
    """
    # Load predictor data
    predictor_ds = xr.open_mfdataset(predictor_path)

    channels = [c for c in CORDEX_LR_CHANNELS if "pressure" in c]
    pressures = sorted({c["pressure"] for c in channels})
    rename_vars = {c["name"]: c["variable"] for c in channels}
    src_names = sorted(rename_vars)

    # Build (time, level, lat, lon) on the predictor grid
    lr = xr.Dataset({
        s: xr.concat(
            [
                predictor_ds[f"{s}_{p}"]
                .assign_coords(time=predictor_ds.time.dt.floor("D")) # normalize to daily timestamps
                # .sel(time=slice("1961-01-01", "1961-01-31"))         # DEBUG
                for p in pressures if f"{s}_{p}" in predictor_ds
            ],
            dim=xr.IndexVariable("level", [
                float(p) for p in pressures if f"{s}_{p}" in predictor_ds
            ]),
        ) for s in src_names
    })

    # Align static grid (handles both 1D lat/lon and 2D lat/lon)
    grid, XLAT, XLONG, dim_rename = align_static_grid(static_ds)

    # Regrid LR -> static grid, then standardize dims/names
    lr_regrid = (
        regrid_dataset(lr, grid)
        # Rename spatial dimensions and data variables to the standardized CorrDiff schema
        # (e.g. lat/lon or y/x → south_north/west_east, and HR variable names)
        .rename({**dim_rename, **rename_vars})
        .transpose("time", "level", "south_north", "west_east")
        .chunk(time=1, level=1)     # (one timestep × one level per chunk)
    )

    # Prepare static orography on (time, south_north, west_east)
    orography = (
        static_ds["orog"]
        .rename(dim_rename).astype("float32")
        .expand_dims(time=lr_regrid.time)
        .transpose("time", "south_north", "west_east")
        .chunk(time=1)
    )

    # Assemble final LR dataset
    lr_out = (
        xr.Dataset(
            coords={
                "time": lr_regrid.time,
                "level": lr_regrid.level,
                "XLAT": XLAT,
                "XLONG": XLONG,
            },
            data_vars={**lr_regrid.data_vars, "orography": orography},
            attrs={"regrid_method": "bilinear"},
        )
        .drop_vars(["lat", "lon", "south_north", "west_east"], errors="ignore")
    )

    return lr, lr_out


def get_datasets(
    exp_domain: ExpDomain = "ALPS",
    train_config: TrainConfig = "ESD_pseudo_reality",
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load and return standardized CORDEX training datasets.

    Parameters
    ----------
    exp_domain:
        Experiment domain identifier (e.g., "ALPS", "NZ").
    train_config:
        Training configuration identifier (e.g., "ESD_pseudo_reality", "Emulator_hist_future").

    Returns
    -------
    hr_out:
        Standardized HR target dataset on (time, south_north, west_east).
    lr_pre_regrid:
        LR dataset stacked into (time, level, lat, lon) *before* regridding.
    lr_out:
        LR dataset regridded onto the orography grid, with standardized names and coords.
    static_ds:
        Static fields dataset providing the LR target grid (lat/lon/orog).
    """
    target_path, predictor_path, static_fields_path = \
        get_train_file_paths(exp_domain, train_config)
    static_ds = xr.open_mfdataset(static_fields_path)

    hr_out = get_hr_dataset(target_path, static_ds)
    print(f"\nCordex HR [train] =>\n {hr_out}")

    lr_pre_regrid, lr_out = get_lr_datasets(predictor_path, static_ds)
    print(f"\nCordex LR [train] =>\n {lr_out}")

    return hr_out, lr_pre_regrid, lr_out, static_ds
