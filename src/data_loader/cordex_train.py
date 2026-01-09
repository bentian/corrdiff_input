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
DIM_RENAME = {"y": "south_north", "x": "west_east"}

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


def get_file_paths(
    exp_domain: ExpDomain,
    train_config: TrainConfig
) -> List[str]:
    """Get file paths to load HR target, LR predictors, and static fields."""
    folder = "../data/cordex/" if is_local_testing() else "/lfs/home/corrdiff/40-CORDEX"
    folder_path = Path(folder) / f"{exp_domain}_domain" / "train" / train_config
    extra = "_2080-2099" if train_config == "Emulator_hist_future" else ""

    return (
        folder_path / "target" / f"pr_tasmax_{TRAIN_SET[exp_domain]}_1961-1980{extra}.nc",
        folder_path / "predictors" / f"{TRAIN_SET[exp_domain]}_1961-1980{extra}.nc",
        folder_path / "predictors" / "Static_fields.nc"
    )


def get_hr_dataset(target_path: Path) -> xr.Dataset:
    """Load and standardize the CORDEX HR target dataset."""
    target = xr.open_mfdataset(target_path)

    return (
        target
        .assign_coords(time=target["time"].dt.floor("D"))       # normalize to daily timestamps
        .sel(time=slice("1961-01-01", "1961-01-31"))            # TODO remove
        .rename({                                               # rename coords & variables
            "lat": "XLAT", "lon": "XLONG", **DIM_RENAME, **CORDEX_HR_CHANNELS
        })
        .drop_vars(DIM_RENAME.values())                         # drop 1D x/y coords (keep dims)
        [["XLAT", "XLONG", *CORDEX_HR_CHANNELS.values()]]       # keep only needed variables
        .transpose("time", *DIM_RENAME.values())                # enforce consistent dim order
        .chunk({"time": 1, "south_north": -1, "west_east": -1}) # chunk for dask processing
    )


def get_lr_datasets(
    predictors_path: Path,
    static_fields_path: Path
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load LR predictors + static fields, stack pressure levels,
    regrid to orography grid, and attach orography data.
    """
    predictors = xr.open_mfdataset(predictors_path)
    static_fields = xr.open_mfdataset(static_fields_path)

    channels = [c for c in CORDEX_LR_CHANNELS if "pressure" in c]
    pressures = sorted({c["pressure"] for c in channels})
    rename_vars = {c["name"]: c["variable"] for c in channels}
    src_names = sorted(rename_vars)

    # Build LR datset by:
    # 1) normalizing time to daily resolution,
    # 2) stacking pressure-specific variables into a single `level` dimension
    lr = xr.Dataset({
        s: xr.concat(
            [
                predictors[f"{s}_{p}"]
                .assign_coords(time=predictors.time.dt.floor("D"))
                .sel(time=slice("1961-01-01", "1961-01-31"))   # TODO remove
                for p in pressures if f"{s}_{p}" in predictors
            ],
            dim=xr.IndexVariable("level", [
                float(p) for p in pressures if f"{s}_{p}" in predictors
            ]),
        ) for s in src_names
    })

    # Regrid LR dataset
    grid = xr.Dataset({"lat": static_fields["lat"], "lon": static_fields["lon"]})
    lr_regrid = (
        regrid_dataset(lr, grid)
        .rename({**DIM_RENAME, **rename_vars})
        .transpose("time", "level", *DIM_RENAME.values())
        .chunk(time=1, level=1)
    )

    # Append orography to LR dataset
    lr_regrid["orography"] = (
        static_fields["orog"]
        .rename(DIM_RENAME)
        .expand_dims(time=lr_regrid.time)
        .transpose("time", *DIM_RENAME.values())
        .chunk(time=1)
    )

    lr_out = xr.Dataset(
        coords={
            "time": lr_regrid.time,
            "level": lr_regrid.level,
            "XLAT":  static_fields["lat"].rename(DIM_RENAME).astype("float32"),
            "XLONG": static_fields["lon"].rename(DIM_RENAME).astype("float32"),
        },
        data_vars=lr_regrid.data_vars,
        attrs={"regrid_method": "bilinear", **lr.attrs},
    ).drop_vars(["lat", "lon", *DIM_RENAME.values()], errors="ignore")

    return lr, lr_out, static_fields


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
    static_fields_path:
        Static fields dataset providing the LR target grid (lat/lon/orog).
    """
    target_path, predictors_path, static_fields_path = get_file_paths(exp_domain, train_config)
    return (
        get_hr_dataset(target_path),
        *get_lr_datasets(predictors_path, static_fields_path)
    )
