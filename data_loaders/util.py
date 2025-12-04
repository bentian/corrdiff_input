"""
Utilities for ERA5-format validation and regridding onto a target WRF-style grid.

This module provides small, focused helpers for:
- Detecting whether the code is running on the local workstation or an HPC
  environment (`is_local_testing`).
- Regridding arbitrary `xarray.Dataset` objects onto a target grid using
  bilinear interpolation via xESMF (`regrid_dataset`).
- Validating that input ERA5 datasets conform to expected surface (SFC) and
  pressure-level (PRS) conventions (`verify_lowres_sfc_format`,
  `verify_lowres_prs_format`).

The verification routines enforce a consistent ERA5 layout, checking:
- Required dimensions and coordinates (time, bnds, level, latitude, longitude)
- Coordinate dtypes (time as datetime64, others numeric)
- Presence and shapes of key data variables (e.g., `time_bnds`, `tp`, or a
  user-specified 4D variable)

These utilities are intended to be used as early sanity checks and
preprocessing steps before further regridding, stacking into CorrDiff-ready
tensors, or model training / inference.
"""
from pathlib import Path

import numpy as np
import xesmf as xe
import xarray as xr

def is_local_testing() -> bool:
    """
    Determines if the current environment is set up for local testing.

    Returns:
    bool: True if the environment is for local testing; False otherwise.
    """
    return not Path("/lfs/archive/Reanalysis/").exists()


def regrid_dataset(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Regrids the input dataset to match the target grid using bilinear interpolation.

    Parameters:
    ds (xr.Dataset): The source dataset to be regridded.
    grid (xr.Dataset): The target grid dataset defining the desired spatial dimensions.

    Returns:
    xr.Dataset: The regridded dataset aligned with the target grid.
    """
    # Regrid the dataset to the target grid:
    # - Use bilinear interpolation to regrid the data.
    # - Extrapolate by using the nearest valid source cell to extrapolate values for
    #   target points outside the source grid.
    remap = xe.Regridder(ds, grid, method="bilinear", extrap_method="nearest_s2d")

    # Regrid each time step while keeping the original coordinates and dimensions
    ds_regrid = xr.concat(
        [remap(ds.isel(time=i)).assign_coords(time=ds.time[i])
            for i in range(ds.sizes["time"])],
        dim="time"
    )

    return ds_regrid


def verify_lowres_sfc_format(ds: xr.Dataset):
    """
    Verify that a dataset matches the expected ERA5 SFC format.
    Raises descriptive errors if anything does not match.

    Expected format:
        Dimensions: time, bnds, latitude, longitude
        Coordinates: time, latitude, longitude
        Variables:
            - time_bnds: shape (time, bnds)
            - tp:        shape (time, latitude, longitude)
    """

    # ---- 1. Required dimensions ----
    required_dims = ["time", "bnds", "latitude", "longitude"]
    for dim in required_dims:
        if dim not in ds.dims:
            raise ValueError(f"Missing required dimension: '{dim}'")

    # ---- 2. Required coordinates ----
    if "time" not in ds.coords:
        raise ValueError("Missing required coordinate 'time'")
    if "latitude" not in ds.coords:
        raise ValueError("Missing required coordinate 'latitude'")
    if "longitude" not in ds.coords:
        raise ValueError("Missing required coordinate 'longitude'")

    # ---- 3. Coordinate dtype checks ----
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        raise TypeError("time coordinate must be datetime64[]")
    if not np.issubdtype(ds["latitude"].dtype, np.number):
        raise TypeError("latitude coordinate must be numeric")
    if not np.issubdtype(ds["longitude"].dtype, np.number):
        raise TypeError("longitude coordinate must be numeric")

    # ---- 4. Required variables ----
    if "time_bnds" not in ds.data_vars:
        raise ValueError("Missing required variable 'time_bnds'")
    if "tp" not in ds.data_vars:
        raise ValueError("Missing required variable 'tp' (precipitation)")

    # ---- 5. Variable shape verification ----
    if ds["time_bnds"].dims != ("time", "bnds"):
        raise ValueError(
            f"time_bnds must have dims ('time', 'bnds'), got {ds['time_bnds'].dims}"
        )

    if ds["tp"].dims != ("time", "latitude", "longitude"):
        raise ValueError(
            f"tp must have dims ('time', 'latitude', 'longitude'), got {ds['tp'].dims}"
        )

    print("✓ Dataset matches ERA5 SFC format.")


def verify_lowres_prs_format(ds: xr.Dataset, var_name: str):
    """
    Verify that a dataset matches the expected ERA5 PRS (pressure-level) format.

    Expected format (based on ERA5 PRS example):
        Dimensions:
            - time
            - bnds
            - level
            - latitude
            - longitude

        Coordinates:
            - time       (datetime64)
            - level      (numeric, usually pressure in hPa)
            - latitude   (numeric)
            - longitude  (numeric)

        Data variables:
            - time_bnds  (time, bnds)
            - <var_name> (time, level, latitude, longitude), e.g. 'u'

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    var_name : str, optional
        Name of the main 4D variable to check (default: 'u').

    Raises
    ------
    ValueError, TypeError
        If the dataset does not match the expected ERA5 PRS structure.
    """

    # ---- 1. Required dimensions ----
    required_dims = ["time", "bnds", "level", "latitude", "longitude"]
    for dim in required_dims:
        if dim not in ds.dims:
            raise ValueError(f"Missing required dimension: '{dim}'")

    # ---- 2. Required coordinates ----
    for coord in ["time", "level", "latitude", "longitude"]:
        if coord not in ds.coords:
            raise ValueError(f"Missing required coordinate '{coord}'")

    # ---- 3. Coordinate dtype checks ----
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        raise TypeError("time coordinate must be datetime64[]")

    for coord in ["level", "latitude", "longitude"]:
        if not np.issubdtype(ds[coord].dtype, np.number):
            raise TypeError(f"{coord} coordinate must be numeric")

    # ---- 4. Required variables ----
    if "time_bnds" not in ds.data_vars:
        raise ValueError("Missing required variable 'time_bnds'")
    if var_name not in ds.data_vars:
        raise ValueError(f"Missing required variable '{var_name}'")

    # ---- 5. Variable shape verification ----
    if ds["time_bnds"].dims != ("time", "bnds"):
        raise ValueError(
            f"time_bnds must have dims ('time', 'bnds'), "
            f"got {ds['time_bnds'].dims}"
        )

    expected_var_dims = ("time", "level", "latitude", "longitude")
    if ds[var_name].dims != expected_var_dims:
        raise ValueError(
            f"{var_name} must have dims {expected_var_dims}, "
            f"got {ds[var_name].dims}"
        )

    print(f"✓ Dataset matches ERA5 PRS format (variable='{var_name}').")
