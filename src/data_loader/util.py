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


def normalize_time_coord_to_datetime64(ds: xr.Dataset, errors: list) -> xr.Dataset:
    """
    Ensure ds['time'] is datetime64[ns].

    - If already datetime64 -> OK
    - If object -> append warning to errors and try to convert in-place
    - Otherwise -> append hard error to errors

    Returns possibly-modified dataset (time coord converted).
    """
    try:
        time = ds["time"]
    except Exception as e:
        errors.append(f"Failed to access 'time' coordinate: {e}")
        return ds

    # Already correct dtype
    if np.issubdtype(time.dtype, np.datetime64):
        return ds

    # Object -> try to convert
    if time.dtype == object:
        errors.append(
            "WARNING: time coordinate is object dtype; attempting to convert to datetime64[ns]."
        )
        try:
            converted = np.array(time.values, dtype="datetime64[ns]")
            ds = ds.assign_coords(time=("time", converted))
        except Exception as conv_e:
            errors.append(
                f"Failed to convert object 'time' to datetime64[ns]: {conv_e}"
            )
        return ds

    # Any other dtype is a hard error
    errors.append(
        f"time coordinate must be datetime64[ns] or object convertible to datetime64[ns], "
        f"got {time.dtype}"
    )
    return ds


def verify_lowres_sfc_format(ds: xr.Dataset):
    """
    Verify that a dataset matches the expected SFC format

    Expected format:
        Dimensions: time, latitude, longitude
        Coordinates:
            - time:      datetime64[ns]
            - latitude:  1D
            - longitude: 1D
        Variables (at least):
            - time_bnds: (time, bnds), datetime64[ns]
            - t2m:       (time, latitude, longitude)
            - tp:        (time, latitude, longitude)
            - u10:       (time, latitude, longitude)
            - v10:       (time, latitude, longitude)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    errors = []

    # ---- 1. Required dimensions ----
    required_dims = ["time", "latitude", "longitude"]
    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"Missing required dimension: '{dim}'")

    # ---- 2. Required coordinates ----
    for coord in ["time", "latitude", "longitude"]:
        if coord not in ds.coords:
            errors.append(f"Missing required coordinate '{coord}'")

    # Stop here if coords missing (prevents cascade failures)
    if errors:
        print("=" * 50)
        print("✗ DATASET FAILED BASIC SFC STRUCTURE CHECKS:")
        for e in errors:
            print("  -", e)
        print("=" * 50)
        return False

    # ---- 3. Coordinate dtype & basic checks ----
    # time (now tolerant of object -> converts)
    ds = normalize_time_coord_to_datetime64(ds, errors)

    # latitude / longitude numeric and finite
    for name in ["latitude", "longitude"]:
        try:
            arr = ds[name].values
            if not np.issubdtype(arr.dtype, np.number):
                errors.append(f"{name} coordinate must be numeric, got {arr.dtype}")
            if not np.isfinite(arr).all():
                errors.append(f"{name} coordinate contains non-finite values")
            if np.any(np.isclose(arr, 9.969e36)):
                errors.append(f"{name} coordinate contains fill values (9.969e36)")
        except Exception as e:
            errors.append(f"Failed checking coordinate '{name}': {e}")

    # ---- 4. Enforce 1-D shapes ----
    try:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if lat.ndim != 1:
            errors.append("latitude must be 1D")
        if lon.ndim != 1:
            errors.append("longitude must be 1D")
    except Exception as e:
        errors.append(f"Failed checking lat/lon dimensionality: {e}")

    # ---- 5. Required variables ----
    required_vars = ["tp", "t2m", "u10", "v10"]
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        errors.append(f"Missing required data variables: {missing}")

    # ---- 6. Variable dtype & shapes ----
    field_dims = ("time", "latitude", "longitude")
    for var_name in required_vars:
        if var_name not in ds.data_vars:
            continue  # already reported above
        try:
            var = ds[var_name]
            if var.dims != field_dims:
                errors.append(f"{var_name} must have dims {field_dims}, got {var.dims}")
            if not np.issubdtype(var.dtype, np.number):
                errors.append(f"{var_name} must be numeric, got dtype {var.dtype}")
        except Exception as e:
            errors.append(f"Failed checking variable '{var_name}': {e}")

    # ---- FINAL REPORT ----
    if errors:
        print("=" * 50)
        print("✗ DATASET FAILED SFC VALIDATION:")
        for e in errors:
            print("  -", e)
        print("\nTotal errors:", len(errors))
        print("=" * 50)
        return False

    print("✓ Dataset matches expected low-res SFC format.")
    return True


def verify_lowres_prs_format(ds: xr.Dataset):
    """
    Verify that a dataset matches the expected ERA5 PRS (pressure-level) format.

    Expected format:
        Dimensions:
            - time
            - level
            - latitude
            - longitude

        Coordinates:
            - time:      datetime64[ns]
            - level:     [500, 700, 850, 925] (hPa)
            - latitude:  1D
            - longitude: 1D

        Data variables (at least):
            - time_bnds: (time, bnds), datetime64[ns]
            - z, t, u, v: (time, level, latitude, longitude)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    errors = []

    # ---- 1. Required dimensions ----
    required_dims = ["time", "level", "latitude", "longitude"]
    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"Missing required dimension: '{dim}'")

    # ---- 2. Required coordinates ----
    for coord in ["time", "level", "latitude", "longitude"]:
        if coord not in ds.coords:
            errors.append(f"Missing required coordinate '{coord}'")

    # If basic structure is missing, don't try to access non-existent vars
    if errors:
        print("=" * 50)
        print("✗ DATASET FAILED BASIC PRS STRUCTURE CHECKS:")
        for e in errors:
            print("  -", e)
        print("\nTotal errors:", len(errors))
        print("=" * 50)
        return False

    # ---- 3. Coordinate dtype & validity checks ----
    # time (now tolerant of object -> converts)
    ds = normalize_time_coord_to_datetime64(ds, errors)

    # numeric coords: level, latitude, longitude
    for coord in ["level", "latitude", "longitude"]:
        try:
            arr = ds[coord].values
            if not np.issubdtype(arr.dtype, np.number):
                errors.append(f"{coord} coordinate must be numeric, got {arr.dtype}")
            if not np.isfinite(arr).all():
                errors.append(f"{coord} coordinate contains non-finite values")
            if np.any(np.isclose(arr, 9.969e36)):
                errors.append(
                    f"{coord} coordinate appears to contain fill values (9.969e36)"
                )
        except Exception as e:
            errors.append(f"Failed checking coordinate '{coord}': {e}")

    # ---- 4. Enforce lat & lon 1-D shapes ----
    try:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if lat.ndim != 1:
            errors.append("latitude must be 1D")
        if lon.ndim != 1:
            errors.append("longitude must be 1D")
    except Exception as e:
        errors.append(f"Failed checking lat/lon dimensionality: {e}")

    # ---- 5. Enforce required pressure levels ----
    try:
        level = ds["level"].values.astype(float)
        required_levels = np.array([500.0, 700.0, 850.0, 925.0], dtype=float)

        if level.ndim != 1:
            errors.append("level coordinate must be 1D")
        else:
            # ---- 1. Check subset requirement ----
            missing_levels = required_levels[~np.isin(required_levels, level)]
            if missing_levels.size > 0:
                errors.append(
                    f"Missing required pressure levels: {missing_levels.tolist()}. "
                    f"Dataset contains: {level.tolist()}"
                )

            # ---- 2. Warning: dataset contains extra levels ----
            extra_levels = level[~np.isin(level, required_levels)]
            if extra_levels.size > 0:
                errors.append(
                    "WARNING: Dataset contains extra pressure levels beyond expected subset: "
                    f"{extra_levels.tolist()}"
                )

            # ---- 3. Warning: exact match expected, but differs ----
            # (Only if no missing levels — otherwise main error already applies)
            if missing_levels.size == 0:
                expected_set = set(required_levels.tolist())
                actual_set = set(level.tolist())
                if expected_set != actual_set:
                    errors.append(
                        "WARNING: Level coordinate does not exactly match expected set "
                        f"{required_levels.tolist()}. Full dataset levels: {level.tolist()}"
                    )

            # ---- 4. Warning: order should be ascending ----
            if not np.all(np.diff(level) > 0):
                errors.append(
                    "WARNING: level coordinate is not strictly descending (ERA5 convention)."
                )

    except Exception as e:
        errors.append(f"Failed checking 'level' coordinate values: {e}")


    # ---- 6. Required variables ----
    required_4d_vars = ["z", "t", "u", "v"]
    missing_4d = [v for v in required_4d_vars if v not in ds.data_vars]
    if missing_4d:
        errors.append(f"Missing required 4D variable(s): {missing_4d}")

    # ---- 7. Variable dtype & shape verification ----
    expected_var_dims = ("time", "level", "latitude", "longitude")
    # 4D fields
    for v in required_4d_vars:
        if v not in ds.data_vars:
            continue  # already counted as missing above
        try:
            var = ds[v]
            if var.dims != expected_var_dims:
                errors.append(
                    f"{v} must have dims {expected_var_dims}, got {var.dims}"
                )
            if not np.issubdtype(var.dtype, np.number):
                errors.append(f"{v} must be numeric, got dtype {var.dtype}")
        except Exception as e:
            errors.append(f"Failed checking variable '{v}': {e}")

    # ---- FINAL REPORT ----
    if errors:
        print("=" * 50)
        print("✗ DATASET FAILED PRS VALIDATION:")
        for e in errors:
            print("  -", e)
        print("\nTotal errors:", len(errors))
        print("=" * 50)
        return False

    print("✓ Dataset matches expected low-res PRS format.")
    return True