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
from typing import List, Iterable

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


def verify_lowres_sfc_format(ds: xr.Dataset) -> bool:
    """
    Verify that a dataset matches the expected SFC format.

    Expected format:
        Dimensions: time, latitude, longitude
        Coordinates:
            - time:      datetime64[ns]
            - latitude:  1D
            - longitude: 1D
        Variables (at least):
            - t2m:       (time, latitude, longitude)
            - tp:        (time, latitude, longitude)
            - u10:       (time, latitude, longitude)
            - v10:       (time, latitude, longitude)

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    label = "SFC"
    errors: List[str] = []

    # 1. Required dims/coords
    _check_required_dims(ds, ["time", "latitude", "longitude"], errors, label)
    _check_required_coords(ds, ["time", "latitude", "longitude"], errors, label)

    if errors:
        _print_report("✗ DATASET FAILED SFC BASIC CHECKS:", errors)
        return False

    # 2. Time coordinate (object -> datetime64, or error)
    ds = _normalize_time_coord_to_datetime64(ds, errors)

    # 3. Numeric coord checks for lat/lon
    _check_numeric_coords(ds, ["latitude", "longitude"], errors, label)

    # 4. 1D lat/lon
    _check_1d_coords(ds, ["latitude", "longitude"], errors, label)

    # 5–6. Variables: presence, dims, dtype
    required_vars = ["tp", "t2m", "u10", "v10"]
    field_dims = ("time", "latitude", "longitude")
    _check_vars_dims_and_dtype(ds, required_vars, field_dims, errors, label)

    # Final report
    if errors:
        _print_report("✗ DATASET FAILED SFC VALIDATION:", errors)
        return False

    print("✓ Dataset matches expected low-res SFC format.")
    return True


def verify_lowres_prs_format(ds: xr.Dataset) -> bool:
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
            - level:     [500, 700, 850, 925] (hPa) subset of dataset levels
            - latitude:  1D
            - longitude: 1D

        Data variables (at least):
            - time_bnds: (time, bnds), datetime64[ns]
            - z, t, u, v: (time, level, latitude, longitude)

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    label = "PRS"
    errors: List[str] = []

    # 1. Required dims/coords
    _check_required_dims(ds, ["time", "level", "latitude", "longitude"], errors, label)
    _check_required_coords(ds, ["time", "level", "latitude", "longitude"], errors, label)

    if errors:
        _print_report("✗ DATASET FAILED PRS BASIC CHECKS:", errors)
        return False

    # 2. Time coordinate (object -> datetime64, or error)
    ds = _normalize_time_coord_to_datetime64(ds, errors)

    # 3. Numeric coords (level, lat, lon)
    _check_numeric_coords(ds, ["level", "latitude", "longitude"], errors, label)

    # 4. 1D lat/lon
    _check_1d_coords(ds, ["latitude", "longitude"], errors, label)

    # 5. Required pressure levels (subset)
    try:
        level = ds["level"].values.astype(float)
        required_levels = np.array([500.0, 700.0, 850.0, 925.0], dtype=float)

        if level.ndim != 1:
            errors.append(f"{label}: level coordinate must be 1D")
        else:
            missing_levels = required_levels[~np.isin(required_levels, level)]
            if missing_levels.size > 0:
                errors.append(
                    f"{label}: missing required pressure levels: "
                    f"{missing_levels.tolist()}. "
                    f"Dataset contains: {level.tolist()}"
                )

            extra_levels = level[~np.isin(level, required_levels)]
            if extra_levels.size > 0:
                errors.append(
                    f"{label}: WARNING: dataset contains extra pressure levels "
                    f"beyond expected subset: {extra_levels.tolist()}"
                )

    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"{label}: failed checking 'level' coordinate values: {exc}")

    # 6–7. Variables: presence, dims, dtype
    required_4d_vars = ["z", "t", "u", "v"]
    expected_var_dims = ("time", "level", "latitude", "longitude")
    _check_vars_dims_and_dtype(ds, required_4d_vars, expected_var_dims, errors, label)

    # Final report
    if errors:
        _print_report("✗ DATASET FAILED PRS VALIDATION:", errors)
        return False

    print("✓ Dataset matches expected low-res PRS format.")
    return True

# ---------- Shared helpers ----------

def _normalize_time_coord_to_datetime64(ds: xr.Dataset, errors: list) -> xr.Dataset:
    """
    Ensure ds['time'] is datetime64[ns].

    - If already datetime64 -> OK
    - If object -> append warning to errors and try to convert in-place
    - Otherwise -> append hard error to errors

    Returns possibly-modified dataset (time coord converted).
    """
    try:
        time = ds["time"]
    except (KeyError, TypeError) as exc:
        errors.append(f"Failed to access 'time' coordinate: {exc}")
        return ds

    # Already correct dtype
    if np.issubdtype(time.dtype, np.datetime64):
        return ds

    # Object -> try to convert
    if time.dtype == object:
        errors.append(
            "WARNING: time coordinate is object dtype; attempting to convert to "
            "datetime64[ns]."
        )
        try:
            converted = np.array(time.values, dtype="datetime64[ns]")
            ds = ds.assign_coords(time=("time", converted))
        except (TypeError, ValueError) as conv_err:
            errors.append(
                "Failed to convert object 'time' to datetime64[ns]: "
                f"{conv_err}"
            )
        return ds

    # Any other dtype is a hard error
    errors.append(
        "time coordinate must be datetime64[ns] or object convertible to "
        f"datetime64[ns], got {time.dtype}"
    )
    return ds


def _check_required_dims(
    ds: xr.Dataset,
    required_dims: Iterable[str],
    errors: List[str],
    label: str,
) -> None:
    """Append errors if any required dimensions are missing."""
    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"{label}: missing required dimension '{dim}'")


def _check_required_coords(
    ds: xr.Dataset,
    required_coords: Iterable[str],
    errors: List[str],
    label: str,
) -> None:
    """Append errors if any required coordinates are missing."""
    for coord in required_coords:
        if coord not in ds.coords:
            errors.append(f"{label}: missing required coordinate '{coord}'")


def _check_numeric_coords(
    ds: xr.Dataset,
    coord_names: Iterable[str],
    errors: List[str],
    label: str,
    check_fill: bool = True,
) -> None:
    """Check that given coordinates are numeric and (optionally) free of ERA5 fill values."""
    for name in coord_names:
        try:
            arr = ds[name].values
            if not np.issubdtype(arr.dtype, np.number):
                errors.append(
                    f"{label}: {name} coordinate must be numeric, got {arr.dtype}"
                )
            if check_fill and np.any(np.isclose(arr, 9.969e36)):
                errors.append(
                    f"{label}: {name} coordinate contains fill values (9.969e36)"
                )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"{label}: failed checking coordinate '{name}': {exc}")


def _check_1d_coords(
    ds: xr.Dataset,
    coord_names: Iterable[str],
    errors: List[str],
    label: str,
) -> None:
    """Check that given coordinates are 1D."""
    for name in coord_names:
        try:
            if ds[name].values.ndim != 1:
                errors.append(f"{label}: {name} must be 1D")
        except (KeyError, AttributeError, TypeError) as exc:
            errors.append(
                f"{label}: failed checking dimensionality of '{name}': {exc}"
            )


def _check_vars_dims_and_dtype(
    ds: xr.Dataset,
    var_names: Iterable[str],
    expected_dims: tuple,
    errors: List[str],
    label: str,
) -> None:
    """Check that given variables exist, have expected dims and numeric dtype."""
    missing = [v for v in var_names if v not in ds.data_vars]
    if missing:
        errors.append(f"{label}: missing required variable(s): {missing}")

    for name in var_names:
        if name not in ds.data_vars:
            continue  # already reported as missing
        try:
            var = ds[name]
            if var.dims != expected_dims:
                errors.append(
                    f"{label}: {name} must have dims {expected_dims}, got {var.dims}"
                )
            if not np.issubdtype(var.dtype, np.number):
                errors.append(
                    f"{label}: {name} must be numeric, got dtype {var.dtype}"
                )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"{label}: failed checking variable '{name}': {exc}")


def _print_report(header: str, errors: List[str]) -> None:
    """Pretty-print validation report."""
    print("=" * 50)
    print(header)
    for err in errors:
        print("  -", err)
    if "FAILED" in header:
        print("\nTotal errors:", len(errors))
    print("=" * 50)
