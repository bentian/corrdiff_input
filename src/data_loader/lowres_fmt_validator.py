"""
Low-resolution dataset validation utilities.

This module provides helpers to validate that surface (SFC) and pressure-level
(PRS) ERA5-style NetCDF datasets conform to expected low-resolution formats.

It can also be run as a script:

    python lowres_fmt_validator.py sfc /path/to/folder_with_nc_files
    python lowres_fmt_validator.py prs /path/to/folder_with_nc_files

The folder should contain one or more .nc files which will be merged into a
single xarray.Dataset (via open_mfdataset) before validation.
"""
from pathlib import Path
from typing import List, Iterable
import argparse
import sys

import numpy as np
import xarray as xr

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

# ---------- CLI / main ----------

def _load_merged_dataset_from_folder(folder: str) -> xr.Dataset:
    """
    Load and merge all .nc files in a folder into a single Dataset.

    Parameters
    ----------
    folder : str
        Path to folder containing one or more NetCDF (.nc) files.

    Returns
    -------
    xr.Dataset
        Merged dataset from all .nc files in the folder.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no .nc files.
    OSError, ValueError
        If xarray fails to open or merge the files.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    nc_files = sorted(folder_path.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in folder: {folder}")

    # Combine by coordinates (typical for ERA5 time-split files).
    return xr.open_mfdataset(nc_files, combine="by_coords",
                             compat="no_conflicts", data_vars="all")


def main() -> None:
    """
    Command-line entry point.

    Usage
    -----
    python lowres_fmt_validator.py sfc /path/to/folder_with_nc_files
    python lowres_fmt_validator.py prs /path/to/folder_with_nc_files
    """
    parser = argparse.ArgumentParser(
        description="Validate low-res SFC/PRS datasets "
        "found in a folder of NetCDF files."
    )
    parser.add_argument(
        "kind",
        choices=["sfc", "prs"],
        help="Type of dataset to verify: 'sfc' for surface, 'prs' for pressure-level.",
    )
    parser.add_argument(
        "folder",
        help="Folder containing one or more .nc files to merge and validate.",
    )

    args = parser.parse_args()

    try:
        ds = _load_merged_dataset_from_folder(args.folder)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"✗ Failed to load dataset from folder '{args.folder}': {exc}")
        sys.exit(1)

    try:
        print(ds)
        ok = verify_lowres_sfc_format(ds) if args.kind == "sfc" \
                else verify_lowres_prs_format(ds)
    finally:
        # Make sure file handles are closed (especially for open_mfdataset)
        ds.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
