"""
Low-resolution dataset validation utilities.

This module provides helpers to validate that surface (SFC) and pressure-level
(PRS) NetCDF datasets conform to expected low-resolution formats.

It can also be run as a script:

    python verify_lowres_fmt.py sfc /path/to/folder_with_nc_files
    python verify_lowres_fmt.py prs /path/to/folder_with_nc_files

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
        Dimensions: time, lat, lon
        Coordinates:
            - time:      datetime64[ns]
            - lat:       1D
            - lon:       1D
        Variables (at least):
            - t2m:       (time, lat, lon)
            - tp:        (time, lat, lon)
            - u10:       (time, lat, lon)
            - v10:       (time, lat, lon)

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    errors: List[str] = []

    # 1. Required dims/coords
    _check_required_dims(ds, ["time", "lat", "lon"], errors)
    _check_required_coords(ds, ["time", "lat", "lon"], errors)

    if _has_true_errors(errors):
        _print_report("✗ DATASET FAILED SFC BASIC CHECKS:", errors)
        return False

    # 2. Time coordinate (object -> datetime64, or error)
    ds = _normalize_time_coord_to_datetime64(ds, errors)

    # 3. Numeric coord checks for lat/lon
    _check_numeric_coords(ds, ["lat", "lon"], errors)

    # 4. 1D lat/lon
    _check_1d_coords(ds, ["lat", "lon"], errors)

    # 5–6. Variables: presence, dims, dtype
    required_vars = ["tp", "t2m", "u10", "v10"]
    field_dims = ("time", "lat", "lon")
    _check_vars_dims_and_dtype(ds, required_vars, field_dims, errors)

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
            - lat
            - lon

        Coordinates:
            - time:      datetime64[ns]
            - level:     [500, 700, 850, 925] (hPa) subset of dataset levels
            - lat:       1D
            - lon:       1D

        Data variables (at least):
            - z, t, u, v: (time, level, lat, lon)

    Returns
    -------
    bool
        True if dataset passes all checks, False otherwise.
    """
    errors: List[str] = []

    # 1. Required dims/coords
    _check_required_dims(ds, ["time", "level", "lat", "lon"], errors)
    _check_required_coords(ds, ["time", "level", "lat", "lon"], errors)

    if _has_true_errors(errors):
        _print_report("✗ DATASET FAILED PRS BASIC CHECKS:", errors)
        return False

    # 2. Time coordinate (object -> datetime64, or error)
    ds = _normalize_time_coord_to_datetime64(ds, errors)

    # 3. Numeric coords (level, lat, lon)
    _check_numeric_coords(ds, ["level", "lat", "lon"], errors)

    # 4. 1D lat/lon
    _check_1d_coords(ds, ["lat", "lon"], errors)

    # 5. Required pressure levels (subset)
    try:
        level = ds["level"].values.astype(float)
        required_levels = np.array([500.0, 700.0, 850.0, 925.0], dtype=float)

        if level.ndim != 1:
            errors.append("Level coordinate must be 1D")
        else:
            missing_levels = required_levels[~np.isin(required_levels, level)]
            if missing_levels.size > 0:
                errors.append(
                    f"Missing required pressure levels: {missing_levels.tolist()}. "
                    f"Dataset contains: {level.tolist()}"
                )

            extra_levels = level[~np.isin(level, required_levels)]
            if extra_levels.size > 0:
                errors.append(
                    f"WARNING: dataset contains extra pressure levels "
                    f"beyond expected subset: {extra_levels.tolist()}"
                )

    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"Failed checking 'level' coordinate values: {exc}")

    # 6–7. Variables: presence, dims, dtype
    required_4d_vars = ["z", "t", "u", "v"]
    expected_var_dims = ("time", "level", "lat", "lon")
    _check_vars_dims_and_dtype(ds, required_4d_vars, expected_var_dims, errors)

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
            "WARNING: time coordinate is object dtype; attempting to convert to datetime64[ns]."
        )
        try:
            converted = np.array(time.values, dtype="datetime64[ns]")
            ds = ds.assign_coords(time=("time", converted))
        except (TypeError, ValueError) as conv_err:
            errors.append(f"Failed to convert object 'time' to datetime64[ns]: {conv_err}")
        return ds

    # Any other dtype is a hard error
    errors.append(
        "time coordinate must be datetime64[ns] or object convertible to datetime64[ns],"
        f"got {time.dtype}"
    )
    return ds


def _check_required_dims(
    ds: xr.Dataset,
    required_dims: Iterable[str],
    errors: List[str]
) -> None:
    """
    Append errors if required dimensions are missing and warnings if extra
    dimensions are present.
    """
    required_set = set(required_dims)

    # Missing required dims
    for dim in required_set:
        if dim not in ds.dims:
            errors.append(f"Missing required dimension '{dim}'")

    # Extra dims present in dataset but not required
    for dim in ds.dims:
        if dim not in required_set:
            errors.append(
                f"WARNING: unexpected extra dimension '{dim}' "
                f"(required: {sorted(required_set)})"
            )


def _check_required_coords(
    ds: xr.Dataset,
    required_coords: Iterable[str],
    errors: List[str]
) -> None:
    """
    Append errors if required coordinates are missing
    and warnings if extra coordinates are present.
    """
    required_set = set(required_coords)

    # Missing required coords
    for coord in required_set:
        if coord not in ds.coords:
            errors.append(f"Missing required coordinate '{coord}'")

    # Extra coordinates
    for coord in ds.coords:
        if coord not in required_set:
            errors.append(
                f"WARNING: unexpected extra coordinate '{coord}' "
                f"(required: {sorted(required_set)})"
            )


def _check_numeric_coords(
    ds: xr.Dataset,
    coord_names: Iterable[str],
    errors: List[str],
    check_fill: bool = True,
) -> None:
    """Check that given coordinates are numeric and (optionally) free of ERA5 fill values."""
    for name in coord_names:
        try:
            arr = ds[name].values
            if not np.issubdtype(arr.dtype, np.number):
                errors.append(
                    f"{name} coordinate must be numeric, got {arr.dtype}"
                )
            if check_fill and np.any(np.isclose(arr, 9.969e36)):
                errors.append(
                    f"{name} coordinate contains fill values (9.969e36)"
                )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"Failed checking coordinate '{name}': {exc}")


def _has_true_errors(messages: List[str]) -> bool:
    """Return True if any message is not a WARNING."""
    return any(not msg.strip().startswith("WARNING") for msg in messages)


def _check_1d_coords(
    ds: xr.Dataset,
    coord_names: Iterable[str],
    errors: List[str]
) -> None:
    """Check that given coordinates are 1D."""
    for name in coord_names:
        try:
            if ds[name].values.ndim != 1:
                errors.append(f"{name} must be 1D")
        except (KeyError, AttributeError, TypeError) as exc:
            errors.append(f"Failed checking dimensionality of '{name}': {exc}")


def _check_vars_dims_and_dtype(
    ds: xr.Dataset,
    var_names: Iterable[str],
    expected_dims: tuple,
    errors: List[str]
) -> None:
    """Check that given variables exist, have expected dims and numeric dtype."""
    required_set = set(var_names)
    actual_set = set(ds.data_vars)

    # Missing required variables
    missing = sorted(required_set - actual_set)
    if missing:
        errors.append(f"Missing required variable(s): {missing}")

    # Extra variables (warnings only)
    extra = sorted(actual_set - required_set)
    for var in extra:
        errors.append(
            f"WARNING: unexpected extra variable '{var}' (required: {sorted(required_set)})"
        )

    for name in var_names:
        if name not in ds.data_vars:
            continue  # already reported as missing
        try:
            var = ds[name]
            if var.dims != expected_dims:
                errors.append(f"{name} must have dims {expected_dims}, got {var.dims}")
            if not np.issubdtype(var.dtype, np.number):
                errors.append(f"{name} must be numeric, got dtype {var.dtype}")
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"Failed checking variable '{name}': {exc}")


def _print_report(header: str, errors: List[str]) -> None:
    """Pretty-print validation report."""
    print("=" * 50)

    print(header)
    for err in errors:
        print("  -", err)
    if "FAILED" in header:
        print("\nTotal errors:", len(errors))

    print("=" * 50)


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
    """Command-line entry point."""

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
