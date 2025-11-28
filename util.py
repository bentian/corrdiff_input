"""
Utility functions for dataset processing and analysis.

This module provides a collection of utility functions for tasks such as:
- Regridding datasets to match a target grid using bilinear interpolation.
- Creating and processing xarray DataArrays with specified dimensions, coordinates, and chunk sizes.
- Verifying datasets to ensure compliance with required dimensions, coordinates, and variables.
- Saving datasets to NetCDF format for debugging and storage.
- Determining the execution environment (local testing or production).

Functions:
- regrid_dataset: Regrids an xarray dataset to align with a target grid.
- create_and_process_dataarray: Creates and processes xarray.DataArray with specified parameters.
- verify_dataset: Verifies that a dataset meets required structure and properties.
- dump_regrid_netcdf: Saves regridded datasets to NetCDF files in a specified directory.
- is_local_testing: Checks whether the environment is set up for local testing.

Dependencies:
- `pathlib.Path`: For file and directory operations.
- `xesmf`: For regridding datasets.
- `xarray`: For handling labeled multi-dimensional arrays.

Example Usage:
    from util import regrid_dataset, create_and_process_dataarray

    # Regrid a dataset to match a target grid
    regridded_ds = regrid_dataset(source_dataset, target_grid)

    # Create and process a DataArray
    dataarray = create_and_process_dataarray(
        name="example_dataarray",
        stack_data=stacked_data,
        dims=["time", "lat", "lon"],
        coords={"time": time_values, "lat": lat_values, "lon": lon_values},
        chunk_sizes={"time": 1, "lat": 100, "lon": 100}
    )

    # Verify dataset structure
    is_valid, message = verify_dataset(dataset)
    print(message)
"""
from pathlib import Path
from typing import List, Dict

import numpy as np
import xesmf as xe
import xarray as xr

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

def create_and_process_dataarray(
    name: str,
    stack_data: np.ndarray,
    dims: List[str],
    coords: Dict[str, np.ndarray],
    chunk_sizes: Dict[str, int]
) -> xr.DataArray:
    """
    Creates and processes an xarray.DataArray with specified
    dimensions, coordinates, and chunk sizes.

    Parameters:
    - name (str): Name of the DataArray.
    - stack_data (np.ndarray): The stacked data to initialize the DataArray.
    - dims (List[str]): A list of dimension names.
    - coords (Dict[str, np.ndarray]): A dictionary of coordinates for the DataArray.
    - chunk_sizes (Dict[str, int]): A dictionary specifying chunk sizes for each dimension.

    Returns:
    - xr.DataArray: An xarray.DataArray with assigned coordinates and chunks.
    """
    # Create the DataArray
    dataarray = xr.DataArray(
        stack_data,
        dims=dims,
        coords=coords,
        name=name
    )

    # Assign daily floored time to the 'time' coordinate
    dataarray = dataarray.assign_coords(time=dataarray["time"].dt.floor("D"))

    # Chunk the DataArray
    dataarray = dataarray.chunk(chunk_sizes)

    return dataarray

def verify_dataset(ds: xr.Dataset) -> tuple[bool, str]:
    """
    Verifies an xarray.Dataset to ensure:
    1. Dimensions 'south_north' and 'west_east' are equal and both are multiples of 16.
    2. The dataset includes all specified coordinates and data variables.

    Parameters:
    - dataset: xarray.Dataset to verify.

    Returns:
    - A tuple (bool, str) where:
      - bool: True if the dataset passes all checks, False otherwise.
      - str: A message describing the result.
    """
    # Required dimensions, coordinates and data variables
    required_dims = [
        "time", "south_north", "west_east", "cwb_channel", "era5_channel"
    ]
    required_coords = [
        "time", "XLONG", "XLAT", "cwb_pressure", "cwb_variable",
        "era5_scale", "era5_pressure", "era5_variable"
    ]
    required_vars = [
        "cwb", "cwb_center", "cwb_scale", "cwb_valid",
        "era5", "era5_center", "era5_valid"
    ]

    # Check required dimensions
    missing_dims = [dim for dim in required_dims if dim not in ds.dims]
    if missing_dims:
        return False, f"Missing required dimensions: {', '.join(missing_dims)}."
    if ds.dims["south_north"] != ds.dims["west_east"]:
        return False, "Dimensions 'south_north' and 'west_east' are not equal."
    if ds.dims["south_north"] % 16 != 0:
        return False, "Dimensions 'south_north' and 'west_east' are not multiples of 16."

    # Check coordinates
    missing_coords = [coord for coord in required_coords if coord not in ds.coords]
    if missing_coords:
        return False, f"Missing required coordinates: {', '.join(missing_coords)}."

    # Check data variables
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        return False, f"Missing required data variables: {', '.join(missing_vars)}."

    # All checks passed
    return True, "Dataset verification passed successfully."

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

def dump_regrid_netcdf(
    subdir: str,
    hr_pre_regrid: xr.Dataset,
    hr_post_regrid: xr.Dataset,
    lr_pre_regrid: xr.Dataset,
    lr_post_regrid: xr.Dataset
) -> None:
    """
    Saves the provided datasets to NetCDF files within a specified subdirectory.

    Parameters:
    subdir (str): The subdirectory path where the NetCDF files will be saved.
    hr_pre_regrid (xr.Dataset): The high-resolution dataset before regridding.
    hr_post_regrid (xr.Dataset): The high-resolution dataset after regridding.
    lr_pre_regrid (xr.Dataset): The low-resolution dataset before regridding.
    lr_post_regrid (xr.Dataset): The low-resolution dataset after regridding.

    Returns:
    None
    """
    folder = Path(f"./nc_dump/{subdir}")
    folder.mkdir(parents=True, exist_ok=True)

    for dataset, name in [
        (hr_pre_regrid, "highres_pre_regrid.nc"),
        (hr_post_regrid, "highres_post_regrid.nc"),
        (lr_pre_regrid, "lowres_pre_regrid.nc"),
        (lr_post_regrid, "lowres_post_regrid.nc")
    ]:
        dataset.to_netcdf(folder / name)

def is_local_testing() -> bool:
    """
    Determines if the current environment is set up for local testing.

    Returns:
    bool: True if the environment is for local testing; False otherwise.
    """
    return not Path("/lfs/archive/Reanalysis/").exists()
