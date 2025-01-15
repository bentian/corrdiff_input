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
- `os`: For file and directory operations.
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
import os
import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid) -> xr.Dataset:
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

def create_and_process_dataarray(name, stack_data, dims, coords, chunk_sizes) -> xr.DataArray:
    """
    Creates and processes an xarray.DataArray with specified
    dimensions, coordinates, and chunk sizes.

    Parameters:
    - name: Name of the DataArray.
    - stack_data: The stacked data to initialize the DataArray.
    - dims: A list of dimension names.
    - coords: A dictionary of coordinates for the DataArray.
    - chunk_sizes: A dictionary specifying chunk sizes for each dimension.

    Returns:
    - An xarray.DataArray with assigned coordinates and chunks.
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

def verify_dataset(dataset) -> tuple[bool, str]:
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
    missing_dims = [dim for dim in required_dims if dim not in dataset.dims]
    if missing_dims:
        return False, f"Missing required dimensions: {', '.join(missing_dims)}."
    if dataset.dims["south_north"] % 16 != 0:
        return False, "Dimensions 'south_north' and 'west_east' are not multiples of 16."

    # Check coordinates
    missing_coords = [coord for coord in required_coords if coord not in dataset.coords]
    if missing_coords:
        return False, f"Missing required coordinates: {', '.join(missing_coords)}."

    # Check data variables
    missing_vars = [var for var in required_vars if var not in dataset.data_vars]
    if missing_vars:
        return False, f"Missing required data variables: {', '.join(missing_vars)}."

    # All checks passed
    return True, "Dataset verification passed successfully."

def dump_regrid_netcdf(subdir, tread_pre_regrid, tread_post_regrid, \
                       era5_pre_regrid, era5_post_regrid) -> None:
    """
    Saves the provided datasets to NetCDF files within a specified subdirectory.

    Parameters:
    subdir (str): The subdirectory path where the NetCDF files will be saved.
    tread_pre_regrid (xr.Dataset): The TReAD dataset before regridding.
    tread_post_regrid (xr.Dataset): The TReAD dataset after regridding.
    era5_pre_regrid (xr.Dataset): The ERA5 dataset before regridding.
    era5_post_regrid (xr.Dataset): The ERA5 dataset after regridding.

    Returns:
    None
    """
    folder = f"./nc_dump/{subdir}/"
    os.makedirs(folder, exist_ok=True)

    tread_pre_regrid.to_netcdf(folder + "tread_pre_regrid.nc")
    tread_post_regrid.to_netcdf(folder + "tread_post_regrid.nc")
    era5_pre_regrid.to_netcdf(folder + "era5_pre_regrid.nc")
    era5_post_regrid.to_netcdf(folder + "era5_post_regrid.nc")

def is_local_testing() -> bool:
    """
    Determines if the current environment is set up for local testing.

    Returns:
    bool: True if the environment is for local testing; False otherwise.
    """
    return not os.path.exists("/lfs/archive/Reanalysis/")
