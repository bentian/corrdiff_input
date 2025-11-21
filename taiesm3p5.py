"""
TaiESM 3.5km Dataset Processing Module.

This module provides utilities for processing TaiESM 3.5km (Taiwan ReAnalysis Dataset) data.
It includes functions to retrieve, preprocess, and regrid datasets, as well as to generate
data arrays and compute various metrics such as mean, standard deviation, and validity.

Features:
- Retrieve and process TaiESM 3.5km datasets for specific date ranges.
- Regrid datasets to match a specified spatial grid.
- Generate CWB-related DataArrays (e.g., pressure levels, variables, mean, and standard deviation).
- Support for calculating aggregated metrics over time and spatial dimensions.

Functions:
- get_file_paths: Generate file paths for TaiESM 3.5km datasets based on a date range.
- get_dataset: Retrieve and preprocess TaiESM 3.5km datasets, including regridding.
- get_cwb_pressure: Create a DataArray for TaiESM 3.5km pressure levels.
- get_cwb_variable: Create a DataArray for TaiESM 3.5km variables.
- get_cwb: Generate a stacked CWB DataArray from TaiESM 3.5km output variables.
- get_cwb_center: Compute mean values for TaiESM 3.5km variables over time and spatial dimensions.
- get_cwb_scale: Compute standard deviation for TaiESM 3.5km variables over
                 time and spatial dimensions.
- get_cwb_valid: Generate a validity mask for TaiESM 3.5km time steps.
- generate_output: Produce processed TaiESM 3.5km outputs and associated metrics.

Dependencies:
- `pathlib.Path`: For file path manipulation.
- `dask.array`: For efficient handling of large datasets with lazy evaluation.
- `numpy`: For numerical operations.
- `pandas`: For handling date ranges and date-time operations.
- `xarray`: For managing multi-dimensional labeled datasets.

Usage Example:
    from taiesm3p5 import generate_output

    # Define inputs
    ref_grid = xr.open_dataset("path/to/ref_grid.nc")
    start_date = "20220101"
    end_date = "20220131"

    # Generate TaiESM 3.5km outputs
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, pre_regrid, regridded = \
        generate_output(ref_grid, start_date, end_date)

    print("Processed TaiESM 3.5km data:", regridded)

Notes:
- Ensure that the input datasets conform to expected variable names and coordinate structures.
- The module is optimized for handling large datasets efficiently using Dask.

"""
from pathlib import Path
from typing import List, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import is_local_testing, create_and_process_dataarray

TAIESM_3P5_CHANNELS = {
    # Baseline
    "RAINNC": "precipitation",
    "T2MEAN": "temperature_2m",
    "U10MEAN": "eastward_wind_10m",
    "V10MEAN": "northward_wind_10m",
}

def get_data_dir(ssp_suffix: str) -> str:
    """
    Return the base directory for the TaiESM 3.5 km dataset based on the
    execution environment.

    Parameters
    ----------
    ssp_suffix (str): Scenario suffix used to select the TaiESM dataset directory on BIG server.

    Returns
    -------
    str
        Path to the TaiESM 3.5 km data directory. This is:
        - `./data/taiesm3p5` when running locally (as detected by is_local_testing())
        - `/lfs/archive/TCCIP_data/TaiESM-WRF/TAIESM_tw3.5km_<suffix>` when
          running on the BIG server.

    Notes
    -----
    This helper centralizes environment-aware path logic so other code does not
    need to handle local vs. remote directory differences.
    """
    return "./data/taiesm3p5" if is_local_testing() else \
            f"/lfs/archive/TCCIP_data/TaiESM-WRF/TAIESM_tw3.5km_{ssp_suffix}"

def get_file_paths(folder: str, start_date: str, end_date: str) -> List[str]:
    """
    Generate a list of file paths for the specified date range.

    Parameters:
        folder (str): The directory containing the files.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.

    Returns:
        list: A list of file paths corresponding to each month in the date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    return [folder_path / f"wrfday_d01_{yyyymm}.nc" for yyyymm in date_range]

def get_dataset(grid: xr.Dataset, start_date: str, end_date: str,
                ssp_suffix: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve and process TaiESM 3.5km dataset within the specified date range.

    Parameters:
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.
        ssp_suffix (str): Scenario suffix used to select the TaiESM dataset directory.

    Returns:
        tuple: A tuple containing the original and regridded TaiESM 3.5km datasets.
    """
    surface_var_names = list(TAIESM_3P5_CHANNELS.keys())
    start_datetime = pd.to_datetime(str(start_date), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Read surface level data.
    file_paths = get_file_paths(get_data_dir(ssp_suffix), start_date, end_date)
    surface_ds = xr.open_mfdataset(
        file_paths,
        preprocess=lambda ds: (
            ds[surface_var_names].assign_coords(            # attach new time coord
                time=pd.to_datetime(ds["Times"].astype(str), format="%Y-%m-%d_%H:%M:%S")
            )
            .rename({"Time": "time"})                       # unify time dimension name
            .drop_vars("Times", errors="ignore")            # remove the raw WRF Times bytes
            .sel(time=slice(start_datetime, end_datetime))  # select requested dates
        )
    )

    def crop_to_center(ds: xr.Dataset, target_shape=(304, 304)) -> xr.Dataset:
        """Crop dataset to a centered (south_north, west_east) subdomain."""
        ny, nx = target_shape
        Ny, Nx = ds.dims["south_north"], ds.dims["west_east"]

        sy = (Ny - ny) // 2
        sx = (Nx - nx) // 2

        return ds.isel(
            south_north=slice(sy, sy + ny),
            west_east=slice(sx, sx + nx),
        )

    # Center crop and attach coordinates from grid
    ds_with_coords = crop_to_center(surface_ds, (304, 304)).assign_coords(
        XLAT=(("south_north", "west_east"), grid["XLAT"].data),
        XLONG=(("south_north", "west_east"), grid["XLONG"].data),
    )

    # Rename variables
    output_ds = ds_with_coords.rename(TAIESM_3P5_CHANNELS)

    # Based on REF grid, regrid TaiESM 3.5km data over spatial dimensions for all timestamps.
    # regridded_daily = regrid_dataset(daily_ds, grid)

    return output_ds, output_ds

def get_cwb_pressure(cwb_channel: np.ndarray) -> xr.DataArray:
    """
    Create a DataArray for TaiESM 3.5km pressure levels.

    Parameters:
        cwb_channel (array-like): Array of TaiESM 3.5km channel indices.

    Returns:
        xarray.DataArray: DataArray representing TaiESM 3.5km pressure levels.
    """
    return xr.DataArray(
        data=da.from_array(
            [np.nan] * len(TAIESM_3P5_CHANNELS),
            chunks=(len(TAIESM_3P5_CHANNELS),)
        ),
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

def get_cwb_variable(cwb_var_names: List[str], cwb_pressure: xr.DataArray) -> xr.DataArray:
    """
    Create a DataArray for TaiESM 3.5km variable names.

    Parameters:
        cwb_var_names (array-like): Array of TaiESM 3.5km variable names.
        cwb_pressure (xarray.DataArray): DataArray of TaiESM 3.5km pressure levels.

    Returns:
        xarray.DataArray: DataArray representing TaiESM 3.5km variables.
    """
    cwb_vars_dask = da.from_array(cwb_var_names, chunks=(len(TAIESM_3P5_CHANNELS),))
    return xr.DataArray(
        cwb_vars_dask,
        dims=["cwb_channel"],
        coords={"cwb_pressure": cwb_pressure},
        name="cwb_variable"
    )

def get_cwb(
        output_ds: xr.Dataset,
        cwb_var_names: List[str],
        cwb_channel: List[str],
        cwb_pressure: xr.DataArray,
        cwb_variable: xr.DataArray
    ) -> xr.DataArray:
    """
    Generate the CWB DataArray by stacking TaiESM 3.5km output variables.

    Parameters:
        output_ds (xarray.Dataset): The regridded TaiESM 3.5km dataset.
        cwb_var_names (array-like): Array of TaiESM 3.5km variable names.
        cwb_channel (array-like): Array of TaiESM 3.5km channel indices.
        cwb_pressure (xarray.DataArray): DataArray of TaiESM 3.5km pressure levels.
        cwb_variable (xarray.DataArray): DataArray of TaiESM 3.5km variables.

    Returns:
        xarray.DataArray: The processed CWB DataArray.
    """
    stack_da = da.stack([output_ds[var].data for var in cwb_var_names], axis=1)
    cwb_dims = ["time", "cwb_channel", "south_north", "west_east"]
    cwb_coords = {
        "time": output_ds["time"],
        "cwb_channel": cwb_channel,
        "south_north": output_ds["south_north"],
        "west_east": output_ds["west_east"],
        "XLAT": output_ds["XLAT"],
        "XLONG": output_ds["XLONG"],
        "cwb_pressure": cwb_pressure,
        "cwb_variable": cwb_variable,
    }
    cwb_chunk_sizes = {
        "time": 1,
        "cwb_channel": cwb_channel.size,
        "south_north": output_ds["south_north"].size,
        "west_east": output_ds["west_east"].size,
    }

    return create_and_process_dataarray("cwb", stack_da, cwb_dims, cwb_coords, cwb_chunk_sizes)

def get_cwb_center(output_ds: xr.Dataset, cwb_pressure: xr.DataArray,
                   cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the mean values of specified variables over time and spatial dimensions.

    Parameters:
        output_ds (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TaiESM 3.5km pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the mean for.

    Returns:
        xarray.DataArray: A DataArray containing the mean values of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    channel_mean_values = da.stack(
        [output_ds[var_name].mean(dim=["time", "south_north", "west_east"]).data
         for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        channel_mean_values,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

def get_cwb_scale(output_ds: xr.Dataset, cwb_pressure: xr.DataArray,
                  cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the standard deviation of specified variables over time and spatial dimensions.

    Parameters:
        output_ds (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TaiESM 3.5km pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the standard
                                         deviation for.

    Returns:
        xarray.DataArray: A DataArray containing the standard deviation of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    channel_std_values = da.stack(
        [output_ds[var_name].std(dim=["time", "south_north", "west_east"]).data
         for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        channel_std_values,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_scale"
    )

def get_cwb_valid(output_ds: xr.Dataset, cwb: xr.DataArray) -> xr.DataArray:
    """
    Generate a DataArray indicating the validity of each time step in the dataset.

    Parameters:
        output_ds (xarray.Dataset): The dataset containing the time dimension.
        cwb (xarray.DataArray): The CWB DataArray with a 'time' coordinate.

    Returns:
        xarray.DataArray: A DataArray of boolean values indicating the validity of each time step,
                          with dimension ['time'] and the same 'time' coordinate as the input
                          dataset.
    """
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [valid] * len(output_ds["time"]),
                chunks=(len(output_ds["time"]))
            ),
        dims=["time"],
        coords={"time": cwb["time"]},
        name="cwb_valid"
    )

def generate_output(
    grid: xr.Dataset,
    start_date: str,
    end_date: str,
    ssp_suffix: str = ''
) -> Tuple[
    xr.DataArray,  # cwb
    xr.DataArray,  # cwb_variable
    xr.DataArray,  # cwb_center
    xr.DataArray,  # cwb_scale
    xr.DataArray,  # cwb_valid
    xr.Dataset,    # pre_regrid
    xr.Dataset     # post_regrid
]:
    """
    Generate processed TaiESM 3.5 km outputs and corresponding CWB diagnostic
    DataArrays for a specified date range.

    Parameters
    ----------
    grid (xr.Dataset): Reference grid defining the target spatial domain for regridding.
    start_date (str): Start date in 'YYYYMMDD' format (inclusive).
    end_date (str): End date in 'YYYYMMDD' format (inclusive).
    ssp_suffix (str, optional): Scenario suffix used to select the TaiESM dataset directory
                                (e.g., 'historical', 'ssp126', 'ssp245').

    Returns
    -------
    tuple
        A tuple containing:
        - **cwb** (xr.DataArray): Final processed CWB tensor used by CorrDiff.
        - **cwb_variable** (xr.DataArray): Names of variables included in the CWB tensor.
        - **cwb_center** (xr.DataArray): Per-variable mean values (centering).
        - **cwb_scale** (xr.DataArray): Per-variable standard deviations (scaling).
        - **cwb_valid** (xr.DataArray): Boolean mask indicating valid time steps.
        - **pre_regrid** (xr.Dataset): Native TaiESM 3.5 km dataset before spatial regridding.
        - **post_regrid** (xr.Dataset): TaiESM 3.5 km dataset regridded to the target domain.

    Notes
    -----
    This function encapsulates the full processing pipeline:
    loading TaiESM data, centering/scaling, computing validity flags,
    and regridding to the specified reference grid.
    """
    # Extract TaiESM 3.5km data from file.
    output_ds, regridded_ds = get_dataset(grid, start_date, end_date, ssp_suffix)
    print(f"\n[{ssp_suffix}] TaiESM_3.5km dataset  =>\n {regridded_ds}")

    # Prepare for generation
    cwb_channel = np.arange(len(TAIESM_3P5_CHANNELS))
    cwb_pressure = get_cwb_pressure(cwb_channel)
    # Define variable names and create DataArray for cwb_variable.
    cwb_var_names = np.array(list(regridded_ds.data_vars.keys()), dtype="<U26")

    # Generate output fields
    cwb_variable = get_cwb_variable(cwb_var_names, cwb_pressure)
    cwb = get_cwb(regridded_ds, cwb_var_names, cwb_channel, cwb_pressure, cwb_variable)
    cwb_center = get_cwb_center(regridded_ds, cwb_pressure, cwb_variable)
    cwb_scale = get_cwb_scale(regridded_ds, cwb_pressure, cwb_variable)
    cwb_valid = get_cwb_valid(regridded_ds, cwb)

    return cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, output_ds, regridded_ds
