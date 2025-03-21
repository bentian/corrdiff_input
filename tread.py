"""
TReAD Dataset Processing Module.

This module provides utilities for processing TReAD (Taiwan ReAnalysis Dataset) data.
It includes functions to retrieve, preprocess, and regrid datasets, as well as to generate
data arrays and compute various metrics such as mean, standard deviation, and validity.

Features:
- Retrieve and process TReAD datasets for specific date ranges.
- Regrid datasets to match a specified spatial grid.
- Generate CWB-related DataArrays (e.g., pressure levels, variables, mean, and standard deviation).
- Support for calculating aggregated metrics over time and spatial dimensions.

Functions:
- get_file_paths: Generate file paths for TReAD datasets based on a date range.
- get_tread_dataset: Retrieve and preprocess TReAD datasets, including regridding.
- get_cwb_pressure: Create a DataArray for TReAD pressure levels.
- get_cwb_variable: Create a DataArray for TReAD variables.
- get_cwb: Generate a stacked CWB DataArray from TReAD output variables.
- get_cwb_center: Compute mean values for TReAD variables over time and spatial dimensions.
- get_cwb_scale: Compute standard deviation for TReAD variables over time and spatial dimensions.
- get_cwb_valid: Generate a validity mask for TReAD time steps.
- generate_tread_output: Produce processed TReAD outputs and associated metrics.

Dependencies:
- `pathlib.Path`: For file path manipulation.
- `dask.array`: For efficient handling of large datasets with lazy evaluation.
- `numpy`: For numerical operations.
- `pandas`: For handling date ranges and date-time operations.
- `xarray`: For managing multi-dimensional labeled datasets.

Usage Example:
    from tread import generate_tread_output

    # Define inputs
    file_path = "path/to/tread/data"
    ref_grid = xr.open_dataset("path/to/ref_grid.nc")
    start_date = "20220101"
    end_date = "20220131"

    # Generate TReAD outputs
    cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, pre_regrid, regridded = \
        generate_tread_output(file_path, ref_grid, start_date, end_date)

    print("Processed TReAD data:", regridded)

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

from util import regrid_dataset, create_and_process_dataarray

TREAD_CHANNELS_ORIGINAL = {
    # Baseline
    "T2": "temperature_2m",
    "U10": "eastward_wind_10m",
    "V10": "northward_wind_10m",
    # C1.x
    "UV10": "windspeed_10m",
    "RH2": "relative_humidity_2m",
    "PSFC": "sea_level_pressure",
}
TREAD_CHANNELS = {
    # Baseline
    "TP": "precipitation",
    **TREAD_CHANNELS_ORIGINAL,
    # C1.x
    "T2MAX": "maximum_temperature_2m",
    "T2MIN": "minimum_temperature_2m",
}

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
    return [folder_path / f"wrfo2D_d02_{yyyymm}.nc" for yyyymm in date_range]

def get_tread_dataset(file: str, grid: xr.Dataset,
                      start_date: str, end_date: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve and process TReAD dataset within the specified date range.

    Parameters:
        file (str): The file path or directory containing the dataset.
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.

    Returns:
        tuple: A tuple containing the original and regridded TReAD datasets.
    """
    channel_keys_original = list(TREAD_CHANNELS_ORIGINAL.keys())
    surface_vars = ['RAINC', 'RAINNC'] + channel_keys_original

    start_datetime = pd.to_datetime(str(start_date), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Read surface level data.
    tread_files = get_file_paths(file, start_date, end_date)
    tread_surface = xr.open_mfdataset(
        tread_files,
        preprocess=lambda ds: ds[surface_vars].assign_coords(
            time=pd.to_datetime(ds['Time'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')
        ).sel(time=slice(start_datetime, end_datetime))
    )

    # Calculate daily mean for original channels.
    tread = tread_surface[channel_keys_original].resample(time='1D').mean()
    # Compute additional channels:
    # - Sum TP = RAINC+RAINNC & accumulate daily, and
    # - Find T2's max and min.
    tread['TP'] = (tread_surface['RAINC'] + tread_surface['RAINNC']).resample(time='1D').sum()
    tread['T2MAX'] = (tread_surface['T2']).resample(time='1D').max()
    tread['T2MIN'] = (tread_surface['T2']).resample(time='1D').min()

    tread = tread[list(TREAD_CHANNELS.keys())].rename(TREAD_CHANNELS)

    # Based on REF grid, regrid TReAD data over spatial dimensions for all timestamps.
    tread_out = regrid_dataset(tread, grid)

    return tread, tread_out

def get_cwb_pressure(cwb_channel: np.ndarray) -> xr.DataArray:
    """
    Create a DataArray for TReAD pressure levels.

    Parameters:
        cwb_channel (array-like): Array of TReAD channel indices.

    Returns:
        xarray.DataArray: DataArray representing TReAD pressure levels.
    """
    return xr.DataArray(
        data=da.from_array(
            [np.nan] * len(TREAD_CHANNELS),
            chunks=(len(TREAD_CHANNELS),)
        ),
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

def get_cwb_variable(cwb_var_names: List[str], cwb_pressure: xr.DataArray) -> xr.DataArray:
    """
    Create a DataArray for TReAD variable names.

    Parameters:
        cwb_var_names (array-like): Array of TReAD variable names.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.

    Returns:
        xarray.DataArray: DataArray representing TReAD variables.
    """
    cwb_vars_dask = da.from_array(cwb_var_names, chunks=(len(TREAD_CHANNELS),))
    return xr.DataArray(
        cwb_vars_dask,
        dims=["cwb_channel"],
        coords={"cwb_pressure": cwb_pressure},
        name="cwb_variable"
    )

def get_cwb(
        tread_out: xr.Dataset,
        cwb_var_names: List[str],
        cwb_channel: List[str],
        cwb_pressure: xr.DataArray,
        cwb_variable: xr.DataArray
    ) -> xr.DataArray:
    """
    Generate the CWB DataArray by stacking TReAD output variables.

    Parameters:
        tread_out (xarray.Dataset): The regridded TReAD dataset.
        cwb_var_names (array-like): Array of TReAD variable names.
        cwb_channel (array-like): Array of TReAD channel indices.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of TReAD variables.

    Returns:
        xarray.DataArray: The processed CWB DataArray.
    """
    stack_tread = da.stack([tread_out[var].data for var in cwb_var_names], axis=1)
    cwb_dims = ["time", "cwb_channel", "south_north", "west_east"]
    cwb_coords = {
        "time": tread_out["time"],
        "cwb_channel": cwb_channel,
        "south_north": tread_out["south_north"],
        "west_east": tread_out["west_east"],
        "XLAT": tread_out["XLAT"],
        "XLONG": tread_out["XLONG"],
        "cwb_pressure": cwb_pressure,
        "cwb_variable": cwb_variable,
    }
    cwb_chunk_sizes = {
        "time": 1,
        "cwb_channel": cwb_channel.size,
        "south_north": tread_out["south_north"].size,
        "west_east": tread_out["west_east"].size,
    }

    return create_and_process_dataarray("cwb", stack_tread, cwb_dims, cwb_coords, cwb_chunk_sizes)

def get_cwb_center(tread_out: xr.Dataset, cwb_pressure: xr.DataArray,
                   cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the mean values of specified variables over time and spatial dimensions.

    Parameters:
        tread_out (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the mean for.

    Returns:
        xarray.DataArray: A DataArray containing the mean values of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    tread_mean = da.stack(
        [tread_out[var_name].mean(dim=["time", "south_north", "west_east"]).data
         for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        tread_mean,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_center"
    )

def get_cwb_scale(tread_out: xr.Dataset, cwb_pressure: xr.DataArray,
                  cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the standard deviation of specified variables over time and spatial dimensions.

    Parameters:
        tread_out (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the standard
                                         deviation for.

    Returns:
        xarray.DataArray: A DataArray containing the standard deviation of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    tread_std = da.stack(
        [tread_out[var_name].std(dim=["time", "south_north", "west_east"]).data
         for var_name in cwb_variable.values],
        axis=0
    )

    return xr.DataArray(
        tread_std,
        dims=["cwb_channel"],
        coords={
            "cwb_pressure": cwb_pressure,
            "cwb_variable": cwb_variable
        },
        name="cwb_scale"
    )

def get_cwb_valid(tread_out: xr.Dataset, cwb: xr.DataArray) -> xr.DataArray:
    """
    Generate a DataArray indicating the validity of each time step in the dataset.

    Parameters:
        tread_out (xarray.Dataset): The dataset containing the time dimension.
        cwb (xarray.DataArray): The CWB DataArray with a 'time' coordinate.

    Returns:
        xarray.DataArray: A DataArray of boolean values indicating the validity of each time step,
                          with dimension ['time'] and the same 'time' coordinate as the input
                          dataset.
    """
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [valid] * len(tread_out["time"]),
                chunks=(len(tread_out["time"]))
            ),
        dims=["time"],
        coords={"time": cwb["time"]},
        name="cwb_valid"
    )

def generate_tread_output(
    file: str,
    grid: xr.Dataset,
    start_date: str,
    end_date: str
) -> Tuple[
    xr.DataArray,  # TReAD dataarray
    xr.DataArray,  # TReAD variable
    xr.DataArray,  # TReAD center
    xr.DataArray,  # TReAD scale
    xr.DataArray,  # TReAD valid
    xr.Dataset,    # TReAD pre-regrid dataset
    xr.Dataset     # TReAD post-regrid dataset
]:
    """
    Generate processed TReAD output datasets and related CWB DataArrays for a specified date range.

    Parameters:
        file (str): The file path or directory containing the dataset.
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.

    Returns:
        tuple: A tuple containing the following elements:
            - cwb (xarray.DataArray): The processed CWB DataArray.
            - cwb_variable (xarray.DataArray): DataArray of TReAD variables.
            - cwb_center (xarray.DataArray): DataArray of mean values for TReAD variables.
            - cwb_scale (xarray.DataArray): DataArray of standard deviations for TReAD variables.
            - cwb_valid (xarray.DataArray): DataArray indicating the validity of each time step.
            - tread_pre_regrid (xarray.Dataset): The original TReAD dataset before regridding.
            - tread_out (xarray.Dataset): The regridded TReAD dataset.
    """
    # Extract TReAD data from file.
    tread_pre_regrid, tread_out = get_tread_dataset(file, grid, start_date, end_date)
    print(f"\nTReAD dataset =>\n {tread_out}")

    # Prepare for generation
    cwb_channel = np.arange(len(TREAD_CHANNELS))
    cwb_pressure = get_cwb_pressure(cwb_channel)
    # Define variable names and create DataArray for cwb_variable.
    cwb_var_names = np.array(list(tread_out.data_vars.keys()), dtype="<U26")

    # Generate output fields
    cwb_variable = get_cwb_variable(cwb_var_names, cwb_pressure)
    cwb = get_cwb(tread_out, cwb_var_names, cwb_channel, cwb_pressure, cwb_variable)
    cwb_center = get_cwb_center(tread_out, cwb_pressure, cwb_variable)
    cwb_scale = get_cwb_scale(tread_out, cwb_pressure, cwb_variable)
    cwb_valid = get_cwb_valid(tread_out, cwb)

    return cwb, cwb_variable, cwb_center, cwb_scale, cwb_valid, tread_pre_regrid, tread_out
