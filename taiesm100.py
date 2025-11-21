"""
TaiESM 100km Dataset Processing Module.

This module provides utilities for processing TaiESM 100km datasets, including retrieving,
preprocessing, regridding, and aggregating variables across multiple dimensions.
It supports generating consolidated outputs, such as means, standard deviations,
and validity masks for TaiESM 100km data channels.

Features:
- Retrieve TaiESM 100km pressure-level and surface data for specific date ranges.
- Regrid TaiESM 100km datasets to align with a reference grid.
- Generate consolidated TaiESM 100km DataArrays with stacked variables.
- Calculate aggregated metrics (e.g., mean, standard deviation) over time and spatial dimensions.
- Support for generating validity masks for TaiESM 100km variables.

Functions:
- get_prs_paths: Generate file paths for TaiESM 100km pressure-level data.
- get_sfc_paths: Generate file paths for TaiESM 100km surface data.
- get_pressure_level_data: Retrieve and preprocess TaiESM 100km pressure-level data.
- get_surface_data: Retrieve and preprocess TaiESM 100km surface data.
- get_era5_dataset: Retrieve and preprocess TaiESM 100km datasets,
                    including regridding and merging.
- get_era5: Create a consolidated DataArray of TaiESM 100km variables across channels.
- get_era5_center: Compute mean values for TaiESM 100km variables over time and spatial dimensions.
- get_era5_scale: Compute standard deviation values for TaiESM 100km variables over time and
                  spatial dimensions.
- get_era5_valid: Generate validity masks for TaiESM 100km variables over time.
- generate_era5_output: Produce consolidated TaiESM 100km outputs, including intermediate and
                        aggregated datasets.

Dependencies:
- `pathlib.Path`: For file path manipulation.
- `dask.array`: For efficient handling of large datasets with lazy evaluation.
- `numpy`: For numerical operations.
- `pandas`: For handling date ranges and date-time operations.
- `xarray`: For managing multi-dimensional labeled datasets.
- `util`: Module for regridding and processing DataArrays.

Usage Example:
    from era5 import generate_era5_output

    # Define inputs
    ref_grid = xr.open_dataset("path/to/ref_grid.nc")
    start_date = "20220101"
    end_date = "20220131"

    # Generate TaiESM 100km outputs
    era5, era5_center, era5_scale, era5_valid, pre_regrid, regridded = \
        generate_era5_output(ref_grid, start_date, end_date)

    print("Processed TaiESM 100km data:", regridded)

Notes:
- Ensure that the input datasets follow the expected structure for variables and coordinates.
- The module leverages Dask for efficient computation, especially for large datasets.
- The TAIESM_100_CHANNELS constant defines the supported TaiESM 100km variables and their mappings.
"""
from pathlib import Path
from typing import List, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import regrid_dataset, create_and_process_dataarray, is_local_testing

TAIESM_100_CHANNELS = [
    {'name': 'tp', 'variable': 'precipitation'},
    # 500
    {'name': 'z', 'pressure': 500, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 500, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 500, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 500, 'variable': 'northward_wind'},
    # 700
    {'name': 'z', 'pressure': 700, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 700, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 700, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 700, 'variable': 'northward_wind'},
    # 850
    {'name': 'z', 'pressure': 850, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 850, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 850, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 850, 'variable': 'northward_wind'},
    # 925
    {'name': 'z', 'pressure': 925, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 925, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 925, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 925, 'variable': 'northward_wind'},
    # Remaining surface channels
    {'name': 't2m', 'variable': 'temperature_2m'},
    {'name': 'u10', 'variable': 'eastward_wind_10m'},
    {'name': 'v10', 'variable': 'northward_wind_10m'},
]

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
        - `./data/taiesm100` when running locally (as detected by is_local_testing())
        - `/lfs/home/corrdiff/data/012-predictor_TaiESM1_ssp/{ssp_suffix}_daily/` when
          running on the BIG server.

    Notes
    -----
    This helper centralizes environment-aware path logic so other code does not
    need to handle local vs. remote directory differences.
    """
    return "./data/taiesm3p5" if is_local_testing() else \
            f"/lfs/home/corrdiff/data/012-predictor_TaiESM1_ssp/{ssp_suffix}_daily/"

def get_prs_paths(
    folder: str,
    subfolder: str,
    variables: List[str],
    start_date: str,
    end_date: str
) -> List[Path]:
    """
    Generate file paths for TaiESM 100km pressure level data files within a specified date range.

    Parameters:
        folder (str): The base directory containing the data files.
        subfolder (str): The subdirectory under 'PRS' where the data files are located.
        variables (list of str): List of variable names to include.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        list: A list of file paths corresponding to the specified variables and date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    if is_local_testing():
        return [folder_path / f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc"
                for var in variables for yyyymm in date_range]

    return [
        folder_path / "PRS" / subfolder / var / yyyymm[:4] / \
            f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc"
        for var in variables for yyyymm in date_range
    ]

def get_sfc_paths(
    folder: str,
    subfolder: str,
    variables: List[str],
    start_date: str,
    end_date: str
) -> List[Path]:
    """
    Generate file paths for TaiESM 100km surface data files within a specified date range.

    Parameters:
        folder (str): The base directory containing the data files.
        subfolder (str): The subdirectory under 'SFC' where the data files are located.
        variables (list of str): List of variable names to include.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        list: A list of file paths corresponding to the specified variables and date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    if is_local_testing():
        return [folder_path / f"ERA5_SFC_{var}_201801_r1440x721_day.nc" for var in variables]

    return [
        folder_path / "SFC" / subfolder / var / yyyymm[:4] / \
            f"ERA5_SFC_{var}_{yyyymm}_r1440x721_day.nc"
        for var in variables for yyyymm in date_range
    ]

def get_pressure_level_data(folder: str, duration: slice) -> xr.Dataset:
    """
    Retrieve and process pressure level data from TaiESM 100km files.

    Parameters:
        folder (str): Base directory containing TaiESM 100km pressure level data files.
        duration (slice): Time slice for the desired data range.

    Returns:
        xarray.Dataset: Processed pressure level data.
    """
    pressure_levels = sorted({ch['pressure'] for ch in TAIESM_100_CHANNELS if 'pressure' in ch})
    pressure_level_vars = list(dict.fromkeys(
        ch['name'] for ch in TAIESM_100_CHANNELS if 'pressure' in ch
    ))

    prs_paths = get_prs_paths(folder, 'day', pressure_level_vars, duration.start, duration.stop)
    return xr.open_mfdataset(prs_paths, combine='by_coords') \
            .sel(level=pressure_levels, time=duration)

def get_surface_data(folder: str, duration: slice) -> xr.Dataset:
    """
    Retrieve and process surface data from TaiESM 100km files.

    Parameters:
        folder (str): Base directory containing TaiESM 100km surface data files.
        surface_vars (list): List of variable names for surface data.
        duration (slice): Time slice for the desired data range.

    Returns:
        xarray.Dataset: Processed surface data.
    """
    surface_vars = list(dict.fromkeys(
        ch['name'] for ch in TAIESM_100_CHANNELS
        if 'pressure' not in ch
    ))

    sfc_paths = get_sfc_paths(folder, 'day', surface_vars, duration.start, duration.stop)
    sfc_data = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    sfc_data['tp'] = sfc_data['tp'] * 24 * 1000  # Convert unit to mm/day
    sfc_data['tp'].attrs['units'] = 'mm/day'

    return sfc_data

def get_dataset(grid: xr.Dataset, start_date: str, end_date: str,
                ssp_suffix: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve, process, and regrid TaiESM 100km datasets for a specified date range, aligning with a
    reference grid.

    Parameters:
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.
        ssp_suffix (str): Scenario suffix used to select the TaiESM dataset directory.

    Returns:
        Tuple[xarray.Dataset, xarray.Dataset]:
            - The cropped TaiESM 100km dataset limited to the spatial domain of the reference grid.
            - The regridded TaiESM 100km dataset aligned with the reference grid, including
              additional topographic variables (terrain height, slope, and aspect).

    Notes:
        - The function follows a structured pipeline:
            1. **Surface Data:** Retrieves TaiESM 100km surface data (`get_surface_data`).
            2. **Pressure Level Data:** Retrieves pressure-level data (`get_pressure_level_data`).
            4. **Cropping:** Limits the dataset to the geographic bounds of the reference grid.
            5. **Regridding:** Interpolate to match TaiESM 100km data to the reference grid resolution.

        - The **final dataset** (`era5_out`) includes:
            - TaiESM 100km atmospheric and surface variables.

        - The dataset is renamed to standard variable names based on `TAIESM_100_CHANNELS`.
    """
    duration = slice(str(start_date), str(end_date))
    folder = get_data_dir(ssp_suffix)

    # Process and merge surface and pressure levels data
    lr_data = xr.merge([
        get_surface_data(folder, duration),
        get_pressure_level_data(folder, duration)
    ])

    # Crop to Taiwan domain given TaiESM 100km is global data.
    lat, lon = grid.XLAT, grid.XLONG
    lr_data_cropped = lr_data.sel(
        latitude=slice(lat.max().item(), lat.min().item()),
        longitude=slice(lon.min().item(), lon.max().item()))

    # Based on REF grid, regrid data over spatial dimensions for all timestamps.
    lr_out = regrid_dataset(lr_data_cropped, grid)

    lr_out = lr_out.rename({ ch['name']: ch['variable'] for ch in TAIESM_100_CHANNELS })

    return lr_data_cropped, lr_out

def get_era5(lr_out: xr.Dataset) -> xr.DataArray:
    """
    Constructs a consolidated TaiESM 100km DataArray by stacking specified variables across channels.

    Parameters:
        lr_out (xarray.Dataset): The processed TaiESM 100km dataset after regridding.

    Returns:
        xarray.DataArray: A DataArray containing the stacked TaiESM 100km variables across defined channels,
                          with appropriate dimensions and coordinates.
    """
    lr_channel = np.arange(len(TAIESM_100_CHANNELS))
    lr_variable = [ch.get('variable') for ch in TAIESM_100_CHANNELS]
    lr_pressure = [ch.get('pressure', np.nan) for ch in TAIESM_100_CHANNELS]

    # Create channel coordinates
    channel_coords = {
        "era5_variable": xr.Variable(["era5_channel"], lr_variable),
        "era5_pressure": xr.Variable(["era5_channel"], lr_pressure),
    }

    # Create TaiESM 100km DataArray
    stack_era5 = da.stack(
        [
            lr_out[ch['variable']].sel(level=ch['pressure']).data
            if 'pressure' in ch else lr_out[ch['variable']].data
            for ch in TAIESM_100_CHANNELS
        ],
        axis=1
    )
    era5_dims = ["time", "era5_channel", "south_north", "west_east"]
    era5_coords = {
        "time": lr_out["time"],
        "era5_channel": lr_channel,
        "south_north": lr_out["south_north"],
        "west_east": lr_out["west_east"],
        "XLAT": lr_out["XLAT"],
        "XLONG": lr_out["XLONG"],
        **channel_coords,
    }
    era5_chunk_sizes = {
        "time": 1,
        "era5_channel": lr_channel.size,
        "south_north": lr_out["south_north"].size,
        "west_east": lr_out["west_east"].size,
    }

    return create_and_process_dataarray(
        "era5", stack_era5, era5_dims, era5_coords, era5_chunk_sizes)

def get_era5_center(era5: xr.DataArray) -> xr.DataArray:
    """
    Computes the mean value for each TaiESM 100km channel across time and spatial dimensions.

    Parameters:
        era5 (xarray.DataArray): The consolidated TaiESM 100km DataArray with multiple channels.

    Returns:
        xarray.DataArray: A DataArray containing the mean values for each channel,
                          with 'era5_channel' as the dimension.
    """
    channel_mean_values = da.stack(
        [
            era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )

    return xr.DataArray(
        channel_mean_values,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_center"
    )

def get_era5_scale(era5: xr.DataArray) -> xr.DataArray:
    """
    Computes the standard deviation for each TaiESM 100km channel across time and spatial dimensions.

    Parameters:
        era5 (xarray.DataArray): The consolidated TaiESM 100km DataArray with multiple channels.

    Returns:
        xarray.DataArray: A DataArray containing the standard deviation values for each channel,
                          with 'era5_channel' as the dimension.
    """
    channel_std_values = da.stack(
        [
            era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    return xr.DataArray(
        channel_std_values,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_scale"
    )

def get_era5_valid(era5: xr.DataArray) -> xr.DataArray:
    """
    Generates a DataArray indicating the validity of each TaiESM 100km channel over time.

    Parameters:
        era5 (xarray.DataArray): The consolidated TaiESM 100km DataArray with multiple channels.

    Returns:
        xarray.DataArray: A boolean DataArray with dimensions 'time' and 'era5_channel',
                          indicating the validity (True) for each channel at each time step.
    """
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [[valid] * len(era5["era5_channel"])] * len(era5["time"]),
                chunks=(len(era5["time"]), len(era5["era5_channel"]))
            ),
        dims=["time", "era5_channel"],
        coords={
            "time": era5["time"],
            "era5_channel": era5["era5_channel"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_valid"
    )

def generate_output(
    grid: xr.Dataset,
    start_date: str,
    end_date: str,
    ssp_suffix: str = ''
) -> Tuple[
    xr.DataArray,  # TaiESM 100km dataarray
    xr.DataArray,  # TaiESM 100km variable
    xr.DataArray,  # TaiESM 100km center
    xr.DataArray,  # TaiESM 100km scale
    xr.DataArray,  # TaiESM 100km valid
    xr.Dataset,    # TaiESM 100km pre-regrid dataset
    xr.Dataset     # TaiESM 100km post-regrid dataset
]:
    """
    Processes TaiESM 100km data files to generate consolidated outputs, including the TaiESM 100km DataArray,
    its mean (center), standard deviation (scale), validity mask, and intermediate datasets.

    Parameters:
        grid (xr.Dataset): Reference grid defining the target spatial domain for regridding.
        start_date (str): Start date in 'YYYYMMDD' format (inclusive).
        end_date (str): End date in 'YYYYMMDD' format (inclusive).
        ssp_suffix (str, optional): Scenario suffix used to select the TaiESM dataset directory
                                    (e.g., 'historical', 'ssp126', 'ssp245').

    Returns:
        tuple:
            - xarray.DataArray: The consolidated TaiESM 100km DataArray with stacked variables.
            - xarray.DataArray: The mean values for each TaiESM 100km channel.
            - xarray.DataArray: The standard deviation values for each TaiESM 100km channel.
            - xarray.DataArray: The validity mask for each TaiESM 100km channel over time.
            - xarray.Dataset: The TaiESM 100km dataset before regridding.
            - xarray.Dataset: The TaiESM 100km dataset after regridding.
    """
    # Extract TaiESM 100km data from file.
    output_ds, regridded_ds = get_dataset(grid, start_date, end_date, ssp_suffix)
    print(f"\n[{ssp_suffix}] TaiESM_100km dataset =>\n {regridded_ds}")

    # Generate output fields
    era5 = get_era5(regridded_ds)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid, output_ds, regridded_ds
