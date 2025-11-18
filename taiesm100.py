"""
ERA5 Dataset Processing Module.

This module provides utilities for processing ERA5 datasets, including retrieving,
preprocessing, regridding, and aggregating variables across multiple dimensions.
It supports generating consolidated outputs, such as means, standard deviations,
and validity masks for ERA5 data channels.

Features:
- Retrieve ERA5 pressure-level and surface data for specific date ranges.
- Regrid ERA5 datasets to align with a reference grid.
- Generate consolidated ERA5 DataArrays with stacked variables.
- Calculate aggregated metrics (e.g., mean, standard deviation) over time and spatial dimensions.
- Support for generating validity masks for ERA5 variables.

Functions:
- get_prs_paths: Generate file paths for ERA5 pressure-level data.
- get_sfc_paths: Generate file paths for ERA5 surface data.
- get_pressure_level_data: Retrieve and preprocess ERA5 pressure-level data.
- get_surface_data: Retrieve and preprocess ERA5 surface data.
- get_era5_orography: Retrieve and preprocess ERA5 orography data.
- get_tread_data: Expand and align reference grid data (from TReAD) for ERA5 processing.
- get_era5_dataset: Retrieve and preprocess ERA5 datasets, including regridding and merging.
- get_era5: Create a consolidated DataArray of ERA5 variables across channels.
- get_era5_center: Compute mean values for ERA5 variables over time and spatial dimensions.
- get_era5_scale: Compute standard deviation values for ERA5 variables over time and
                  spatial dimensions.
- get_era5_valid: Generate validity masks for ERA5 variables over time.
- generate_era5_output: Produce consolidated ERA5 outputs, including intermediate and
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
    folder = "path/to/era5/data"
    ref_grid = xr.open_dataset("path/to/ref_grid.nc")
    start_date = "20220101"
    end_date = "20220131"

    # Generate ERA5 outputs
    era5, era5_center, era5_scale, era5_valid, pre_regrid, regridded = \
        generate_era5_output(folder, ref_grid, start_date, end_date)

    print("Processed ERA5 data:", regridded)

Notes:
- Ensure that the input datasets follow the expected structure for variables and coordinates.
- The module leverages Dask for efficient computation, especially for large datasets.
- The ERA5_CHANNELS constant defines the supported ERA5 variables and their mappings.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import regrid_dataset, create_and_process_dataarray, is_local_testing

ERA5_CHANNELS = [
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
    {'name': 'w', 'pressure': 850, 'variable': 'vertical_velocity'}, # W for 850 only
    # 925
    {'name': 'z', 'pressure': 925, 'variable': 'geopotential_height'},
    {'name': 't', 'pressure': 925, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 925, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 925, 'variable': 'northward_wind'},
    # 1000
    {'name': 'q', 'pressure': 1000, 'variable': 'specific_humidity'},
    # Remaining surface channels
    {'name': 't2m', 'variable': 'temperature_2m'},
    {'name': 'u10', 'variable': 'eastward_wind_10m'},
    {'name': 'v10', 'variable': 'northward_wind_10m'},
    {'name': 'msl', 'variable': 'mean_sea_level_pressure'},
    # Orography channels
    {'name': 'oro', 'variable': 'terrain_height'}, # to replace with TER
    {'name': 'slope', 'variable': 'slope_angle'},
    {'name': 'aspect', 'variable': 'slope_aspect'},
    {'name': 'wtp', 'variable': 'weighted_precipitation'}, # tp * TER / oro
]

def get_prs_paths(
    folder: str,
    subfolder: str,
    variables: List[str],
    start_date: str,
    end_date: str
) -> List[Path]:
    """
    Generate file paths for ERA5 pressure level data files within a specified date range.

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
    Generate file paths for ERA5 surface data files within a specified date range.

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
    Retrieve and process pressure level data from ERA5 files.

    Parameters:
        folder (str): Base directory containing ERA5 pressure level data files.
        duration (slice): Time slice for the desired data range.

    Returns:
        xarray.Dataset: Processed pressure level data.
    """
    pressure_levels = sorted({ch['pressure'] for ch in ERA5_CHANNELS if 'pressure' in ch})
    pressure_level_vars = list(dict.fromkeys(
        ch['name'] for ch in ERA5_CHANNELS if 'pressure' in ch
    ))

    prs_paths = get_prs_paths(folder, 'day', pressure_level_vars, duration.start, duration.stop)
    return xr.open_mfdataset(prs_paths, combine='by_coords') \
            .sel(level=pressure_levels, time=duration)

def get_surface_data(folder: str, duration: slice) -> xr.Dataset:
    """
    Retrieve and process surface data from ERA5 files.

    Parameters:
        folder (str): Base directory containing ERA5 surface data files.
        surface_vars (list): List of variable names for surface data.
        duration (slice): Time slice for the desired data range.

    Returns:
        xarray.Dataset: Processed surface data.
    """
    surface_vars = list(dict.fromkeys(
        ch['name'] for ch in ERA5_CHANNELS
        if 'pressure' not in ch and ch['name'] not in {'oro', 'slope', 'aspect', 'wtp'}
    ))

    sfc_paths = get_sfc_paths(folder, 'day', surface_vars, duration.start, duration.stop)
    sfc_data = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    sfc_data['tp'] = sfc_data['tp'] * 24 * 1000  # Convert unit to mm/day
    sfc_data['tp'].attrs['units'] = 'mm/day'

    return sfc_data

def get_era5_orography(folder: str, time_coord: xr.DataArray) -> xr.Dataset:
    """
    Retrieve and process orography data from ERA5 files.

    Parameters:
        folder (str): Base directory containing ERA5 orography data files.
        time_coord (xarray.DataArray): Time coordinate to align with.

    Returns:
        xarray.Dataset: Processed orography data.
    """
    topo = xr.open_mfdataset(folder + '/ERA5_oro_r1440x721.nc')[['oro']]
    topo = topo.expand_dims(time=time_coord)
    return topo.reindex(time=time_coord)

def get_tread_data(terrain: xr.DataArray, time_coord: xr.DataArray) -> da.Array:
    """
    Expand a 2D terrain-related variable along the time dimension to match ERA5 data.

    This function duplicates a 2D spatial dataset (e.g., terrain height, slope, aspect)
    along the `time` dimension, ensuring alignment with ERA5 data for integration.

    Parameters:
        terrain (xarray.DataArray): A 2D DataArray representing a terrain-related variable
                                    (e.g., terrain height, slope, aspect).
                                    Shape: [south_north, west_east].
        time_coord (xarray.DataArray): A 1D DataArray containing time coordinates.
                                       The terrain data will be expanded along this dimension.

    Returns:
        dask.array.Array: A Dask-backed DataArray with an added time dimension.
                          Shape: [time, south_north, west_east].

    Raises:
        ValueError: If `terrain` is not a 2D array.

    Notes:
        - The function ensures that `terrain` remains unchanged spatially but is
          repeated along the time dimension to match `time_coord`.
        - This transformation is crucial for consistency when merging terrain variables
          with ERA5 atmospheric data.
        - The resulting DataArray is chunked (`time=1`) for efficient processing in Dask.
    """
    if terrain.ndim != 2:
        raise ValueError(f"Expected `terrain` to be 2D (south_north, west_east),"
                         f"but got shape {terrain.shape}")

    return xr.DataArray(
        terrain.expand_dims(time=time_coord.time).chunk({"time": 1}),
        dims=["time", "south_north", "west_east"],
        coords={"time": time_coord.time}
    )

def get_era5_dataset(
    folder: str,
    grid: xr.Dataset,
    layers: Dict[str, xr.DataArray],
    start_date: str,
    end_date: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve, process, and regrid ERA5 datasets for a specified date range, aligning with a
    reference grid and integrating additional topographic data.

    Parameters:
        folder (str): The base directory containing the ERA5 data files.
        grid (xarray.Dataset): The reference grid dataset for spatial alignment and cropping.
        layers (Dict[str, xarray.DataArray]): Dictionary of terrain-related layers from the
                                              reference dataset (e.g., terrain height, slope,
                                              aspect).
        start_date (str): The start date of the desired data range (YYYY-MM-DD format).
        end_date (str): The end date of the desired data range (YYYY-MM-DD format).

    Returns:
        Tuple[xarray.Dataset, xarray.Dataset]:
            - The cropped ERA5 dataset limited to the spatial domain of the reference grid.
            - The regridded ERA5 dataset aligned with the reference grid, including
              additional topographic variables (terrain height, slope, and aspect).

    Notes:
        - The function follows a structured pipeline:
            1. **Surface Data:** Retrieves ERA5 surface data (`get_surface_data`).
            2. **Pressure Level Data:** Retrieves pressure-level data (`get_pressure_level_data`).
            3. **Orography Data:** Extracts and processes ERA5 orography (`get_era5_orography`).
            4. **Cropping:** Limits the dataset to the geographic bounds of the reference grid.
            5. **Regridding:** Interpolate to match ERA5 data to the reference grid resolution.
            6. **TReAD Layer Integration:** Adds topographic data (terrain height, slope, aspect)
               from the reference dataset, applying regridding to ensure compatibility.

        - The **final dataset** (`era5_out`) includes:
            - ERA5 atmospheric and surface variables.
            - Orography (`oro`), slope, and aspect, derived from the TReAD dataset.
            - Weighted total precipitation (`wtp`), computed using terrain height adjustments.

        - The dataset is renamed to standard variable names based on `ERA5_CHANNELS`.
    """
    duration = slice(str(start_date), str(end_date))

    # Process and merge surface, pressure levels, and orography data
    era5_sfc = get_surface_data(folder, duration)
    era5 = xr.merge([
        era5_sfc,
        get_pressure_level_data(folder, duration),
        get_era5_orography(folder, era5_sfc.time)
    ])

    # Crop to Taiwan domain given ERA5 is global data.
    lat, lon = grid.XLAT, grid.XLONG
    era5_crop = era5.sel(
        latitude=slice(lat.max().item(), lat.min().item()),
        longitude=slice(lon.min().item(), lon.max().item()))

    # Based on REF grid, regrid TReAD data over spatial dimensions for all timestamps.
    era5_out = regrid_dataset(era5_crop, grid)

    # Update era5_out with TReAD layers
    tread_data = {key: get_tread_data(layer, era5_sfc.time) for key, layer in layers.items()}
    era5_out.update({
        "wtp": era5_out["tp"] * tread_data["ter"] / era5_out["oro"],
        "oro": tread_data["ter"],
        "slope": tread_data["slope"],
        "aspect": tread_data["aspect"]
    })

    era5_out = era5_out.rename({ ch['name']: ch['variable'] for ch in ERA5_CHANNELS })

    return era5_crop, era5_out

def get_era5(era5_out: xr.Dataset) -> xr.DataArray:
    """
    Constructs a consolidated ERA5 DataArray by stacking specified variables across channels.

    Parameters:
        era5_out (xarray.Dataset): The processed ERA5 dataset after regridding.

    Returns:
        xarray.DataArray: A DataArray containing the stacked ERA5 variables across defined channels,
                          with appropriate dimensions and coordinates.
    """
    era5_channel = np.arange(len(ERA5_CHANNELS))
    era5_variable = [ch.get('variable') for ch in ERA5_CHANNELS]
    era5_pressure = [ch.get('pressure', np.nan) for ch in ERA5_CHANNELS]

    # Create channel coordinates
    channel_coords = {
        "era5_variable": xr.Variable(["era5_channel"], era5_variable),
        "era5_pressure": xr.Variable(["era5_channel"], era5_pressure),
    }

    # Create ERA5 DataArray
    stack_era5 = da.stack(
        [
            era5_out[ch['variable']].sel(level=ch['pressure']).data
            if 'pressure' in ch else era5_out[ch['variable']].data
            for ch in ERA5_CHANNELS
        ],
        axis=1
    )
    era5_dims = ["time", "era5_channel", "south_north", "west_east"]
    era5_coords = {
        "time": era5_out["time"],
        "era5_channel": era5_channel,
        "south_north": era5_out["south_north"],
        "west_east": era5_out["west_east"],
        "XLAT": era5_out["XLAT"],
        "XLONG": era5_out["XLONG"],
        **channel_coords,
    }
    era5_chunk_sizes = {
        "time": 1,
        "era5_channel": era5_channel.size,
        "south_north": era5_out["south_north"].size,
        "west_east": era5_out["west_east"].size,
    }

    return create_and_process_dataarray(
        "era5", stack_era5, era5_dims, era5_coords, era5_chunk_sizes)

def get_era5_center(era5: xr.DataArray) -> xr.DataArray:
    """
    Computes the mean value for each ERA5 channel across time and spatial dimensions.

    Parameters:
        era5 (xarray.DataArray): The consolidated ERA5 DataArray with multiple channels.

    Returns:
        xarray.DataArray: A DataArray containing the mean values for each channel,
                          with 'era5_channel' as the dimension.
    """
    era5_mean = da.stack(
        [
            era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )

    return xr.DataArray(
        era5_mean,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_center"
    )

def get_era5_scale(era5: xr.DataArray) -> xr.DataArray:
    """
    Computes the standard deviation for each ERA5 channel across time and spatial dimensions.

    Parameters:
        era5 (xarray.DataArray): The consolidated ERA5 DataArray with multiple channels.

    Returns:
        xarray.DataArray: A DataArray containing the standard deviation values for each channel,
                          with 'era5_channel' as the dimension.
    """
    era5_std = da.stack(
        [
            era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    return xr.DataArray(
        era5_std,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_scale"
    )

def get_era5_valid(era5: xr.DataArray) -> xr.DataArray:
    """
    Generates a DataArray indicating the validity of each ERA5 channel over time.

    Parameters:
        era5 (xarray.DataArray): The consolidated ERA5 DataArray with multiple channels.

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

def generate_era5_output(
    folder: str,
    grid: xr.Dataset,
    terrain: xr.DataArray,
    start_date: str,
    end_date: str
) -> Tuple[
    xr.DataArray,  # ERA5 dataarray
    xr.DataArray,  # ERA5 variable
    xr.DataArray,  # ERA5 center
    xr.DataArray,  # ERA5 scale
    xr.DataArray,  # ERA5 valid
    xr.Dataset,    # ERA5 pre-regrid dataset
    xr.Dataset     # ERA5 post-regrid dataset
]:
    """
    Processes ERA5 data files to generate consolidated outputs, including the ERA5 DataArray,
    its mean (center), standard deviation (scale), validity mask, and intermediate datasets.

    Parameters:
        folder (str): The base directory containing the ERA5 data files.
        grid (xarray.Dataset): The reference grid dataset for regridding.
        terrain (xarray.DataArray): Orography (terrain height) data for the reference grid.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        tuple:
            - xarray.DataArray: The consolidated ERA5 DataArray with stacked variables.
            - xarray.DataArray: The mean values for each ERA5 channel.
            - xarray.DataArray: The standard deviation values for each ERA5 channel.
            - xarray.DataArray: The validity mask for each ERA5 channel over time.
            - xarray.Dataset: The ERA5 dataset before regridding.
            - xarray.Dataset: The ERA5 dataset after regridding.
    """
    # Extract ERA5 data from file.
    era5_pre_regrid, era5_out = get_era5_dataset(folder, grid, terrain, start_date, end_date)
    print(f"\nERA5 dataset =>\n {era5_out}")

    # Generate output fields
    era5 = get_era5(era5_out)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_out
