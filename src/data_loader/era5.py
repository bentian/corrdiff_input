"""
ERA5 reanalysis loader and preprocessor for CorrDiff low-resolution inputs.

This module is responsible for:
- Locating ERA5 pressure-level and surface files based on the execution
  environment (local testing vs BIG server).
- Building file paths for:
    * Pressure levels (PRS)
    * Surface fields (SFC)
    * Orography (static topography)
- Loading multi-month ERA5 data with `xarray.open_mfdataset` and sub-selecting
  the requested time range.
- Splitting the workflow into:
    * Pressure-level fields (z, t, u, v, w on 500/700/850/925/1000 hPa)
    * Surface fields (tp, t2m, u10, v10, msl, etc.)
    * Orography / terrain data
- Cropping the global ERA5 domain to the Taiwan / reference grid extent.
- Regridding ERA5 to a given WRF-style reference grid using `regrid_dataset`.
- Expanding high-resolution terrain layers (TER, slope, aspect) in time and
  injecting them into the ERA5 grid, including:
    * `oro` / terrain height
    * `slope` / `aspect`
    * `wtp` (weighted precipitation = tp * TER / oro)

Key public helpers:
    - `get_era5_channels()`:
        Returns the channel specification (name, pressure, variable) used to
        map ERA5 fields to CorrDiff channels.
    - `get_data_dir()`:
        Resolves the root ERA5 directory depending on environment.
    - `get_era5_dataset(grid, layers, start_date, end_date)`:
        High-level entry point that loads, crops, regrids, and augments ERA5
        into:
            * a cropped native-resolution dataset, and
            * a regridded, terrain-enhanced dataset ready for CorrDiff.

All lower-level helpers (`get_prs_paths`, `get_sfc_paths`, `get_surface_data`,
`get_pressure_level_data`, `get_era5_orography`, `get_tread_data`) are
factored out to keep the main pipeline clear and reusable.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import dask.array as da
import pandas as pd
import xarray as xr

from .util import regrid_dataset, is_local_testing

# Surface + pressure level channels that are common
BASELINE_CHANNELS: List[Dict[str, dict]] = [
    {"name": "tp", "variable": "precipitation"},
    *[
        {"name": name, "pressure": pressure, "variable": variable}
        for pressure in (500, 700, 850, 925)
        for name, variable in (
            ("z", "geopotential_height"),
            ("t", "temperature"),
            ("u", "eastward_wind"),
            ("v", "northward_wind"),
        )
    ],
    {'name': 't2m', 'variable': 'temperature_2m'},
    {'name': 'u10', 'variable': 'eastward_wind_10m'},
    {'name': 'v10', 'variable': 'northward_wind_10m'},
]

ERA5_CHANNELS = [
    *BASELINE_CHANNELS,
    # Orography channels
    {'name': 'oro', 'variable': 'terrain_height'}, # to replace with TER
    {'name': 'slope', 'variable': 'slope_angle'},
    {'name': 'aspect', 'variable': 'slope_aspect'},
    {'name': 'wtp', 'variable': 'weighted_precipitation'}, # tp * TER / oro
]

def get_era5_channels() -> dict:
    """Returns ERA5 channel list."""
    return ERA5_CHANNELS

def get_data_dir() -> str:
    """
    Return the base directory for ERA5 data based on the execution environment.

    Returns
    -------
    str
        Path to the ERA5 dataset:
        - `./data/era5` when running in a local testing environment
        - `/lfs/archive/Reanalysis/ERA5` when running on the BIG server

    Notes
    -----
    This function centralizes environment-dependent path selection so other
    parts of the codebase can access ERA5 data without needing to know
    where it is stored.
    """
    return "../data/era5" if is_local_testing() else "/lfs/archive/Reanalysis/ERA5"

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
    return xr.open_mfdataset(prs_paths, combine='by_coords', compat="no_conflicts") \
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
    sfc_data = xr.open_mfdataset(sfc_paths, combine='by_coords', compat="no_conflicts") \
                .sel(time=duration)
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
    grid: xr.Dataset,
    layers: Dict[str, xr.DataArray],
    start_date: str,
    end_date: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve, process, and regrid ERA5 datasets for a specified date range, aligning with a
    reference grid and integrating additional topographic data.

    Parameters:
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
    folder = get_data_dir()

    # Process and merge surface, pressure levels, and orography data
    era5_sfc = get_surface_data(folder, duration)
    era5_parts = [
        era5_sfc,
        get_pressure_level_data(folder, duration),
        get_era5_orography(folder, era5_sfc.time),
    ]
    era5 = xr.merge(era5_parts, compat="no_conflicts").drop_vars("time_bnds", errors="ignore")

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
