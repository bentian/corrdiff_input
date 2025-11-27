"""
Utilities for building CorrDiff training inputs from TReAD (high-resolution) and
ERA5 (low-resolution) datasets on a common reference grid.

This module provides:

- Reference grid handling:
  - `get_ref_grid()` loads the fixed WRF reference grid, returning:
    - a lightweight grid dataset with latitude/longitude fields,
    - the original grid coordinate variables (e.g. XLAT, XLONG),
    - terrain-related layers (TER, SLOPE, ASPECT).

- High-resolution (TReAD / CWB-style) processing:
  - `generate_tread_output()` reads, regrids, and packs TReAD fields into:
    - a stacked high-res DataArray (`cwb`) with time * channel * y * x,
    - companion 1D channel metadata arrays: pressure, variable name,
      per-channel mean (`cwb_center`), scale (`cwb_scale`), and validity (`cwb_valid`).

- Low-resolution (ERA5) processing:
  - `generate_era5_output()` reads, regrids, and packs ERA5 fields into:
    - a stacked low-res DataArray (`era5`) with time * channel * y * x,
    - companion 1D channel metadata arrays: pressure, variable name,
      per-channel mean (`era5_center`), scale (`era5_scale`), and validity (`era5_valid`).

The resulting high-res and low-res DataArrays are aligned on the same spatial grid
and share consistent channel metadata, making them suitable as paired inputs/targets
for CorrDiff model training and evaluation.
"""
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import xarray as xr

from tread import get_tread_dataset, get_tread_channels
from era5 import get_era5_dataset, get_era5_channels
from util import create_and_process_dataarray

# -------------------------------------------------------------------
# REF grid
# -------------------------------------------------------------------
REF_GRID_NC = "./ref_grid/wrf_208x208_grid_coords.nc"
GRID_COORD_KEYS = ["XLAT", "XLONG"]

def get_ref_grid() -> Tuple[xr.Dataset, dict, dict]:
    """
    Load the reference grid dataset and extract its coordinates and terrain-related variables.

    This function reads a predefined reference grid NetCDF file and extracts:
    - A dataset containing latitude (`lat`) and longitude (`lon`) grids.
    - A dictionary of coordinate arrays specified by `GRID_COORD_KEYS`.
    - A dictionary of terrain-related variables (`TER`, `SLOPE`, `ASPECT`) for use in
      regridding and terrain processing.

    Returns:
        tuple:
            - grid (xarray.Dataset): A dataset containing the latitude ('lat') and
              longitude ('lon') grids for spatial alignment.
            - grid_coords (dict): A dictionary of extracted coordinate arrays defined
              by `GRID_COORD_KEYS` for downstream processing.
            - terrain_layers (dict): A dictionary containing terrain-related variables
              ('ter', 'slope', 'aspect') from the reference grid.

    Notes:
        - The reference grid file path is defined by the global constant `REF_GRID_NC`.
        - The coordinate keys to extract are defined in `GRID_COORD_KEYS`.
        - The terrain-related variables are returned as a dictionary with lowercase keys
          for consistency in downstream processing.
    """
    # Reference grid paths
    ref = xr.open_dataset(REF_GRID_NC, engine='netcdf4')

    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }
    layers = { key.lower(): ref[key] for key in ["TER", "SLOPE", "ASPECT"] }

    return grid, grid_coords, layers

# -------------------------------------------------------------------
# TReAD & ERA5 outputs
# -------------------------------------------------------------------

def generate_output_dataset(start_date: str, end_date: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Generates output datasets for TReAD and ERA5 based on a specified date range
    and a common reference grid.

    This function first retrieves a reference grid, its coordinates, and layer
    information. It then uses this grid to generate two xarray.Dataset objects:
    one representing TReAD output and another for ERA5 output, both
    constrained by the given start and end dates.

    Args:
        start_date (str): The start date for generating the datasets in format "YYYYMMDD".
        end_date (str): The end date for generating the datasets in format "YYYYMMDD".

    Returns:
        Tuple[xr.Dataset, xr.Dataset, xr.Dataset]: A tuple containing three elements:
            - tread_output (xr.Dataset): An xarray Dataset containing the
                                         generated TReAD output data.
            - era5_output (xr.Dataset): An xarray Dataset containing the
                                        generated ERA5 output data.
            - grid_coords (xr.Dataset): An xarray Dataset containing the
                                        coordinates of the reference grid.
    """
    grid, grid_coords, layers = get_ref_grid()
    return (
        generate_tread_output(grid, start_date, end_date),
        generate_era5_output(grid, layers, start_date, end_date),
        grid_coords
    )

def generate_tread_output(
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
    tread_pre_regrid, tread_out = get_tread_dataset(grid, start_date, end_date)
    print(f"\nTReAD dataset =>\n {tread_out}")

    # Generate cwb_* fields
    cwb_fields = get_cwb_fields(tread_out, get_tread_channels())
    return (*cwb_fields, tread_pre_regrid, tread_out)

def generate_era5_output(
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
        grid (xarray.Dataset): The reference grid dataset for regridding.
        terrain (xarray.DataArray): Orography (terrain height) data for the reference grid.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        tuple:
            - era5 (xarray.DataArray): The consolidated ERA5 DataArray with stacked variables.
            - era5_center (xarray.DataArray): The mean values for each ERA5 channel.
            - era5_scale (xarray.DataArray): The standard deviation values for each ERA5 channel.
            - era5_valid (xarray.DataArray): The validity mask for each ERA5 channel over time.
            - era5_pre_regrid (xarray.Dataset): The ERA5 dataset before regridding.
            - era5_out(xarray.Dataset): The ERA5 dataset after regridding.
    """
    # Extract ERA5 data from file.
    era5_pre_regrid, era5_out = get_era5_dataset(grid, terrain, start_date, end_date)
    print(f"\nERA5 dataset =>\n {era5_out}")

    # Generate era5_* fields
    era5_fields = get_era5_fields(era5_out, get_era5_channels())
    return (*era5_fields, era5_pre_regrid, era5_out)

# -------------------------------------------------------------------
# High-res fields (cwb_*)
# -------------------------------------------------------------------

def get_cwb_fields(
    highres_ds: xr.Dataset,
    channels: Dict[str, str]
) -> Tuple[
    xr.DataArray,  # cwb
    xr.DataArray,  # cwb_variable
    xr.DataArray,  # cwb_center
    xr.DataArray,  # cwb_scale
    xr.DataArray,  # cwb_valid
]:
    """
    Compute all CWB diagnostic fields (stacked tensor, variable metadata, mean,
    scale, and validity mask) from a high-resolution dataset.

    This function assembles the full set of CorrDiff-ready high-resolution fields:
    - The stacked CWB tensor (`cwb`)
    - Per-channel variable names (`cwb_variable`)
    - Per-channel mean offsets (`cwb_center`)
    - Per-channel standard deviations (`cwb_scale`)
    - A time-validity mask (`cwb_valid`)

    Parameters
    ----------
    highres_ds : xr.Dataset
        The high-resolution dataset after spatial regridding, where each data
        variable represents one physical channel (e.g., temperature, wind).
    channels : Dict[str, str]
        Mapping of channel names to their standardized output names. The number
        of channels determines the length of the CWB tensors.

    Returns
    -------
    tuple
        A tuple of five xarray.DataArray objects:
        - **cwb** : Stacked tensor containing all high-resolution variables
                    with dimensions (time, cwb_channel, south_north, west_east).
        - **cwb_variable** : Variable names associated with each stacked channel.
        - **cwb_center** : Mean value per channel over time and space.
        - **cwb_scale** : Standard deviation per channel.
        - **cwb_valid** : Boolean validity mask for each time step.

    Notes
    -----
    - The function assumes `highres_ds` variables are already renamed according
      to `channels`.
    - Channel ordering follows the order of keys in `channels`.
    - This function acts as a thin wrapper assembling all intermediate steps
      (variable extraction, stacking, centering, scaling, validity).
    """
    # Prepare for generation
    cwb_channel = np.arange(len(channels))
    cwb_pressure = get_cwb_pressure(cwb_channel, channels)
    # Define variable names and create DataArray for cwb_variable.
    cwb_var_names = np.array(list(highres_ds.data_vars.keys()), dtype="<U26")

    # Generate output fields
    cwb_variable = get_cwb_variable(cwb_var_names, cwb_pressure, channels)
    cwb = get_cwb(highres_ds, cwb_var_names, cwb_channel, cwb_pressure, cwb_variable)
    return (
        cwb, cwb_variable,
        get_cwb_center(highres_ds, cwb_pressure, cwb_variable),
        get_cwb_scale(highres_ds, cwb_pressure, cwb_variable),
        get_cwb_valid(highres_ds, cwb)
    )

def get_cwb_pressure(cwb_channel: np.ndarray, channels: Dict[str, str]) -> xr.DataArray:
    """
    Create a DataArray for TReAD pressure levels.

    Parameters:
        cwb_channel (array-like): Array of TReAD channel indices.
        channels (dict): Mapping of variable names used to determine the number of channels.

    Returns:
        xarray.DataArray: DataArray representing TReAD pressure levels.
    """
    return xr.DataArray(
        data=da.from_array(
            [np.nan] * len(channels),
            chunks=(len(channels),)
        ),
        dims=["cwb_channel"],
        coords={"cwb_channel": cwb_channel},
        name="cwb_pressure"
    )

def get_cwb_variable(cwb_var_names: List[str], cwb_pressure: xr.DataArray,
                     channels: Dict[str, str]) -> xr.DataArray:
    """
    Create a DataArray for TReAD variable names.

    Parameters:
        cwb_var_names (array-like): Array of TReAD variable names.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.

    Returns:
        xarray.DataArray: DataArray representing TReAD variables.
    """
    cwb_vars_dask = da.from_array(cwb_var_names, chunks=(len(channels),))
    return xr.DataArray(
        cwb_vars_dask,
        dims=["cwb_channel"],
        coords={"cwb_pressure": cwb_pressure},
        name="cwb_variable"
    )

def get_cwb(
        highres_ds: xr.Dataset,
        cwb_var_names: List[str],
        cwb_channel: List[str],
        cwb_pressure: xr.DataArray,
        cwb_variable: xr.DataArray
    ) -> xr.DataArray:
    """
    Generate the CWB DataArray by stacking TReAD output variables.

    Parameters:
        highres_ds (xarray.Dataset): The regridded TReAD dataset.
        cwb_var_names (array-like): Array of TReAD variable names.
        cwb_channel (array-like): Array of TReAD channel indices.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of TReAD variables.

    Returns:
        xarray.DataArray: The processed CWB DataArray.
    """
    stack_da = da.stack([highres_ds[var].data for var in cwb_var_names], axis=1)
    cwb_dims = ["time", "cwb_channel", "south_north", "west_east"]
    cwb_coords = {
        "time": highres_ds["time"],
        "cwb_channel": cwb_channel,
        "south_north": highres_ds["south_north"],
        "west_east": highres_ds["west_east"],
        "XLAT": highres_ds["XLAT"],
        "XLONG": highres_ds["XLONG"],
        "cwb_pressure": cwb_pressure,
        "cwb_variable": cwb_variable,
    }
    cwb_chunk_sizes = {
        "time": 1,
        "cwb_channel": cwb_channel.size,
        "south_north": highres_ds["south_north"].size,
        "west_east": highres_ds["west_east"].size,
    }

    return create_and_process_dataarray("cwb", stack_da, cwb_dims, cwb_coords, cwb_chunk_sizes)

def get_cwb_center(highres_ds: xr.Dataset, cwb_pressure: xr.DataArray,
                   cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the mean values of specified variables over time and spatial dimensions.

    Parameters:
        highres_ds (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the mean for.

    Returns:
        xarray.DataArray: A DataArray containing the mean values of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    channel_mean_values = da.stack(
        [highres_ds[var_name].mean(dim=["time", "south_north", "west_east"]).data
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

def get_cwb_scale(highres_ds: xr.Dataset, cwb_pressure: xr.DataArray,
                  cwb_variable: xr.DataArray) -> xr.DataArray:
    """
    Calculate the standard deviation of specified variables over time and spatial dimensions.

    Parameters:
        highres_ds (xarray.Dataset): The dataset containing the variables.
        cwb_pressure (xarray.DataArray): DataArray of TReAD pressure levels.
        cwb_variable (xarray.DataArray): DataArray of variable names to calculate the standard
                                         deviation for.

    Returns:
        xarray.DataArray: A DataArray containing the standard deviation of the specified variables,
                          with dimensions ['cwb_channel'] and coordinates for 'cwb_pressure'
                          and 'cwb_variable'.
    """
    channel_std_values = da.stack(
        [highres_ds[var_name].std(dim=["time", "south_north", "west_east"]).data
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

def get_cwb_valid(highres_ds: xr.Dataset, cwb: xr.DataArray) -> xr.DataArray:
    """
    Generate a DataArray indicating the validity of each time step in the dataset.

    Parameters:
        highres_ds (xarray.Dataset): The dataset containing the time dimension.
        cwb (xarray.DataArray): The CWB DataArray with a 'time' coordinate.

    Returns:
        xarray.DataArray: A DataArray of boolean values indicating the validity of each time step,
                          with dimension ['time'] and the same 'time' coordinate as the input
                          dataset.
    """
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [valid] * len(highres_ds["time"]),
                chunks=(len(highres_ds["time"]))
            ),
        dims=["time"],
        coords={"time": cwb["time"]},
        name="cwb_valid"
    )

# -------------------------------------------------------------------
# Low-res fields (era5_*)
# -------------------------------------------------------------------

def get_era5_fields(
    lowres_ds: xr.Dataset,
    channels: dict
) -> Tuple[
    xr.DataArray,  # era5
    xr.DataArray,  # era5_center
    xr.DataArray,  # era5_scale
    xr.DataArray,  # era5_valid
]:
    """
    Compute all ERA5 diagnostic fields (stacked tensor, mean, scale, and
    validity mask) from a low-resolution dataset.

    This function assembles the full set of CorrDiff-ready ERA5 fields:
    - The stacked ERA5 tensor (`era5`)
    - Per-channel mean offsets (`era5_center`)
    - Per-channel standard deviations (`era5_scale`)
    - A time–channel validity mask (`era5_valid`)

    Parameters
    ----------
    lowres_ds : xr.Dataset
        The low resolution dataset after spatial regridding, containing the atmospheric,
        pressure-level, and surface variables needed for model conditioning.
    channels : dict
        List or mapping defining the ERA5 channels, including variable names
        and optional pressure levels. The length determines the output channel
        dimension.

    Returns
    -------
    tuple
        A tuple containing four xarray.DataArray objects:
        - **era5** : Stacked ERA5 tensor with dimensions
                     (time, era5_channel, south_north, west_east).
        - **era5_center** : Mean value per ERA5 channel.
        - **era5_scale** : Standard deviation per ERA5 channel.
        - **era5_valid** : Boolean validity mask for each time / channel entry.

    Notes
    -----
    - The function assumes variables in `lowres_ds` match the specifications
      given in `channels`.
    - Channel ordering follows the order in the `channels` list.
    - This helper function centralizes the creation of ERA5 model inputs,
      ensuring consistent normalization and metadata handling.
    """
    era5 = get_era5(lowres_ds, channels)
    return (
        era5,
        get_era5_center(era5),
        get_era5_scale(era5),
        get_era5_valid(era5)
    )

def get_era5(lowres_ds: xr.Dataset, channels: dict) -> xr.DataArray:
    """
    Constructs a consolidated ERA5 DataArray by stacking specified variables across channels.

    Parameters:
        lowres_ds (xarray.Dataset): The processed ERA5 dataset after regridding.

    Returns:
        xarray.DataArray: A DataArray containing the stacked ERA5 variables across defined channels,
                          with appropriate dimensions and coordinates.
    """
    era5_channel = np.arange(len(channels))
    era5_variable = [ch.get('variable') for ch in channels]
    era5_pressure = [ch.get('pressure', np.nan) for ch in channels]

    # Create channel coordinates
    channel_coords = {
        "era5_variable": xr.Variable(["era5_channel"], era5_variable),
        "era5_pressure": xr.Variable(["era5_channel"], era5_pressure),
    }

    # Create ERA5 DataArray
    stack_era5 = da.stack(
        [
            lowres_ds[ch['variable']].sel(level=ch['pressure']).data
            if 'pressure' in ch else lowres_ds[ch['variable']].data
            for ch in channels
        ],
        axis=1
    )
    era5_dims = ["time", "era5_channel", "south_north", "west_east"]
    era5_coords = {
        "time": lowres_ds["time"],
        "era5_channel": era5_channel,
        "south_north": lowres_ds["south_north"],
        "west_east": lowres_ds["west_east"],
        "XLAT": lowres_ds["XLAT"],
        "XLONG": lowres_ds["XLONG"],
        **channel_coords,
    }
    era5_chunk_sizes = {
        "time": 1,
        "era5_channel": era5_channel.size,
        "south_north": lowres_ds["south_north"].size,
        "west_east": lowres_ds["west_east"].size,
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
