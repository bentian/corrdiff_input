"""
Utility functions for constructing CorrDiff-ready high- and low-resolution tensors.

This module centralizes the logic for turning gridded atmospheric datasets into
stacked, normalized, Dask-backed `xarray.DataArray` objects with consistent
(channel, time, south_north, west_east) conventions.

Main components
---------------
- Generic helper:
    * `create_and_process_dataarray`:
        Wraps stacked Dask arrays into `xarray.DataArray` objects with
        coordinates, daily-floored time, and chunking patterns suitable
        for CorrDiff training and inference.

- High-resolution (cwb_*) fields:
    * `get_cwb_fields`:
        High-level entry point returning:
          - `cwb`          : stacked high-resolution tensor,
          - `cwb_variable` : per-channel variable names,
          - `cwb_center`   : per-channel mean,
          - `cwb_scale`    : per-channel standard deviation,
          - `cwb_valid`    : per-time validity mask.
    * `get_cwb`, `get_cwb_pressure`, `get_cwb_variable`,
      `get_cwb_center`, `get_cwb_scale`, `get_cwb_valid`:
        Lower-level helpers used to build and describe the high-resolution tensor.

- Low-resolution (era5_*) fields:
    * `get_era5_fields`:
        High-level entry point returning:
          - `era5`        : stacked low-resolution tensor,
          - `era5_center` : per-channel mean,
          - `era5_scale`  : per-channel standard deviation,
          - `era5_valid`  : per-time validity mask.
    * `get_era5`, `get_era5_center`, `get_era5_scale`, `get_era5_valid`:
        Lower-level helpers that assemble and normalize low-resolution tensor.

All returned tensors are designed to be directly consumable by CorrDiff-style
models, with consistent metadata (variable names, pressure levels) embedded as
coordinates for easier downstream inspection and debugging.
"""
from typing import List, Dict, Tuple

import numpy as np
import xarray as xr
import dask.array as da

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
        xarray.DataArray: A DataArray containing the stacked ERA5 variables across
                          defined channels, with appropriate dimensions and coordinates.
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
