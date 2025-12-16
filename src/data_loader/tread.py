"""
TReAD (Taiwan ReAnalysis Downscaling) surface data loader and preprocessor.

This module handles:
- Locating TReAD monthly surface files based on execution environment
  (local testing vs. BIG server).
- Loading multi-month WRF-formatted TReAD data using `xarray.open_mfdataset`.
- Converting WRF "Times" strings into a proper `time` coordinate.
- Computing daily aggregates needed by CorrDiff:
    * Daily mean of baseline variables (T2, U10, V10, RH2, PSFC).
    * Daily precipitation accumulation (TP = RAINC + RAINNC).
    * Daily max/min temperature (T2MAX, T2MIN).
- Mapping native variable names to CorrDiff channel names.
- Regridding the processed dataset to a provided reference grid.

The main public function is:
    - `get_tread_dataset(grid, start_date, end_date)`

which returns:
    * the pre-regrid daily dataset, and
    * the same dataset regridded to match the target grid.

Constants:
    - `TREAD_CHANNELS_ORIGINAL` — mapping of native WRF variables to CorrDiff names.
    - `TREAD_CHANNELS` — full CorrDiff channel mapping including derived fields.
    - `get_tread_channels()` — returns the channel specification used by CorrDiff.

This module forms the “high-resolution” branch of the CorrDiff input pipeline
for CWA mode.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xarray as xr

from .util import is_local_testing, regrid_dataset

TREAD_CHANNELS_ORIGINAL = {
    # Baseline
    "T2": "temperature_2m",
    "U10": "eastward_wind_10m",
    "V10": "northward_wind_10m",
    # C1.x
    # "UV10": "windspeed_10m",
    # "RH2": "relative_humidity_2m",
    # "PSFC": "sea_level_pressure",
}
TREAD_CHANNELS = {
    # Baseline
    "TP": "precipitation",
    **TREAD_CHANNELS_ORIGINAL,
    # C1.x
    # "T2MAX": "maximum_temperature_2m",
    # "T2MIN": "minimum_temperature_2m",
}

def get_tread_channels() -> Dict[str, str]:
    """Returns TReAD channel list."""
    return TREAD_CHANNELS

def get_data_dir() -> str:
    """
    Return the base directory for TReAD surface data based on the execution environment.

    Returns
    -------
    str
        Path to the TReAD data directory:
        - `./data/tread` when running in a local testing environment
        - `/lfs/archive/TCCIP_data/TReAD/SFC/hr` when running on the BIG server

    Notes
    -----
    This helper centralizes environment-dependent path handling so the rest of the
    codebase can reference TReAD data without worrying about where it is stored.
    """
    return "../data/tread" if is_local_testing() else "/lfs/archive/TCCIP_data/TReAD/V1/SFC/hr"

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

def get_tread_dataset(grid: xr.Dataset, start_date: str, end_date: str
                      ) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve and process TReAD dataset within the specified date range.

    Parameters:
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
    tread_files = get_file_paths(get_data_dir(), start_date, end_date)
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
