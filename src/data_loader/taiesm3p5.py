"""
TaiESM 3.5 km loader and preprocessor for CorrDiff high-resolution inputs.

This module is responsible for:
- Locating TaiESM 3.5 km daily WRF outputs based on the execution environment:
    * local testing:    ./data/taiesm3p5
    * BIG server:       /lfs/archive/TCCIP_data/TaiESM-WRF/TAIESM_tw3.5km_<ssp_level>
- Building monthly file paths for a requested date range.
- Loading and stitching multiple NetCDF files with `xarray.open_mfdataset`.
- Converting WRF "Times" byte strings to a proper `time` coordinate and
  sub-selecting a user-specified time window.
- Cropping the native TaiESM 3.5 km domain to the CorrDiff reference grid and
  attaching latitude/longitude coordinates from that grid.
- Regridding the cropped dataset to the reference grid using `regrid_dataset`.
- Renaming model variable names to CorrDiff-friendly channel names.

Key public helpers:
    - `get_taiesm3p5_channels()`:
        Returns the mapping from raw TaiESM variable names (e.g., "RAINNC",
        "T2MEAN") to CorrDiff variable names (e.g., "precipitation",
        "temperature_2m").
    - `get_data_dir(ssp_level)`:
        Resolves the TaiESM 3.5 km base directory, choosing between local and
        BIG server paths and embedding the requested SSP level.
    - `get_file_paths(folder, start_date, end_date)`:
        Builds the list of monthly TaiESM files covering the requested period.
    - `get_taiesm3p5_dataset(grid, start_date, end_date, ssp_level)`:
        High-level entry point that loads, crops, regrids, and renames the
        TaiESM 3.5 km data, returning both the pre-regrid and regridded
        datasets ready for CorrDiff.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xarray as xr

from .util import is_local_testing, regrid_dataset

TAIESM_3P5_CHANNELS = {
    "PRCP": "precipitation",
    "T2": "temperature_2m",
    "U10": "eastward_wind_10m",
    "V10": "northward_wind_10m",
}

def get_taiesm3p5_channels() -> Dict[str, str]:
    """Returns TaiESM 3.5km channel list."""
    return TAIESM_3P5_CHANNELS

def get_data_dir(ssp_level: str) -> str:
    """
    Return the base directory for the TaiESM 3.5km dataset based on the
    execution environment.

    Parameters
    ----------
    ssp_level (str): SSP level used to select the TaiESM dataset directory on BIG server.

    Returns
    -------
    str
        Path to the TaiESM 3.5km data directory.

    Notes
    -----
    This helper centralizes environment-aware path logic so other code does not
    need to handle local vs. remote directory differences.
    """
    return "../data/taiesm3p5" if is_local_testing() else \
            f"/lfs/home/corrdiff/data/013-TaiESM_Corrdiff/TaiESM1-WRF/{ssp_level}"

def get_file_paths(folder: str, start_date: str, end_date: str, ssp_level: str) -> List[str]:
    """
    Generate a list of file paths for the specified date range.

    Parameters:
        folder (str): The directory containing the files.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.
        ssp_level (str): SSP level used to compose the TaiESM dataset file name.

    Returns:
        list: A list of file paths corresponding to each month in the date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    return [folder_path / f"TaiESM1-WRF_tw3.5_{ssp_level}_wrfday_d01_{yyyymm}.nc"
            for yyyymm in date_range]

def get_taiesm3p5_dataset(grid: xr.Dataset, start_date: str, end_date: str,
                          ssp_level: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve and process TaiESM 3.5km dataset within the specified date range.

    Parameters:
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.
        ssp_level (str): SSP level used to select the TaiESM dataset directory.

    Returns:
        tuple: A tuple containing the original and regridded TaiESM 3.5km datasets.
    """
    surface_var_names = list(TAIESM_3P5_CHANNELS.keys())
    start_datetime = pd.to_datetime(str(start_date), format='%Y%m%d')
    end_datetime = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Read surface level data.
    surface_ds = xr.open_mfdataset(
        get_file_paths(get_data_dir(ssp_level), start_date, end_date, ssp_level),
        preprocess=lambda ds: (
            ds[surface_var_names].assign_coords(            # attach new time coord
                time=pd.to_datetime(ds["Times"].astype(str), format="%Y-%m-%d_%H:%M:%S"))
            .rename({"Time": "time"})                       # unify time dimension name
            .sel(time=slice(start_datetime, end_datetime))  # select requested dates
        )
    ).rename(TAIESM_3P5_CHANNELS)

    # Based on REF grid, regrid TaiESM 3.5km data over spatial dimensions for all timestamps.
    output_ds = (
        regrid_dataset(surface_ds, grid)
        .chunk({"time": 1, "south_north": -1, "west_east": -1}) # chunk to grid size per timestep
    )

    return surface_ds, output_ds
