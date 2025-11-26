from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xarray as xr

from util import is_local_testing, regrid_dataset

TAIESM_3P5_CHANNELS = {
    # Baseline
    "RAINNC": "precipitation",
    "T2MEAN": "temperature_2m",
    "U10MEAN": "eastward_wind_10m",
    "V10MEAN": "northward_wind_10m",
}

def get_taiesm3p5_channels() -> Dict[str, str]:
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
        Path to the TaiESM 3.5km data directory. This is:
        - `./data/taiesm3p5` when running locally (as detected by is_local_testing())
        - `/lfs/archive/TCCIP_data/TaiESM-WRF/TAIESM_tw3.5km_<ssp_level>` when
          running on the BIG server.

    Notes
    -----
    This helper centralizes environment-aware path logic so other code does not
    need to handle local vs. remote directory differences.
    """
    return "./data/taiesm3p5" if is_local_testing() else \
            f"/lfs/archive/TCCIP_data/TaiESM-WRF/TAIESM_tw3.5km_{ssp_level}"

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
    file_paths = get_file_paths(get_data_dir(ssp_level), start_date, end_date)
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

    # Crop & attach coordinates per REF grid, and rename variables.
    cropped_with_coords = (
        # FIXME - Remove hardcoded lat/lon once XLAT & XLONG are available
        surface_ds
            .isel(south_north=slice(0, 304), west_east=slice(4, 308))
            .assign_coords(
                lat=(("south_north", "west_east"), grid.XLAT.data),
                lon=(("south_north", "west_east"), grid.XLONG.data),
            )
            .rename(TAIESM_3P5_CHANNELS)
    )

    # Based on REF grid, regrid TaiESM 3.5km data over spatial dimensions for all timestamps.
    output_ds = regrid_dataset(cropped_with_coords, grid)

    return cropped_with_coords, output_ds