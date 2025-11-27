from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xarray as xr

from util import is_local_testing

TAIESM_100_CHANNELS = [
    {'name': 'pr', 'variable': 'precipitation'},
    # 500
    # {'name': 'z', 'pressure': 500, 'variable': 'geopotential_height'},
    # {'name': 't', 'pressure': 500, 'variable': 'temperature'},
    # {'name': 'u', 'pressure': 500, 'variable': 'eastward_wind'},
    # {'name': 'v', 'pressure': 500, 'variable': 'northward_wind'},
    # 700
    # {'name': 'z', 'pressure': 700, 'variable': 'geopotential_height'},
    # {'name': 't', 'pressure': 700, 'variable': 'temperature'},
    # {'name': 'u', 'pressure': 700, 'variable': 'eastward_wind'},
    # {'name': 'v', 'pressure': 700, 'variable': 'northward_wind'},
    # 850
    # {'name': 'z', 'pressure': 850, 'variable': 'geopotential_height'},
    # {'name': 't', 'pressure': 850, 'variable': 'temperature'},
    {'name': 'u', 'pressure': 850, 'variable': 'eastward_wind'},
    {'name': 'v', 'pressure': 850, 'variable': 'northward_wind'},
    # 925
    # {'name': 'z', 'pressure': 925, 'variable': 'geopotential_height'},
    # {'name': 't', 'pressure': 925, 'variable': 'temperature'},
    # {'name': 'u', 'pressure': 925, 'variable': 'eastward_wind'},
    # {'name': 'v', 'pressure': 925, 'variable': 'northward_wind'},
    # Remaining surface channels
    {'name': 'ts', 'variable': 'temperature_2m'},
    # {'name': 'u10', 'variable': 'eastward_wind_10m'},
    # {'name': 'v10', 'variable': 'northward_wind_10m'},
]

def get_taiesm100_channels() -> dict:
    """Returns TaiESM 100km channel list."""
    return TAIESM_100_CHANNELS

def get_data_dir(ssp_level: str) -> str:
    """
    Return the base directory for the TaiESM 3.5 km dataset based on the
    execution environment.

    Parameters
    ----------
    ssp_level (str): SSP level used to select the TaiESM dataset directory on BIG server.

    Returns
    -------
    str
        Path to the TaiESM 3.5 km data directory. This is:
        - `./data/taiesm100` when running locally (as detected by is_local_testing())
        - `/lfs/home/corrdiff/data/012-predictor_TaiESM1_ssp/{ssp_level}_daily/` when
          running on the BIG server.

    Notes
    -----
    This helper centralizes environment-aware path logic so other code does not
    need to handle local vs. remote directory differences.
    """
    return "./data/taiesm100" if is_local_testing() else \
            f"/lfs/home/corrdiff/data/012-predictor_TaiESM1_ssp/{ssp_level}_daily/"

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
        return [folder_path / f"TaiESM1_PRS_{var}_{yyyymm}_r1440x721_day.nc"
                for var in variables for yyyymm in date_range]

    return [
        folder_path / "PRS" / subfolder / var / yyyymm[:4] / \
            f"TaiESM1_PRS_{var}_{yyyymm}_r1440x721_day.nc"
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
        return [folder_path / f"TaiESM1_SFC_{var}_{yyyymm}_r1440x721_day.nc"
                for var in variables for yyyymm in date_range]

    return [
        folder_path / "SFC" / subfolder / var / yyyymm[:4] / \
            f"TaiESM1_SFC_{var}_{yyyymm}_r1440x721_day.nc"
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
        f"{ch['name']}{ch['pressure']}" for ch in TAIESM_100_CHANNELS if 'pressure' in ch
    ))

    # Read data from file
    prs_paths = get_prs_paths(folder, 'day', pressure_level_vars, duration.start, duration.stop)
    prs_data = xr.open_mfdataset(prs_paths, combine="by_coords")
    print(f'\nprs_data => {prs_data}')

    # Try to revise format to match ERA5
    prs_data = to_era5_like_format(prs_data)
    print(f'\n=== NEW prs_data => {prs_data}')

    return prs_data.sel(level=pressure_levels, time=duration)

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

    # Read data from file
    sfc_paths = get_sfc_paths(folder, 'day', surface_vars, duration.start, duration.stop)
    sfc_data = xr.open_mfdataset(sfc_paths, combine='by_coords')
    print(f'\nsfc_data => {sfc_data}')

    # Try to revise format to match ERA5
    sfc_data = to_era5_like_format(sfc_data.sel(time=duration))
    print(f'\n=== NEW sfc_data => {sfc_data}')

    sfc_data['pr'] = sfc_data['pr'] * 24 * 1000  # Convert unit to mm/day
    sfc_data['pr'].attrs['units'] = 'mm/day'

    return sfc_data

def get_taiesm100_dataset(grid: xr.Dataset, start_date: str, end_date: str,
                          ssp_level: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Retrieve, process, and regrid TaiESM 100km datasets for a specified date range, aligning with a
    reference grid.

    Parameters:
        grid (xarray.Dataset): The reference grid for regridding.
        start_date (str): The start date in 'YYYYMMDD' format.
        end_date (str): The end date in 'YYYYMMDD' format.
        ssp_level (str): SSP level used to select the TaiESM dataset directory.

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
            5. **Regridding:** Interpolate to match TaiESM 100km data to
                                the reference grid resolution.

        - The **final dataset** (`output_ds`) includes:
            - TaiESM 100km atmospheric and surface variables.

        - The dataset is renamed to standard variable names based on `TAIESM_100_CHANNELS`.
    """
    duration = slice(str(start_date), str(end_date))
    folder = get_data_dir(ssp_level)

    # Process and merge surface and pressure levels data
    sfc_prs_ds = xr.merge([
        get_surface_data(folder, duration),
        get_pressure_level_data(folder, duration)
    ])

    # Crop to Taiwan domain given TaiESM 100km is global data.
    lat, lon = grid.XLAT, grid.XLONG
    cropped_with_coords = sfc_prs_ds.sel(
        latitude=slice(lat.min().item(), lat.max().item()),
        longitude=slice(lon.min().item(), lon.max().item())
    ).rename({ ch['name']: ch['variable'] for ch in TAIESM_100_CHANNELS })

    # TODO - enlarge cropped_with_coords to output_ds
    output_ds = cropped_with_coords

    return cropped_with_coords, output_ds

def to_era5_like_format(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a TaiESM100 SFC/PRS-style dataset into an ERA5-like format.

    Common steps (SFC + PRS):
    - Rename spatial coordinates: (lat, lon) -> (latitude, longitude)
    - Convert CFTimeIndex (no-leap) to NumPy datetime64 via string roundtrip
      to avoid cftime → pandas conversion issues
    - Drop bounds variables not present in ERA5 samples: lat_bnds, lon_bnds

    PRS-specific steps (applied only if 'plev' is a coordinate):
    - Convert pressure levels from Pa -> hPa and keep them as a coordinate
    - Rename plev -> level
    - Rename TaiESM wind variables (ua, va) to ERA5 naming (u, v) when present

    Parameters
    ----------
    ds : xr.Dataset
        TaiESM100-style surface (SFC) or pressure-level (PRS) dataset.

    Returns
    -------
    xr.Dataset
        Dataset in an ERA5-like layout suitable for format checks and
        downstream processing.
    """
    is_prs_data = "plev" in ds.coords

    # Rename spatial + (optionally) PRS variables
    prs_name_mapping = {"plev": "level", "ua": "u", "va": "v"} if is_prs_data else {}
    out = ds.rename({
        "lat": "latitude",
        "lon": "longitude",
        **prs_name_mapping
    })

    # Always assign converted time; assign converted plev only for PRS datasets
    assign_kwargs = { "time": ("time", pd.to_datetime(out["time"].astype(str))) }
    if is_prs_data:
        assign_kwargs["level"] = out["level"] / 100.0   # Convert from Pa to hPa value
    out = out.assign_coords(**assign_kwargs)

    # drop bounds vars not present in ERA5 sample
    out = out.drop_vars(["lat_bnds", "lon_bnds"], errors="ignore")

    return out
