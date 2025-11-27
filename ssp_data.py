"""
Pipeline for generating CorrDiff-ready TaiESM high-res (3.5km) and low-res
(100km) datasets on a common reference grid.

This module provides:

- Reference grid handling
  - `get_ref_grid()` loads the fixed 304*304 WRF reference grid and returns:
    - a lightweight grid dataset with latitude/longitude (lat, lon),
    - the original grid coordinate variables (e.g. XLAT, XLONG).

- TaiESM 3.5 km (high-resolution) processing
  - `generate_taiesm3p5_output()`:
    - reads TaiESM 3.5km data for a given date range and SSP level,
    - regrids it to the reference grid,
    - stacks selected variables into a CorrDiff-style tensor,
    - computes per-channel metadata:
      - `cwb_variable` (variable names),
      - `cwb_pressure` (pressure levels, if applicable),
      - `cwb_center` (mean),
      - `cwb_scale` (standard deviation),
      - `cwb_valid` (valid time steps).

- TaiESM 100 km (low-resolution) processing
  - `generate_taiesm100_output()`:
    - reads TaiESM 100km data for the same period and SSP level,
    - regrids it to the same reference grid,
    - stacks selected variables into a low-res tensor (`era5`-style),
    - computes per-channel center, scale, and validity masks.

- Joint output
  - `generate_output_dataset()` ties everything together, returning:
    - the processed TaiESM 3.5km outputs,
    - the processed TaiESM 100km outputs,
    - the reference grid coordinates.

The resulting high-res and low-res products are spatially aligned and share a
consistent channel description, making them suitable as paired input/target
fields for CorrDiff training and evaluation under different SSP scenarios.
"""
from typing import Tuple
import xarray as xr

from taiesm3p5 import get_taiesm3p5_dataset, get_taiesm3p5_channels
from taiesm100 import get_taiesm100_dataset, get_taiesm100_channels
from cwa_data import GRID_COORD_KEYS, get_cwb_fields, get_era5_fields

# -------------------------------------------------------------------
# REF grid
# -------------------------------------------------------------------
REF_GRID_NC = "./ref_grid/wrf_304x304_grid_coords.nc"

def get_ref_grid() -> Tuple[xr.Dataset, dict]:
    """
    Load the reference grid dataset and extract its coordinates.

    This function reads a predefined reference grid NetCDF file and extracts:
    - A dataset containing latitude (`lat`) and longitude (`lon`) grids.
    - A dictionary of coordinate arrays specified by `GRID_COORD_KEYS`.

    Returns:
        tuple:
            - grid (xarray.Dataset): A dataset containing the latitude ('lat') and
              longitude ('lon') grids for spatial alignment.
            - grid_coords (dict): A dictionary of extracted coordinate arrays defined
              by `GRID_COORD_KEYS` for downstream processing.

    Notes:
        - The reference grid file path is defined by the global constant `REF_GRID_NC`.
        - The coordinate keys to extract are defined in `GRID_COORD_KEYS`.
    """
    # Reference grid paths
    ref = xr.open_dataset(REF_GRID_NC, engine='netcdf4')

    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }

    return grid, grid_coords

# -------------------------------------------------------------------
# TaiESM 3.5km & 100km outputs
# -------------------------------------------------------------------

def generate_output_dataset(start_date: str, end_date: str, ssp_level: str
                            ) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Generates output datasets for TaiESM 3.5km and TaiESM 100km data under a
    specified Shared Socioeconomic Pathway (SSP) level and date range.

    This function first retrieves a common reference grid and its coordinates.
    It then uses this grid to generate two xarray.Dataset objects: one for
    TaiESM 3.5km data and another for TaiESM 100km data. Both datasets are
    generated for the specified date range and according to the given SSP scenario.

    Args:
        start_date (str): The start date for generating the datasets in format "YYYYMMDD".
        end_date (str): The end date for generating the datasets in format "YYYYMMDD".
        ssp_level (str): The Shared Socioeconomic Pathway (SSP) level to be used
                         for generating the datasets. (e.g., "ssp126", "ssp245", "ssp585")

    Returns:
        Tuple[xr.Dataset, xr.Dataset, xr.Dataset]: A tuple containing three elements:
            - taiesm3p5_output (xr.Dataset): An xarray Dataset containing the
                                             generated TaiESM 3.5km output data.
            - taiesm100_output (xr.Dataset): An xarray Dataset containing the
                                             generated TaiESM 100km output data.
            - grid_coords (xr.Dataset): An xarray Dataset containing the
                                        coordinates of the reference grid.
    """
    grid, grid_coords = get_ref_grid()
    return (
        generate_taiesm3p5_output(grid, start_date, end_date, ssp_level),
        generate_taiesm100_output(grid, start_date, end_date, ssp_level),
        grid_coords
    )

def generate_taiesm3p5_output(
    grid: xr.Dataset,
    start_date: str,
    end_date: str,
    ssp_level: str = ''
) -> Tuple[
    xr.DataArray,  # TaiESM 3.5km dataarray
    xr.DataArray,  # TaiESM 3.5km variable
    xr.DataArray,  # TaiESM 3.5km center
    xr.DataArray,  # TaiESM 3.5km scale
    xr.DataArray,  # TaiESM 3.5km valid
    xr.Dataset,    # TaiESM 3.5km pre-regrid dataset
    xr.Dataset     # TaiESM 3.5km post-regrid dataset
]:
    """
    Generate processed TaiESM 3.5km outputs and corresponding CWB diagnostic
    DataArrays for a specified date range.

    Parameters
    ----------
    grid (xr.Dataset): Reference grid defining the target spatial domain for regridding.
    start_date (str): Start date in 'YYYYMMDD' format (inclusive).
    end_date (str): End date in 'YYYYMMDD' format (inclusive).
    ssp_level (str, optional): SSP level used to select the TaiESM dataset directory
                                (e.g., 'historical', 'ssp126', 'ssp245').

    Returns
    -------
    tuple
        A tuple containing:
        - xr.DataArray: Final processed TaiESM 3.5km tensor used by CorrDiff.
        - xr.DataArray: Names of variables included in the TaiESM 3.5km tensor.
        - xr.DataArray: Per-variable mean values (centering).
        - xr.DataArray: Per-variable standard deviations (scaling).
        - xr.DataArray: Boolean mask indicating valid time steps.
        - xr.Dataset: Native TaiESM 3.5km dataset before spatial regridding.
        - xr.Dataset: TaiESM 3.5km dataset regridded to the target domain.

    Notes
    -----
    This function encapsulates the full processing pipeline:
    loading TaiESM data, centering/scaling, computing validity flags,
    and regridding to the specified reference grid.
    """
    # Extract TaiESM 3.5km data from file.
    taiesm3p5_pre_regrid, taiesm3p5_out = \
        get_taiesm3p5_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM_3.5km dataset [{ssp_level}] =>\n {taiesm3p5_out}")

    # Generate cwb_* fields
    cwb_fields = get_cwb_fields(taiesm3p5_out, get_taiesm3p5_channels())
    return *cwb_fields, taiesm3p5_pre_regrid, taiesm3p5_out

def generate_taiesm100_output(
    grid: xr.Dataset,
    start_date: str,
    end_date: str,
    ssp_level: str = ''
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
    Processes TaiESM 100km data files to generate consolidated outputs, including
    the TaiESM 100km DataArray, its mean (center), standard deviation (scale),
    validity mask, and intermediate datasets.

    Parameters:
        grid (xarray.Dataset): The reference grid dataset for regridding.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

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
    taisem100_pre_regrid, taisem100_out = \
        get_taiesm100_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM 100km dataset [{ssp_level}] =>\n {taisem100_out}")

    # Generate era5_* fields
    era5_fields = get_era5_fields(taisem100_out, get_taiesm100_channels())
    return *era5_fields, taisem100_pre_regrid, taisem100_out