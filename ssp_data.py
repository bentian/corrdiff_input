"""
TaiESM SSP processing utilities for CorrDiff training inputs.

This module generates CorrDiff-ready high-resolution and low-resolution tensors
from the TaiESM climate model under a specified Shared Socioeconomic Pathway
(SSP) scenario. It supports two model resolutions:

    • TaiESM 3.5 km   (high-resolution, WRF-downscaled)
    • TaiESM 100 km   (low-resolution, native TaiESM atmospheric fields)

Major features provided by this module
--------------------------------------

1. **Reference Grid Loading**
   - `get_ref_grid()` loads a WRF-style reference grid (`REF_GRID_NC`)
     and extracts:
       • latitude / longitude fields  
       • grid coordinate metadata (`GRID_COORD_KEYS`)  
     This ensures the 3.5 km and 100 km datasets are aligned to the same
     spatial domain and indexing.

2. **Dataset Generation**
   - `generate_output_dataset(start_date, end_date, ssp_level)` runs the
     complete processing pipeline for both TaiESM datasets:
       • Loads model data for the specified date range  
       • Crops / regrids to the reference grid  
       • Converts variables into CorrDiff-ready stacked tensors  
       • Computes per-channel diagnostics (mean, std, validity)

3. **Shared CorrDiff-ready Outputs**
   - High-resolution TaiESM 3.5 km fields use the CWB-format utilities
     from `cwa_data` (`get_cwb_fields`).
   - Low-resolution TaiESM 100 km fields use the ERA5-format utilities
     (`get_era5_fields`), since both share the same stacked-tensor structure.

Returned objects follow the standard CorrDiff convention:
    (tensor, variable_names, center, scale, valid_mask, pre_regrid_ds, post_regrid_ds)

This module acts as the SSP counterpart to the TReAD/ERA5 `cwa_data` pipeline:
it ensures all TaiESM inputs (both high-res and low-res) can be produced
consistently and merged into CorrDiff Zarr datasets.
"""
from typing import Tuple
import xarray as xr

from taiesm3p5 import get_taiesm3p5_dataset, get_taiesm3p5_channels
from taiesm100 import get_taiesm100_dataset, get_taiesm100_channels
from cwa_data import GRID_COORD_KEYS, get_cwb_fields, get_era5_fields

# -------------------------------------------------------------------
# REF grid
# -------------------------------------------------------------------
REF_GRID_NC = "./ref_grid/ssp_208x208_grid_coords.nc"

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
            - taiesm3p5_outputs (xr.Dataset): An xarray Dataset containing the
                                              generated TaiESM 3.5km output data.
            - taiesm100_outputs (xr.Dataset): An xarray Dataset containing the
                                              generated TaiESM 100km output data.
            - grid_coords (xr.Dataset): An xarray Dataset containing the
                                        coordinates of the reference grid.
    """
    grid, grid_coords = get_ref_grid()

    # TaiESM 3.5km
    taiesm3p5_pre_regrid, taiesm3p5_out = \
        get_taiesm3p5_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM_3.5km dataset [{ssp_level}] =>\n {taiesm3p5_out}")
    taiesm3p5_outputs = (
        *get_cwb_fields(taiesm3p5_out, get_taiesm3p5_channels()),
        taiesm3p5_pre_regrid, taiesm3p5_out
    )

    # TaiESM 100km
    taisem100_pre_regrid, taisem100_out = \
        get_taiesm100_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM 100km dataset [{ssp_level}] =>\n {taisem100_out}")
    taiesm100_outputs = (
        *get_era5_fields(taisem100_out, get_taiesm100_channels()),
        taisem100_pre_regrid, taisem100_out
    )

    return taiesm3p5_outputs, taiesm100_outputs, grid_coords
