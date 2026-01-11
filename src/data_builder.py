"""
CorrDiff reference-grid utilities and dataset assembly pipelines.

This module centralizes the logic needed to build “CorrDiff-ready” training and
inference inputs across multiple climate data sources on a unified WRF-style grid.
It supports:

- CWA workflow: high-resolution TReAD + low-resolution ERA5
- SSP workflow: high-resolution TaiESM 3.5 km + low-resolution TaiESM 100 km
- CORDEX workflow: high-resolution CORDEX targets + low-resolution CORDEX predictors (train/test)

All pipelines return standardized output tuples produced by:
- `get_cwb_fields()` for HR-like sources (TReAD, TaiESM 3.5 km, CORDEX HR)
- `get_era5_fields()` for LR-like sources (ERA5, TaiESM 100 km, CORDEX LR)

Each tuple includes:
- normalized tensors and metadata (center/scale/valid masks)
- the raw pre-regrid dataset
- the final post-regrid (aligned) dataset

Reference grids
---------------
The module loads one of two 208x208 WRF-style reference grids:

- CWA reference grid (used for TReAD/ERA5): includes optional terrain fields
  (TER, SLOPE, ASPECT) for downstream preprocessing.
- SSP reference grid (used for TaiESM SSP scenarios): terrain fields are omitted.

Both expose:
- 2D latitude/longitude as `XLAT` / `XLONG`
- `grid_coords` containing only the grid-coordinate keys used by CorrDiff:
  `GRID_COORD_KEYS = ["XLAT", "XLONG"]`

Output convention
-----------------
All returned aligned datasets follow CorrDiff's WRF-style spatial convention:

    (time, south_north, west_east)              for HR / surface fields
    (time, level, south_north, west_east)       for LR pressure-level fields

Large arrays remain Dask-backed to support scalable I/O and training workloads.
"""

from typing import Tuple
import xarray as xr

from data_loader import (
    get_tread_dataset, get_tread_channels,
    get_era5_dataset, get_era5_channels,
    get_taiesm3p5_dataset, get_taiesm3p5_channels,
    get_taiesm100_dataset, get_taiesm100_channels,
    get_cordex_train_datasets, get_cordex_test_datasets,
    get_cordex_hr_channels, get_cordex_lr_channels,
)
from tensor_fields import get_cwb_fields, get_era5_fields

# -------------------------------------------------------------------
# REF grid
# -------------------------------------------------------------------
CWA_REF_GRID = "../ref_grid/wrf_208x208_grid_coords.nc"
SSP_REF_GRID = "../ref_grid/ssp_208x208_grid_coords.nc"
GRID_COORD_KEYS = ["XLAT", "XLONG"]


def get_ref_grid(ssp_level: str = '') -> Tuple[xr.Dataset, dict, dict]:
    """
    Load the reference grid used for spatial alignment of model inputs and outputs.

    This function loads either the SSP reference grid or the CWA reference grid
    depending on the `ssp_level` parameter. It extracts the 2D latitude/longitude
    fields required for interpolation or expansion onto the target grid, plus
    additional grid-related coordinate variables and optional terrain metadata.

    Parameters
    ----------
    ssp_level : str, optional
        Scenario identifier. If a non-empty string is provided, the SSP reference
        grid is loaded. If empty (default), the CWA reference grid is used.
        The terrain fields (TER, SLOPE, ASPECT) are included only when
        `ssp_level` is empty.

    Returns
    -------
    grid : xr.Dataset
        Dataset containing the target grid's spatial coordinate fields:
        - `lat`: 2D latitude field (south_north x west_east)
        - `lon`: 2D longitude field (south_north x west_east)

    grid_coords : dict
        Dictionary of additional coordinate variables extracted from the
        reference file, keyed by `GRID_COORD_KEYS`. These may include
        projection metadata, map factors, or other grid descriptors.

    terrain : dict
        Terrain-related fields from the reference grid:
        - empty dict when `ssp_level` is provided
        - otherwise contains:
            { "ter": TER, "slope": SLOPE, "aspect": ASPECT }
        where each value is an xarray DataArray.

    Notes
    -----
    - This function does not modify or preprocess the grid; it merely loads and
      exposes the required pieces for downstream interpolation or expansion.
    - `grid` is intended to be passed to functions such as `expand_to_grid()`.
    """
    # Choose reference grid file based on whether we are using an SSP scenario
    ref_grid_path = SSP_REF_GRID if ssp_level else CWA_REF_GRID

    # Open reference grid
    ref = xr.open_dataset(ref_grid_path , engine='netcdf4')

    # Extra grid-related coordinates
    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }

    # Terrain fields only for the CWA grid
    if ssp_level:
        terrain = {}
    else:
        terrain = { key.lower(): ref[key] for key in ["TER", "SLOPE", "ASPECT"] }

    return grid, grid_coords, terrain


# -------------------------------------------------------------------
# TReAD & ERA5 outputs
# -------------------------------------------------------------------

def generate_cwa_outputs(start_date: str, end_date: str
                         ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Generates output datasets for TReAD and ERA5 based on a specified date range
    and a common reference grid.

    This function first retrieves a reference grid, its coordinates, and terrain layer
    information. It then uses this grid to generate two xarray.Dataset objects:
    one representing TReAD output and another for ERA5 output, both
    constrained by the given start and end dates.

    Args:
        start_date (str): The start date for generating the datasets in format "YYYYMMDD".
        end_date (str): The end date for generating the datasets in format "YYYYMMDD".

    Returns:
        Tuple[xr.Dataset, xr.Dataset, xr.Dataset]: A tuple containing three elements:
            - tread_outputs (xr.Dataset): An xarray Dataset containing the
                                          generated TReAD output data.
            - era5_outputs (xr.Dataset): An xarray Dataset containing the
                                         generated ERA5 output data.
            - grid_coords (xr.Dataset): An xarray Dataset containing the
                                        coordinates of the reference grid.
    """
    grid, grid_coords, terrain = get_ref_grid()

    # TReAD
    tread_pre_regrid, tread_out = get_tread_dataset(grid, start_date, end_date)
    print(f"\nTReAD dataset =>\n {tread_out}")
    tread_outputs = (
        *get_cwb_fields(tread_out, get_tread_channels()),
        tread_pre_regrid,
        tread_out
    )

    # ERA5
    era5_pre_regrid, era5_out = get_era5_dataset(grid, terrain, start_date, end_date)
    print(f"\nERA5 dataset =>\n {era5_out}")
    era5_outputs = (
        *get_era5_fields(era5_out, get_era5_channels()),
        era5_pre_regrid,
        era5_out
    )

    return tread_outputs, era5_outputs, grid_coords


# -------------------------------------------------------------------
# TaiESM 3.5km & 100km outputs
# -------------------------------------------------------------------

def generate_ssp_outputs(start_date: str, end_date: str, ssp_level: str
                         ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
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
    grid, grid_coords, _ = get_ref_grid(ssp_level)

    # TaiESM 3.5km
    taiesm3p5_pre_regrid, taiesm3p5_out = \
        get_taiesm3p5_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM_3.5km dataset [{ssp_level}] =>\n {taiesm3p5_out}")
    taiesm3p5_outputs = (
        *get_cwb_fields(taiesm3p5_out, get_taiesm3p5_channels()),
        taiesm3p5_pre_regrid,
        taiesm3p5_out
    )

    # TaiESM 100km
    taisem100_pre_regrid, taisem100_out = \
        get_taiesm100_dataset(grid, start_date, end_date, ssp_level)
    print(f"\nTaiESM 100km dataset [{ssp_level}] =>\n {taisem100_out}")
    taiesm100_outputs = (
        *get_era5_fields(taisem100_out, get_taiesm100_channels()),
        taisem100_pre_regrid,
        taisem100_out
    )

    return taiesm3p5_outputs, taiesm100_outputs, grid_coords


def validate_ssp_level(raw: str) -> str:
    """
    Validate and normalize an SSP suffix string.
    """
    allowed_ssp_levels = {"historical", "ssp126", "ssp245", "ssp370", "ssp585"}
    if raw not in allowed_ssp_levels:
        raise ValueError(f"ssp_level must be one of {allowed_ssp_levels}")

    return raw


# -------------------------------------------------------------------
# Cordex outputs
# -------------------------------------------------------------------

def _assemble_cordex_outputs(
    hr_out: xr.Dataset,
    lr_pre: xr.Dataset,
    lr_out: xr.Dataset,
    grid_coords,
) -> Tuple[tuple, tuple, xr.Dataset]:
    """
    Assemble CorrDiff-ready HR and LR output tuples.

    This helper constructs the standardized output tuples expected by the
    CorrDiff pipeline by:
    - extracting and normalizing HR (CWB-style) fields from `hr_out`
    - extracting and normalizing LR (ERA5-style) fields from `lr_out`
    - appending pre- and post-regrid datasets in the expected order

    Parameters
    ----------
    hr_out : xr.Dataset
        Final high-resolution dataset on the CorrDiff grid.
    lr_pre : xr.Dataset
        Low-resolution dataset before regridding (stacked by level).
    lr_out : xr.Dataset
        Low-resolution dataset after regridding to the CorrDiff grid.
    grid_coords : xr.Dataset or dict
        Dataset or mapping containing grid coordinate arrays (e.g., XLAT, XLONG).

    Returns
    -------
    hr_outputs : tuple
        Tuple of HR outputs in CorrDiff order:
        (fields, variable metadata, center, scale, valid mask, pre_regrid, post_regrid).
    lr_outputs : tuple
        Tuple of LR outputs in CorrDiff order:
        (fields, metadata, center, scale, valid mask, pre_regrid, post_regrid).
    grid_coords : xr.Dataset or dict
        The unchanged grid coordinate container, forwarded for downstream use.
    """
    hr_outputs = (*get_cwb_fields(hr_out, get_cordex_hr_channels()), hr_out, hr_out)
    lr_outputs = (*get_era5_fields(lr_out, get_cordex_lr_channels()), lr_pre, lr_out)
    return hr_outputs, lr_outputs, grid_coords


def generate_cordex_train_outputs(
    exp_domain: str,
    train_config: str
) -> Tuple[tuple, tuple, xr.Dataset]:
    """Generate CorrDiff training outputs from CORDEX datasets."""
    return _assemble_cordex_outputs(
        *get_cordex_train_datasets(exp_domain, train_config)
    )


def generate_cordex_test_outputs(
    exp_domain: str,
    train_config: str,
    test_config: str,
    perfect: bool
) -> Tuple[tuple, tuple, xr.Dataset]:
    """Generate CorrDiff test outputs from CORDEX datasets."""
    return _assemble_cordex_outputs(
        *get_cordex_test_datasets(exp_domain, train_config, test_config, perfect)
    )
