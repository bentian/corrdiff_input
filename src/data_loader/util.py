"""
Utilities for ERA5-format validation and regridding onto a target WRF-style grid.

This module provides small, focused helpers for:
- Detecting whether the code is running on the local workstation or an HPC
  environment (`is_local_testing`).
- Regridding arbitrary `xarray.Dataset` objects onto a target grid using
  bilinear interpolation via xESMF (`regrid_dataset`).
- Validating that input ERA5 datasets conform to expected surface (SFC) and
  pressure-level (PRS) conventions (`verify_lowres_sfc_format`,
  `verify_lowres_prs_format`).

The verification routines enforce a consistent ERA5 layout, checking:
- Required dimensions and coordinates (time, bnds, level, latitude, longitude)
- Coordinate dtypes (time as datetime64, others numeric)
- Presence and shapes of key data variables (e.g., `tp`, `t2m`, or a
  user-specified 4D variable)

These utilities are intended to be used as early sanity checks and
preprocessing steps before further regridding, stacking into CorrDiff-ready
tensors, or model training / inference.
"""
from pathlib import Path

import xesmf as xe
import xarray as xr

def is_local_testing() -> bool:
    """
    Determines if the current environment is set up for local testing.

    Returns:
    bool: True if the environment is for local testing; False otherwise.
    """
    return not Path("/lfs/archive/Reanalysis/").exists()


def regrid_dataset(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Regrids the input dataset to match the target grid using bilinear interpolation.

    Parameters:
    ds (xr.Dataset): The source dataset to be regridded.
    grid (xr.Dataset): The target grid dataset defining the desired spatial dimensions.

    Returns:
    xr.Dataset: The regridded dataset aligned with the target grid.
    """
    # Regrid the dataset to the target grid:
    # - Use bilinear interpolation to regrid the data.
    # - Extrapolate by using the nearest valid source cell to extrapolate values for
    #   target points outside the source grid.
    remap = xe.Regridder(ds, grid, method="bilinear", extrap_method="nearest_s2d")

    # Regrid each time step while keeping the original coordinates and dimensions
    ds_regrid = xr.concat(
        [remap(ds.isel(time=i)).assign_coords(time=ds.time[i])
            for i in range(ds.sizes["time"])],
        dim="time"
    )

    return ds_regrid
