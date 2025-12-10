"""
Helpers for loading and preprocessing TaiESM 100 km predictor data.

This module provides utilities to:

- Define the TaiESM 100 km channel configuration (`TAIESM_100_CHANNELS`) used
  when building CorrDiff low-resolution inputs.
- Resolve data directories in a way that is aware of the execution environment
  (local testing vs. BIG server) via `get_data_dir`.
- Construct file paths for TaiESM 100 km surface (SFC) and pressure-level (PRS)
  NetCDF files over a given date range (`get_sfc_paths`, `get_prs_paths`).
- Load and subset TaiESM 100 km SFC and PRS data for a specified period
  (`get_surface_data`, `get_pressure_level_data`), including:
    * Converting TaiESM time coordinates (cftime no-leap) into NumPy
      `datetime64[ns]`.
    * Normalizing coordinates and variable names into an ERA5-like layout.
    * Converting precipitation units to mm/day.
- Assemble a merged TaiESM 100 km dataset aligned to a reference grid domain
  (`get_taiesm100_dataset`) suitable for downstream regridding and CorrDiff
  conditioning.

In short, this module is the TaiESM-100 km analogue of the ERA5 ingestion
pipeline: it standardizes file paths, variable naming, coordinate systems, and
basic units so that TaiESM 100 km data can be processed alongside ERA5 with
minimal special-case logic.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .era5 import BASELINE_CHANNELS
from .util import is_local_testing

TAIWAN_CLAT, TAIWAN_CLON = 23.6745, 120.9465  # Center latitude / longitude
TAIESM_100_CHANNELS = [
    ch for ch in BASELINE_CHANNELS
    if ch.get("pressure") != 925 and ch["name"] not in ("u10", "v10")
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
    return "../data/taiesm100" if is_local_testing() else \
            f"/lfs/home/corrdiff/data/013-TaiESM_Corrdiff/TaiESM1/{ssp_level}"

def get_prs_paths(
    folder: str,
    variables: List[str],
    start_date: str,
    end_date: str
) -> List[Path]:
    """
    Generate file paths for TaiESM 100km pressure level data files within a specified date range.

    Parameters:
        folder (str): The base directory containing the data files.
        variables (list of str): List of variable names to include.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        list: A list of file paths corresponding to the specified variables and date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    if is_local_testing():
        return [folder_path / "PRS" / f"TaiESM1_ssp126_r1i1p1f1_{var}_EA_{yyyymm}_day.nc"
                for var in variables for yyyymm in date_range]

    return [
        folder_path / var / f"TaiESM1_ssp126_r1i1p1f1_{var}_EA_{yyyymm}_day.nc"
        for var in variables for yyyymm in date_range
    ]

def get_sfc_paths(
    folder: str,
    variables: List[str],
    start_date: str,
    end_date: str
) -> List[Path]:
    """
    Generate file paths for TaiESM 100km surface data files within a specified date range.

    Parameters:
        folder (str): The base directory containing the data files.
        variables (list of str): List of variable names to include.
        start_date (str or datetime-like): The start date of the desired data range.
        end_date (str or datetime-like): The end date of the desired data range.

    Returns:
        list: A list of file paths corresponding to the specified variables and date range.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)
    if is_local_testing():
        return [folder_path / "SFC" / f"TaiESM1_ssp126_r1i1p1f1_{var}_EA_{yyyymm}_day.nc"
                for var in variables for yyyymm in date_range]

    return [
        folder_path / var / f"TaiESM1_ssp126_r1i1p1f1_{var}_EA_{yyyymm}_day.nc"
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
        ch['name'] for ch in TAIESM_100_CHANNELS if 'pressure' in ch
    ))

    # Read data from file
    prs_data = (
        xr.open_mfdataset(
            get_prs_paths(folder, pressure_level_vars, duration.start, duration.stop),
            combine="by_coords", compat="no_conflicts", data_vars="all"
        ).sel(level=pressure_levels, time=duration)
    )

    return prs_data

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
        ch['name'] for ch in TAIESM_100_CHANNELS if 'pressure' not in ch
    ))

    # Read data from file
    sfc_data = (
        xr.open_mfdataset(
            get_sfc_paths(folder, surface_vars, duration.start, duration.stop),
            combine='by_coords', compat="no_conflicts", data_vars="all"
        ).sel(time=duration)
    )

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
    sfc_prs_ds = convert_to_era5_format(
        xr.merge([
            get_surface_data(folder, duration),
            get_pressure_level_data(folder, duration),
        ], compat="no_conflicts")
    )

    # From Taiwan center, crop +/- 20 degrees lat/lon per discussion.
    cropped_with_coords = sfc_prs_ds.sel(
        latitude=slice(TAIWAN_CLAT - 20, TAIWAN_CLAT + 20),
        longitude=slice(TAIWAN_CLON - 20, TAIWAN_CLON + 20)
    ).rename({ ch['name']: ch['variable'] for ch in TAIESM_100_CHANNELS })  # rename variables

    # Expand cropped data to REF grid size, ignoring original latitude/longtitude.
    output_ds = expand_to_grid(cropped_with_coords, grid)

    return cropped_with_coords, output_ds

def convert_to_era5_format(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a TaiESM100 SFC or PRS dataset into an ERA5-compatible format.

    This function normalizes coordinate names, time formats, and selected
    variables so that TaiESM100 outputs resemble ERA5 low-resolution data.
    The resulting dataset can be passed directly into ERA5-style validation or
    downstream CorrDiff preprocessing.

    Standard transformations (applied to both SFC and PRS):
    - Rename spatial coordinates from TaiESM convention ("lat", "lon")
      to ERA5 convention ("latitude", "longitude").
    - Convert all time values to NumPy datetime64[ns] using a safe
      string round-trip, ensuring compatibility with xarray and NetCDF
      encoders (avoids CFTimeIndex / no-leap calendar issues).
    - Drop TaiESM-specific bounds and auxiliary variables that do not appear
      in ERA5 products (e.g., "lat_bnds", "lon_bnds", "height").

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
    # Rename spatial
    out = ds.rename({
        "lat": "latitude",
        "lon": "longitude"
    })

    # Always assign converted time for nc dump
    out = out.assign_coords(time=pd.to_datetime(out["time"].astype(str)))

    # Drop bounds vars not present in ERA5 sample
    out = out.drop_vars(["lat_bnds", "lon_bnds", "height"], errors="ignore")

    return out

def expand_to_grid(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Expand a coarse-resolution dataset onto a high-resolution target grid by
    interpolating in index space rather than geographic space.

    This function takes an input dataset defined on a (latitude, longitude)
    grid of shape (nlat, nlon) and resamples it to match the REF grid
    described by (south_north, west_east). The physical latitude/longitude
    coordinates of the coarse dataset are ignored; only their index positions
    are used. This produces an efficient and deterministic "index-space"
    interpolation suitable for embedding low-resolution fields into a target
    model grid.

    Parameters
    ----------
    ds : xr.Dataset
        Low-resolution dataset with dimensions (time, [level], latitude, longitude).
        The physical values of latitude/longitude are ignored; only their index order
        is used for interpolation.
    grid : xr.Dataset
        Dataset containing REF grid dimensions (south_north, west_east) and
        2D fields XLAT and XLONG.

    Returns
    -------
    xr.Dataset
        A new dataset with dimensions (time, [level], south_north, west_east),
        containing index-interpolated variables aligned with the REF grid and
        annotated with XLAT/XLONG spatial coordinates. The attribute
        `regrid_method="bilinear"` is added for provenance.

    Notes
    -----
    - This method does *not* perform geographic interpolation. It is strictly
      an index-based expansion to embed large-scale fields into a high-resolution
      domain.
    - Useful when coarse fields serve as contextual features for fine-grid models
      and the fine-grid spatial structure is supplied by XLAT/XLONG.
    """
    # Extract dataset and REF grid sizes
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    ny, nx = grid.sizes["south_north"], grid.sizes["west_east"]

    # Treat latitude/longitude as pure indices
    ds_idx = ds.assign_coords(
        latitude=("latitude", np.arange(nlat)),
        longitude=("longitude", np.arange(nlon)),
    )

    # Index-space positions for REF grid
    new_i = xr.DataArray(np.linspace(0, nlat - 1, ny), dims=("south_north",), name="latitude")
    new_j = xr.DataArray(np.linspace(0, nlon - 1, nx), dims=("west_east",), name="longitude")

    # Interpolate in index space, remove latitude/longitude, and attach REF grid coords
    expanded = (
        ds_idx
        .interp(latitude=new_i, longitude=new_j)
        .drop_vars(["latitude", "longitude", "time_bnds"], errors="ignore")
        .assign_coords(
            XLAT=(("south_north", "west_east"), grid["XLAT"].values),
            XLONG=(("south_north", "west_east"), grid["XLONG"].values)
        )
    )

    # Ensure level is a coordinate if it exists
    if "level" in expanded.dims and "level" not in expanded.coords:
        expanded = expanded.assign_coords(level=ds["level"])

    # Rechunk for desired pattern.
    # e.g. precipitation: (1, 208, 208), winds: (1, 1, 208, 208)
    chunks = {"time": 1, "south_north": ny, "west_east": nx}
    if "level" in expanded.dims:
        chunks["level"] = 1
    expanded = expanded.chunk(chunks)

    return expanded.assign_attrs(regrid_method="bilinear")
