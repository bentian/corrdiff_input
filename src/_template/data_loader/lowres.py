"""
CorrDiff low-resolution data loader (TEMPLATE).

Based on attached examples:
- era5.py: ERA5 LR loader (PRS + SFC + static terrain) + crop + regrid + inject terrain layers
- taiesm100.py: TaiESM 100km LR loader (PRS + SFC) + time conversion + ERA5-like normalization

What you should customize:
1) DATASET_NAME + get_data_dir(): your LR source(s) and storage layout
2) CHANNELS spec: which variables/pressure-levels become CorrDiff channels
3) File patterns in get_prs_paths / get_sfc_paths / get_static_paths
4) preprocess() in open_mfdataset(): time parsing, coord normalization, var renames, unit fixes
5) crop_to_grid(): how you subset to your reference domain (lat/lon or WRF dims)
6) Terrain injection (oro/slope/aspect) + any derived channels (e.g., wtp)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------
# Project utilities (expected to exist in your repo)
# ---------------------------------------------------------------------
# - is_local_testing(): returns True when running locally, False on BIG server
# - regrid_dataset(ds, grid): regrid ds onto grid (e.g., xESMF / custom)
# - (optional) crop helpers
#
from .util import regrid_dataset


# ---------------------------------------------------------------------
# 1) Channel specification (EDIT ME)
# ---------------------------------------------------------------------
# Following era5.py style: each channel is a dict:
#   - surface:  {"name": "t2m", "variable": "temperature_2m"}
#   - pressure: {"name": "t", "pressure": 850, "variable": "temperature"}
#
# CorrDiff will typically consume a stacked channel list (or mapping derived from it).
BASELINE_CHANNELS: List[Dict[str, object]] = [
    # TODO: edit to match your LR conditioning
    # {"name": "tp",  "variable": "precipitation"},
    # {"name": "t2m", "variable": "temperature_2m"},
    # *[
    #     {"name": name, "pressure": p, "variable": var}
    #     for p in (500, 700, 850, 925)
    #     for name, var in (
    #         ("z", "geopotential_height"),
    #         ("t", "temperature"),
    #         ("u", "eastward_wind"),
    #         ("v", "northward_wind"),
    #     )
    # ],
]

# Optional static/aux channels (era5.py pattern)
STATIC_CHANNELS: List[Dict[str, object]] = [
    # {"name": "oro",    "variable": "terrain_height"},
    # {"name": "slope",  "variable": "slope_angle"},
    # {"name": "aspect", "variable": "slope_aspect"},
]

LR_CHANNELS: List[Dict[str, object]] = [*BASELINE_CHANNELS, *STATIC_CHANNELS]


def get_lr_channels() -> List[Dict[str, object]]:
    """Return LR channel list used to build CorrDiff conditioning."""
    return LR_CHANNELS


# ---------------------------------------------------------------------
# 2) Paths + file list builders (EDIT ME)
# ---------------------------------------------------------------------
def _get_data_dir(*, scenario: Optional[str] = None) -> str:
    """
    Return base directory for LR data.

    Instructions:
    - For ERA5: environment-aware path (../data/era5 vs /lfs/archive/Reanalysis/ERA5)
    - For GCM (TaiESM): include `scenario` / `ssp_level` in the server path if needed
    """
    # TODO: Return base directory for this LR dataset
    # Example pattern:
    # return "../data/your_lr" if is_local_testing() else f"/lfs/archive/.../{scenario}"
    raise NotImplementedError("Implement get_data_dir() for your environment")


def _get_prs_paths(folder: str, variables: List[str], start_date: str, end_date: str) -> List[Path]:
    """
    Build monthly file list for pressure-level data.

    Instructions:
    - Use pd.date_range(..., freq="MS").strftime("%Y%m")
    - Compose filenames that match your archive (ERA5 PRS folder vs GCM per-variable folders)
    - Return list[Path] ordered but open_mfdataset can handle multi-var multi-month
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)

    # TODO: replace with your PRS naming convention.
    # Example (ERA5-like): folder_path/"PRS"/"<subfolder>"/f"{var}_{yyyymm}.nc"
    # Example (TaiESM-like): folder_path/var/f"TaiESM1_{scenario}_{var}_EA_{yyyymm}_day.nc"
    FILE_PATTERN = "PRS/{var}_{yyyymm}.nc"

    return [folder_path / FILE_PATTERN.format(var=var, yyyymm=yyyymm)
            for var in variables for yyyymm in date_range]


def _get_sfc_paths(folder: str, variables: List[str], start_date: str, end_date: str) -> List[Path]:
    """
    Build monthly file list for surface data.

    Instructions:
    - Same approach as get_prs_paths but for SFC folder / naming.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)

    # TODO: replace with your SFC naming convention.
    FILE_PATTERN = "SFC/{var}_{yyyymm}.nc"

    return [folder_path / FILE_PATTERN.format(var=var, yyyymm=yyyymm)
            for var in variables for yyyymm in date_range]


def _get_static_paths(folder: str) -> Dict[str, Path]:
    """
    Return file paths for static layers (orography / slope / aspect).

    Instructions:
    - era5.py loads a global/region orography and then later injects HR terrain.
    - If you don't have static files, you can skip this and return {}.
    """
    folder_path = Path(folder)
    return {
        # TODO: set correct paths if available
        # "oro": folder_path / "static" / "orography.nc",
        # "slope": folder_path / "static" / "slope.nc",
        # "aspect": folder_path / "static" / "aspect.nc",
    }


# ---------------------------------------------------------------------
# 3) Open + normalize helpers (EDIT ME)
# ---------------------------------------------------------------------
def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize time coordinate to numpy datetime64[ns].

    Instructions:
    - ERA5 usually already has a proper time coordinate.
    - TaiESM example converts cftime no-leap -> pandas/np datetime64.
      If your dataset uses cftime, implement conversion here.
    """
    # If ds["time"] is cftime:
    # ds["time"] = pd.to_datetime(ds.indexes["time"].to_datetimeindex()).values
    return ds


def _normalize_coords_and_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize coordinates and variable naming to an ERA5-like layout.

    Instructions:
    - taiesm100.py renames coords/vars so downstream code can reuse ERA5 logic.
    - Make sure you have consistent:
        * coords: time, lat, lon (or y/x)
        * vertical: level (pressure levels) if needed
        * variable names match your channel 'variable' keys (or map them here)
    """
    # Examples:
    # ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # ds = ds.rename({"Z": "geopotential_height", "T": "temperature"})
    return ds


def _open_monthlies(paths: List[Path], *, preprocess=None) -> xr.Dataset:
    """Thin wrapper around open_mfdataset."""
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        preprocess=preprocess,
        # engine="netcdf4",  # uncomment if needed
        # parallel=True,     # if dask desired
    )


# ---------------------------------------------------------------------
# 4) Load PRS + SFC and merge into one LR dataset
# ---------------------------------------------------------------------
def _get_pressure_level_data(folder: str, duration: slice,
                             *, scenario: Optional[str] = None) -> xr.Dataset:
    """
    Load LR pressure-level data (multi-month) and return normalized Dataset.

    Instructions:
    - Determine pressure levels and variables from get_lr_channels()
    - Build file paths for the needed PRS variables
    - open_mfdataset + subset time + normalize time/coords/vars
    - Subset pressure levels if file contains many levels
    """
    channels = get_lr_channels()
    prs_vars = sorted({ch["variable"] for ch in channels if "pressure" in ch})
    prs_levels = sorted({int(ch["pressure"]) for ch in channels if "pressure" in ch})

    start = duration.start.strftime("%Y%m%d")
    end = duration.stop.strftime("%Y%m%d")

    paths = _get_prs_paths(folder, prs_vars, start, end)

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        ds = ds[prs_vars]
        ds = _normalize_time(ds)
        ds = _normalize_coords_and_vars(ds)
        if "time" in ds.coords:
            ds = ds.sel(time=duration)
        # pressure coord name differs across datasets (level/plev/etc)
        if "level" in ds.coords:
            ds = ds.sel(level=prs_levels)
        elif "plev" in ds.coords:
            ds = ds.sel(plev=[p * 100 for p in prs_levels])  # if Pa
        return ds

    ds_prs = _open_monthlies(paths, preprocess=_preprocess)
    return ds_prs


def _get_surface_data(folder: str, duration: slice,
                      *, scenario: Optional[str] = None) -> xr.Dataset:
    """
    Load LR surface data (multi-month) and return normalized Dataset.

    Instructions:
    - Determine surface variables from get_lr_channels()
    - open_mfdataset + subset time + normalize
    - Apply unit conversions (e.g., precipitation to mm/day) if needed
    """
    channels = get_lr_channels()
    sfc_vars = sorted({ch["variable"] for ch in channels if "pressure" not in ch})
    # Remove static variables from SFC list if you load them separately
    static_var_names = {ch["variable"] for ch in STATIC_CHANNELS}
    sfc_vars = [v for v in sfc_vars if v not in static_var_names]

    start = duration.start.strftime("%Y%m%d")
    end = duration.stop.strftime("%Y%m%d")
    paths = _get_sfc_paths(folder, sfc_vars, start, end)

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        ds = ds[sfc_vars]
        ds = _normalize_time(ds)
        ds = _normalize_coords_and_vars(ds)
        if "time" in ds.coords:
            ds = ds.sel(time=duration)

        # TODO: unit fixes (TaiESM example converts precipitation units)
        # Example:
        # if "precipitation" in ds:
        #     ds["precipitation"] = ds["precipitation"] * 86400.0  # kg/m2/s -> mm/day
        return ds

    ds_sfc = _open_monthlies(paths, preprocess=_preprocess)
    return ds_sfc


def _get_static_data(folder: str) -> xr.Dataset:
    """
    Load static layers (oro/slope/aspect) as a Dataset.

    Instructions:
    - If your static layers are already on LR grid, just open them and normalize coords.
    - If you instead inject HR terrain (TReAD TER) later, you may return empty dataset here.
    """
    static_paths = _get_static_paths(folder)
    if not static_paths:
        return xr.Dataset()

    dsets = []
    for key, path in static_paths.items():
        ds = xr.open_dataset(path)
        ds = _normalize_coords_and_vars(ds)
        dsets.append(ds)

    ds_static = xr.merge(dsets) if dsets else xr.Dataset()
    return ds_static


# ---------------------------------------------------------------------
# 5) Crop + regrid + terrain injection (EDIT ME)
# ---------------------------------------------------------------------
def _crop_to_grid(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Crop LR dataset to the reference domain before regridding.

    Instructions:
    - era5.py crops global ERA5 by lat/lon bounds derived from the grid.
    - Implement one of:
        A) lat/lon slice using grid.lat/grid.lon min/max with padding
        B) nearest-to-center (TaiESM sample uses Taiwan center) then window
    """
    # Example (lat/lon):
    # lat_min = float(grid["lat"].min()); lat_max = float(grid["lat"].max())
    # lon_min = float(grid["lon"].min()); lon_max = float(grid["lon"].max())
    # return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return ds


def _inject_hr_terrain_layers(
    ds_lr: xr.Dataset,
    layers_hr: xr.Dataset,
    *,
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Inject high-resolution terrain layers into LR dataset after regridding.

    Instructions (era5.py pattern):
    - `layers_hr` typically contains TER / slope / aspect on the TARGET grid
    - Expand those statics along time dimension to match ds_lr[time_dim]
    - Replace `oro` with TER, and attach slope/aspect
    - Optionally create derived channel `wtp = tp * TER / oro_original`
    """
    if not layers_hr:
        return ds_lr

    out = ds_lr

    # TODO: choose correct variable names in `layers_hr`
    # ter = layers_hr["TER"]  # or layers_hr["oro"]
    # slope = layers_hr["slope"]
    # aspect = layers_hr["aspect"]

    # Example expansion:
    # ter_t = ter.expand_dims({time_dim: out[time_dim]}).transpose(time_dim, ...)
    # out["oro"] = ter_t
    # out["slope"] = slope.expand_dims({time_dim: out[time_dim]})
    # out["aspect"] = aspect.expand_dims({time_dim: out[time_dim]})

    return out


def get_lr_dataset(
    grid: xr.Dataset,
    layers_hr: Optional[xr.Dataset],
    start_date: str,
    end_date: str,
    *,
    scenario: Optional[str] = None,
    chunk_after_regrid: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    High-level entry point: load LR PRS+SFC(+static), crop, regrid, inject terrain.

    Returns
    -------
    (ds_daily, ds_regridded)

    Mirrors era5.py / taiesm100.py:
      1) build duration slice
      2) load PRS + SFC + STATIC
      3) merge
      4) crop to grid extent
      5) regrid to target grid
      6) inject HR terrain layers and derived channels
      7) optional chunking for training speed
    """
    start_dt = pd.to_datetime(str(start_date), format="%Y%m%d")
    end_dt = pd.to_datetime(str(end_date), format="%Y%m%d")
    duration = slice(start_dt, end_dt)

    folder = _get_data_dir(scenario=scenario)

    ds_prs = _get_pressure_level_data(folder, duration, scenario=scenario)
    ds_sfc = _get_surface_data(folder, duration, scenario=scenario)
    ds_static = _get_static_data(folder)

    ds_lr = xr.merge([ds_prs, ds_sfc, ds_static], compat="override")

    ds_crop = _crop_to_grid(ds_lr, grid)

    ds_regrid = regrid_dataset(ds_crop, grid)

    if layers_hr is not None:
        ds_regrid = _inject_hr_terrain_layers(ds_regrid, layers_hr)

    if chunk_after_regrid and "time" in ds_regrid.dims:
        # Adjust spatial dims to your grid (WRF: south_north/west_east; lat/lon; y/x)
        spatial_dims = [
            d for d in ["south_north", "west_east", "y", "x", "lat", "lon"] if d in ds_regrid.dims
        ]
        chunk_spec = {"time": 1}
        for d in spatial_dims:
            chunk_spec[d] = -1
        ds_regrid = ds_regrid.chunk(chunk_spec)

    return ds_crop, ds_regrid


# ---------------------------------------------------------------------
# 6) Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # grid = xr.open_dataset("path/to/reference_grid.nc")
    # layers_hr = xr.open_dataset("path/to/hr_terrain_layers.nc")  # optional
    #
    # ds_daily, ds_regridded = get_lr_dataset(
    #     grid=grid,
    #     layers_hr=layers_hr,
    #     start_date="20100101",
    #     end_date="20101231",
    #     scenario="ssp585",   # optional
    # )
    #
    # print(ds_regridded)
    pass


# -----------------------------------------------------------------
# Sample output
# -----------------------------------------------------------------
# >>> ds_regridded
#
# <xarray.Dataset> Size: 129MB
# Dimensions:                 (time: 31, south_north: 208, west_east: 208, level: 4)
# Coordinates:
#   * time                    (time) datetime64[ns] 248B 2018-01-01T11:00:00 .....
#   * level                   (level) float64 32B 500.0 700.0 850.0 925.0
#     XLAT                    (south_north, west_east) float32 173kB 21.75 ... ...
#     XLONG                   (south_north, west_east) float32 173kB 118.9 ... ...
# Dimensions without coordinates: south_north, west_east
# Data variables:
#     precipitation           (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     temperature_2m          (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     eastward_wind_10m       (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     northward_wind_10m      (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     geopotential_height     (time, level, south_north, west_east) float32 21MB dask.array<chunksize=(1, 1, 12, 208), meta=np.ndarray>
#     temperature             (time, level, south_north, west_east) float32 21MB dask.array<chunksize=(1, 1, 12, 208), meta=np.ndarray>
#     eastward_wind           (time, level, south_north, west_east) float32 21MB dask.array<chunksize=(1, 1, 12, 208), meta=np.ndarray>
#     northward_wind          (time, level, south_north, west_east) float32 21MB dask.array<chunksize=(1, 1, 12, 208), meta=np.ndarray>
#     terrain_height          (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
# Attributes:
#     regrid_method:  bilinear
