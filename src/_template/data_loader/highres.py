"""
CorrDiff high-resolution data loader (TEMPLATE).

This file is a *template* you can copy and adapt for your CorrDiff HR branch.
It is based on the attached examples:
- tread.py: TReAD HR surface loader with daily aggregation + derived vars + regridding
- taiesm3p5.py: TaiESM 3.5km HR loader with monthly files + WRF Times parsing + regridding

What you should customize:
1) DATASET_NAME / get_data_dir(): your storage layout (local vs server)
2) CHANNELS_ORIGINAL / CHANNELS: raw -> CorrDiff variable mapping
3) FILE_PATTERN in get_file_paths(): monthly naming convention
4) preprocess() inside open_mfdataset(): how to parse time & subset variables
5) derived variables / daily aggregation logic (if needed)
6) spatial dims names (e.g., south_north/west_east) if different
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
#
from .util import regrid_dataset


# ---------------------------------------------------------------------
# 1) Dataset naming + channel mapping (EDIT ME)
# ---------------------------------------------------------------------
DATASET_NAME = "YOUR_HR_DATASET"  # e.g., "TReAD" or "TaiESM3p5"

# TODO: raw variable -> CorrDiff channel name
# (Keep CorrDiff names consistent with your model’s expected channel list.)
CHANNELS_ORIGINAL: Dict[str, str] = {
    # Examples:
    # "T2": "temperature_2m",
    # "U10": "eastward_wind_10m",
    # "V10": "northward_wind_10m",
}

# If you compute derived variables (e.g., precipitation accumulation),
# include them in CHANNELS as well.
CHANNELS: Dict[str, str] = {
    # Example: "TP": "precipitation",
    **CHANNELS_ORIGINAL,
}


def get_channels() -> Dict[str, str]:
    """Return CorrDiff channel mapping for this HR dataset."""
    return CHANNELS


# ---------------------------------------------------------------------
# 2) Paths + file list builder (EDIT ME)
# ---------------------------------------------------------------------
def _get_data_dir(*, scenario: Optional[str] = None) -> str:
    """
    Return base directory for this HR dataset.

    Instructions:
    - Match the approach from tread.py / taiesm3p5.py:
        local testing -> ../data/<something>
        BIG server    -> /lfs/.../<something>
    - If your HR dataset depends on scenario (ssp245/ssp585/etc),
      accept it via `scenario` and embed it in the server path.
    """
    # TODO: return base directory for this HR dataset
    # Example pattern:
    # return "../data/your_hr" if is_local_testing() else f"/lfs/archive/.../{scenario}"
    raise NotImplementedError("Implement get_data_dir() for your environment")


def _get_file_paths(folder: str, start_date: str, end_date: str) -> List[Path]:
    """
    Build the list of *monthly* NetCDF file paths covering [start_date, end_date].

    start_date/end_date format: 'YYYYMMDD'

    Instructions:
    - Follow tread.py / taiesm3p5.py:
        date_range = pd.date_range(..., freq="MS").strftime("%Y%m")
        return [folder / f"<pattern>_{yyyymm}.nc" for yyyymm in date_range]
    - Make sure your pattern matches the actual filenames on disk.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    folder_path = Path(folder)

    # TODO: replace FILE_PATTERN with your dataset naming convention.
    # Examples:
    # - f"wrfo2D_d02_{yyyymm}.nc"
    # - f"TaiESM1-WRF_tw3.5_{scenario}_wrfday_d01_{yyyymm}.nc"
    FILE_PATTERN = "YOUR_FILE_PREFIX_{yyyymm}.nc"

    return [folder_path / FILE_PATTERN.format(yyyymm=yyyymm) for yyyymm in date_range]


# ---------------------------------------------------------------------
# 3) Time parsing helpers (EDIT IF NEEDED)
# ---------------------------------------------------------------------
def _parse_wrf_times_to_datetime(times: xr.DataArray) -> pd.DatetimeIndex:
    """
    Convert common WRF-style 'Times' or 'Time' strings into pandas datetime.

    Instructions:
    - tread.py uses ds['Time'] values like 'YYYY-mm-dd_HH:MM:SS'
    - taiesm3p5.py uses ds['Times'] with same formatting
    - Adjust `format=` if your timestamps differ.
    """
    return pd.to_datetime(times.astype(str), format="%Y-%m-%d_%H:%M:%S")


# ---------------------------------------------------------------------
# 4) Core loader: open_mfdataset + optional daily aggregation + regrid
# ---------------------------------------------------------------------
def get_hr_dataset(
    grid: xr.Dataset,
    start_date: str,    # 'YYYYMMDD'
    end_date: str,      # 'YYYYMMDD'
    *,
    scenario: Optional[str] = None,
    daily: bool = True,
    chunk_after_regrid: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    High-level entry point for CorrDiff HR data.

    Returns
    -------
    (ds_raw_or_daily, ds_regridded)

    Notes
    -----
    - Mirrors the pattern in tread.py / taiesm3p5.py:
        1) build file list
        2) xr.open_mfdataset(..., preprocess=...)
        3) (optional) resample to daily + derived variables
        4) rename to CorrDiff channel names
        5) regrid to reference grid
        6) optional chunking for training performance
    """
    start_dt = pd.to_datetime(str(start_date), format="%Y%m%d")
    end_dt = pd.to_datetime(str(end_date), format="%Y%m%d")

    # -----------------------------------------------------------------
    # 4.1 Select which raw variables to read (EDIT ME)
    # -----------------------------------------------------------------
    raw_vars = list(CHANNELS_ORIGINAL.keys())

    # If you need extra raw vars to compute derived channels,
    # add them here (e.g., RAINC/RAINNC for TP).
    #
    # Example (TReAD-style):
    # extra_vars = ["RAINC", "RAINNC"]
    # raw_vars = extra_vars + raw_vars

    # TODO:
    extra_vars: List[str] = []
    raw_vars = extra_vars + raw_vars

    # -----------------------------------------------------------------
    # 4.2 Define preprocess() for each monthly file (EDIT ME)
    # -----------------------------------------------------------------
    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        """
        Per-file preprocessing before concatenation.

        Instructions:
        - Select only needed variables: ds[raw_vars]
        - Attach a unified time coordinate:
            * If ds has 'Times' -> parse it
            * If ds has 'Time'  -> parse it
        - Subset time window: .sel(time=slice(start_dt, end_dt))
        - Unify dimension naming if needed (e.g., rename({'Time':'time'}))
        """
        ds = ds[raw_vars]

        # Detect common WRF time variables.
        if "Times" in ds.variables:
            time_index = _parse_wrf_times_to_datetime(ds["Times"])
            ds = ds.assign_coords(time=time_index)
        elif "Time" in ds.variables:
            time_index = _parse_wrf_times_to_datetime(ds["Time"])
            ds = ds.assign_coords(time=time_index)
        else:
            # If your files already have a proper `time` coordinate, do nothing.
            # Otherwise, implement parsing here.
            pass

        # If your dataset uses a different time dimension name, rename it to "time".
        # Example from taiesm3p5.py:
        # ds = ds.rename({"Time": "time"})
        if "Time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"Time": "time"})

        # Subset requested time window
        if "time" in ds.coords:
            ds = ds.sel(time=slice(start_dt, end_dt))

        return ds

    # -----------------------------------------------------------------
    # 4.3 Load monthly files (open_mfdataset)
    # -----------------------------------------------------------------
    folder = _get_data_dir(scenario=scenario)
    file_list = _get_file_paths(folder, start_date, end_date)

    ds_hr = xr.open_mfdataset(
        file_list,
        preprocess=_preprocess,
        combine="by_coords",
        # engine="netcdf4",  # uncomment if needed
        # parallel=True,     # if dask available + desired
    )

    # -----------------------------------------------------------------
    # 4.4 Optional daily aggregation + derived channels (EDIT ME)
    # -----------------------------------------------------------------
    if daily:
        ds_hr = _aggregate_to_daily(ds_hr)

    # -----------------------------------------------------------------
    # 4.5 Keep only CHANNELS keys, then rename to CorrDiff names
    # -----------------------------------------------------------------
    # If you created derived vars, ensure they exist before selecting.
    ds_hr = ds_hr[list(CHANNELS.keys())].rename(CHANNELS)

    # -----------------------------------------------------------------
    # 4.6 Regrid to reference grid
    # -----------------------------------------------------------------
    ds_out = regrid_dataset(ds_hr, grid)

    # Optional: chunking like taiesm3p5.py (good for training throughput)
    if chunk_after_regrid and "time" in ds_out.dims:
        # Adjust spatial dims to your dataset ("south_north"/"west_east" are WRF defaults)
        spatial_dims = [
            d for d in ["south_north", "west_east", "y", "x", "lat", "lon"] if d in ds_out.dims
        ]
        chunk_spec = {"time": 1}
        for d in spatial_dims:
            chunk_spec[d] = -1
        ds_out = ds_out.chunk(chunk_spec)

    return ds_hr, ds_out


def _aggregate_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert sub-daily WRF outputs into daily CorrDiff-ready variables.

    Instructions:
    - If your HR files are already daily, just return ds unchanged.
    - If you need daily means (T2/U10/V10), do:
        ds[vars].resample(time="1D").mean()
    - If you need daily precipitation accumulation, do:
        (RAINC + RAINNC).resample(time="1D").sum()
    - If you need daily max/min temperature, do:
        ds["T2"].resample(time="1D").max() / min()

    Use tread.py as the reference pattern.
    """
    if "time" not in ds.dims:
        return ds

    # TODO: decide whether you need daily mean for CHANNELS_ORIGINAL keys
    base_keys = list(CHANNELS_ORIGINAL.keys())
    ds_daily = ds[base_keys].resample(time="1D").mean()

    # TODO: derived variables examples (uncomment/adapt as needed)
    # if "RAINC" in ds and "RAINNC" in ds:
    #     ds_daily["TP"] = (ds["RAINC"] + ds["RAINNC"]).resample(time="1D").sum()
    #
    # if "T2" in ds:
    #     ds_daily["T2MAX"] = ds["T2"].resample(time="1D").max()
    #     ds_daily["T2MIN"] = ds["T2"].resample(time="1D").min()

    return ds_daily


# ---------------------------------------------------------------------
# 5) Example usage (for your reference)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # grid = xr.open_dataset("path/to/reference_grid.nc")
    # ds_daily, ds_regridded = get_hr_dataset(
    #     grid=grid,
    #     start_date="20100101",
    #     end_date="20101231",
    #     scenario="ssp585",
    #     daily=True,
    # )
    #
    # print(ds_regridded)
    pass


# -----------------------------------------------------------------
# Sample output
# -----------------------------------------------------------------
# >>> ds_regridded
#
# <xarray.Dataset> Size: 22MB
# Dimensions:             (time: 31, south_north: 208, west_east: 208)
# Coordinates:
#   * time                (time) datetime64[ns] 248B 2018-01-01 ... 2018-01-31
#     XLAT                (south_north, west_east) float32 173kB 21.75 ... 25.57
#     XLONG               (south_north, west_east) float32 173kB 118.9 ... 123.1
# Dimensions without coordinates: south_north, west_east
# Data variables:
#     precipitation       (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     temperature_2m      (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     eastward_wind_10m   (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
#     northward_wind_10m  (time, south_north, west_east) float32 5MB dask.array<chunksize=(1, 208, 208), meta=np.ndarray>
