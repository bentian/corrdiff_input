import os
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from util import regrid_dataset, is_local_testing

pressure_levels = [500, 700, 850, 925]

def get_prs_paths(folder, subfolder, variables, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    if is_local_testing():
        return [
            os.path.join(folder, f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc")
            for var in variables for yyyymm in date_range
        ]

    return [
        os.path.join(
            folder, "PRS", subfolder, var, yyyymm[:4],
            f"ERA5_PRS_{var}_{yyyymm}_r1440x721_day.nc"
        )
        for var in variables for yyyymm in date_range
    ]

def get_sfc_paths(folder, subfolder, variables, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()
    if is_local_testing():
        return [
            os.path.join(folder, f"ERA5_SFC_{var}_201801_r1440x721_day.nc")
            for var in variables
        ]

    return [
        os.path.join(
            folder, "SFC", subfolder, var, yyyymm[:4],
            f"ERA5_SFC_{var}_{yyyymm}_r1440x721_day.nc"
        )
        for var in variables for yyyymm in date_range
    ]

def get_era5_dataset(dir, grid, start_date, end_date):
    pressure_level_vars = ['z', 't', 'u', 'v']
    surface_vars = ['tp', 't2m', 'u10', 'v10']

    # pressure_level
    duration = slice(str(start_date), str(end_date))
    prs_paths = get_prs_paths(dir, "day", pressure_level_vars, start_date, end_date)
    era5_prs = xr.open_mfdataset(prs_paths, combine='by_coords').sel(level=pressure_levels, time=duration)

    # surface
    sfc_paths = get_sfc_paths(dir, "day", surface_vars, start_date, end_date)
    era5_sfc = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day' # Convert unit

    # orography
    era5_topo = xr.open_mfdataset(dir + "/ERA5_oro_r1440x721.nc")[['oro']]
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo and rename variables.
    era5 = xr.merge([era5_prs, era5_sfc, era5_topo]).rename({
        "z": "geopotential_height",
        "t": "temperature",
        "u": "eastward_wind",
        "v": "northward_wind",
        "tp" : "precipitation",
        "t2m": "temperature_2m",
        "u10": "eastward_wind_10m",
        "v10": "northward_wind_10m",
        "oro": "terrain_height"
    })

    # Crop to Taiwan domain given ERA5 is global data.
    lat, lon = grid.XLAT, grid.XLONG
    era5_crop = era5.sel(
        latitude=slice(lat.max().item(), lat.min().item()),
        longitude=slice(lon.min().item(), lon.max().item()))

    # Based on REF grid, regrid TReAD data over spatial dimensions for all timestamps.
    era5_out = regrid_dataset(era5_crop, grid)

    return era5_crop, era5_out

def get_era5(era5_out, stack_era5, era5_channel, era5_pressure_values, era5_variables_values):
    era5_pressure = xr.DataArray(
        da.from_array(era5_pressure_values, chunks=(len(era5_pressure_values))),
        dims=["era5_channel"],
        coords={"era5_channel": era5_channel},
        name="era5_pressure"
    )
    era5_variable = xr.DataArray(
        data=da.from_array(era5_variables_values, chunks=(len(era5_variables_values))),
        dims=["era5_channel"],
        coords={"era5_channel": era5_channel},
        name="era5_variable"
    )

    return xr.DataArray(
        stack_era5,
        dims=["time", "era5_channel", "south_north", "west_east"],
        coords={
            "time": era5_out["time"],
            "era5_channel": era5_channel,
            "south_north": era5_out["south_north"],
            "west_east": era5_out["west_east"],
            "XLAT": era5_out["XLAT"],
            "XLONG": era5_out["XLONG"],
            "era5_pressure": era5_pressure,
            "era5_variable": era5_variable,
        },
        name="era5"
    )

def get_era5_center(era5):
    era5_mean = da.stack(
        [
            era5.isel(era5_channel=channel).mean(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )

    return xr.DataArray(
        era5_mean,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_center"
    )

def get_era5_scale(era5):
    era5_std = da.stack(
        [
            era5.isel(era5_channel=channel).std(dim=["time", "south_north", "west_east"]).data
            for channel in era5["era5_channel"].values
        ],
        axis=0
    )
    return xr.DataArray(
        era5_std,
        dims=["era5_channel"],
        coords={
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_scale"
    )

def get_era5_valid(era5):
    valid = True
    return xr.DataArray(
        data=da.from_array(
                [[valid] * len(era5["era5_channel"])] * len(era5["time"]),
                chunks=(len(era5["time"]), len(era5["era5_channel"]))
            ),
        dims=["time", "era5_channel"],
        coords={
            "time": era5["time"],
            "era5_channel": era5["era5_channel"],
            "era5_pressure": era5["era5_pressure"],
            "era5_variable": era5["era5_variable"]
        },
        name="era5_valid"
    )

def generate_era5_output(dir, grid, start_date, end_date):
    # Extract ERA5 data from file.
    era5_pre_regrid, era5_out = get_era5_dataset(dir, grid, start_date, end_date)

    ## Prep for generation

    era5_channel = np.arange(21)
    era5_pressure_values = np.array(
        [np.nan] + list(np.repeat(pressure_levels, 4)) + [np.nan] * 4
    )
    era5_variables_values = (
        ['precipitation'] + ['geopotential_height', 'temperature', 'eastward_wind', 'northward_wind'] * 4 +
        ['temperature_2m', 'eastward_wind_10m', 'northward_wind_10m', 'terrain_height']
    )

    stack_era5 = da.stack(
        [
            era5_out[var].sel(level=plev).data if "level" in era5_out[var].dims else era5_out[var].data
            for var, plev in zip(era5_variables_values, era5_pressure_values)
        ],
        axis=1
    )

    ## Generate output fields

    era5 = get_era5(era5_out, stack_era5, era5_channel, era5_pressure_values, era5_variables_values)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_out