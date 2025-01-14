import os
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import re

from util import regrid_dataset, create_and_process_dataarray, is_local_testing

PRESSURE_LEVELS = [500, 700, 850, 925]
ERA5_PRS_CHANNELS = {
    "z": "geopotential_height",
    "t": "temperature",
    "u": "eastward_wind",
    "v": "northward_wind",
}
ERA5_SFC_CHANNELS = {
    "tp" : "precipitation",
    "t2m": "temperature_2m",
    "u10": "eastward_wind_10m",
    "v10": "northward_wind_10m",
    "msl": "mean_sea_level_pressure",
}
ERA5_ADD_CHANNELS = {
    "w850": "850hPa_vertical_velocity",
    "q1000": "10000hPa_specific_humidity",
}
ERA5_ORO_CHANNEL = {
    "oro": "terrain_height"
}
ERA5_CHANNEL_NUM = \
    len(ERA5_PRS_CHANNELS) * len(PRESSURE_LEVELS) + len(ERA5_SFC_CHANNELS) + len(ERA5_ADD_CHANNELS) + len(ERA5_ORO_CHANNEL)

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

def read_add_vars(folder, subfolder, add_variables, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y%m").tolist()

    addvar = []
    for var in add_variables:
        var_name = re.match(r"[a-zA-Z]+", var).group(0)
        level = int(re.search(r"\d+", var).group(0))

        paths = [
            os.path.join(
                folder if is_local_testing() else os.path.join(folder, "PRS", subfolder, var_name, yyyymm[:4]),
                f"ERA5_PRS_{var_name}_{yyyymm}_r1440x721_day.nc"
            )
            for yyyymm in date_range
        ]

        ds = xr.open_mfdataset(paths, combine="by_coords").sel(level=level)
        ds = ds.drop_vars("level").rename({var_name: var})
        addvar.append(ds)

    return xr.merge(addvar)

def get_era5_dataset(dir, grid, start_date, end_date):
    pressure_level_vars = list(ERA5_PRS_CHANNELS.keys())
    surface_vars = list(ERA5_SFC_CHANNELS.keys())
    other_vars = list(ERA5_ADD_CHANNELS.keys())

    # pressure_level
    duration = slice(str(start_date), str(end_date))
    prs_paths = get_prs_paths(dir, "day", pressure_level_vars, start_date, end_date)
    era5_prs = xr.open_mfdataset(prs_paths, combine='by_coords').sel(level=PRESSURE_LEVELS, time=duration)

    # surface
    sfc_paths = get_sfc_paths(dir, "day", surface_vars, start_date, end_date)
    era5_sfc = xr.open_mfdataset(sfc_paths, combine='by_coords').sel(time=duration)
    era5_sfc['tp'] = era5_sfc['tp'] * 24 * 1000
    era5_sfc['tp'].attrs['units'] = 'mm/day' # Convert unit

    # other_variables 
    era5_add = read_add_vars(dir, "day", other_vars, start_date, end_date)

    # orography
    era5_topo = xr.open_mfdataset(dir + "/ERA5_oro_r1440x721.nc")[['oro']]
    era5_topo = era5_topo.expand_dims(time=era5_sfc.time)
    era5_topo = era5_topo.reindex(time=era5_sfc.time)

    # Merge prs, sfc, topo and rename variables.
    era5 = xr.merge([era5_prs, era5_sfc, era5_add, era5_topo]).rename({
        **ERA5_PRS_CHANNELS,
        **ERA5_SFC_CHANNELS,
        **ERA5_ADD_CHANNELS,
        **ERA5_ORO_CHANNEL
    })

    # Crop to Taiwan domain given ERA5 is global data.
    lat, lon = grid.XLAT, grid.XLONG
    era5_crop = era5.sel(
        latitude=slice(lat.max().item(), lat.min().item()),
        longitude=slice(lon.min().item(), lon.max().item()))

    # Based on REF grid, regrid TReAD data over spatial dimensions for all timestamps.
    era5_out = regrid_dataset(era5_crop, grid)

    return era5_crop, era5_out

def get_era5_pressure(era5_channel):
    era5_pressure_values = np.array(
        [np.nan] +  # First surface channel ("precipitation")
        list(np.repeat(PRESSURE_LEVELS, len(ERA5_PRS_CHANNELS))) +  # PRS_CHANNELS repeated for each pressure level
        [np.nan] * len(ERA5_SFC_CHANNELS) +  # Remaining surface channels + Orography channel
        [np.nan] * len(ERA5_ADD_CHANNELS)    # Add variables channel
    )
    era5_pressure = xr.DataArray(
        da.from_array(era5_pressure_values, chunks=(len(era5_pressure_values))),
        dims=["era5_channel"],
        coords={ "era5_channel": era5_channel },
        name="era5_pressure"
    )

    return era5_pressure, era5_pressure_values

def get_era5_variable(era5_channel):
    era5_variables_values = (
        list(ERA5_SFC_CHANNELS.values())[:1] +  # First surface channel ("precipitation")
        list(ERA5_PRS_CHANNELS.values()) * len(PRESSURE_LEVELS) +  # PRS_CHANNELS repeated for each pressure level
        list(ERA5_SFC_CHANNELS.values())[1:] +  # Remaining surface channels
        list(ERA5_ADD_CHANNELS.values()) +      # Other Variables channels 
        list(ERA5_ORO_CHANNEL.values())         # Orography channel
    )
    era5_variable = xr.DataArray(
        data=da.from_array(era5_variables_values, chunks=(len(era5_variables_values))),
        dims=["era5_channel"],
        coords={ "era5_channel": era5_channel },
        name="era5_variable"
    )

    return era5_variable, era5_variables_values

def get_era5(era5_out):
    era5_channel = np.arange(ERA5_CHANNEL_NUM)
    era5_pressure, era5_pressure_values = get_era5_pressure(era5_channel)
    era5_variable, era5_variables_values = get_era5_variable(era5_channel)

    # Create ERA5 DataArray
    stack_era5 = da.stack(
        [
            era5_out[var].sel(level=plev).data if "level" in era5_out[var].dims else era5_out[var].data
            for var, plev in zip(era5_variables_values, era5_pressure_values)
        ],
        axis=1
    )
    era5_dims = ["time", "era5_channel", "south_north", "west_east"]
    era5_coords = {
        "time": era5_out["time"],
        "era5_channel": era5_channel,
        "south_north": era5_out["south_north"],
        "west_east": era5_out["west_east"],
        "XLAT": era5_out["XLAT"],
        "XLONG": era5_out["XLONG"],
        "era5_pressure": era5_pressure,
        "era5_variable": era5_variable,
    }
    era5_chunk_sizes = {
        "time": 1,
        "era5_channel": era5_channel.size,
        "south_north": era5_out["south_north"].size,
        "west_east": era5_out["west_east"].size,
    }

    return create_and_process_dataarray("era5", stack_era5, era5_dims, era5_coords, era5_chunk_sizes)

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
    print(f"\nERA5 dataset =>\n {era5_out}")

    # Generate output fields
    era5 = get_era5(era5_out)
    era5_center = get_era5_center(era5)
    era5_scale = get_era5_scale(era5)
    era5_valid = get_era5_valid(era5)

    return era5, era5_center, era5_scale, era5_valid, era5_pre_regrid, era5_out
