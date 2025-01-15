"""
Regrid Dataset and Save to NetCDF.

This script takes an input NetCDF file containing geospatial grid data, extrapolates
the data to a larger grid size using bilinear interpolation, and saves the regridded
data to a new NetCDF file. It leverages the `xESMF` library for regridding and supports
nearest-neighbor extrapolation for points outside the source grid.

Features:
- Generates a larger grid with specified dimensions (`ny` x `nx`).
- Uses bilinear interpolation for regridding and nearest-neighbor extrapolation.
- Processes geospatial variables (`TER`, `LANDMASK`) to align with the new grid.
- Saves the regridded dataset, including metadata, into a NetCDF file.

Parameters:
- `clon`, `clat` (float): Center longitude and latitude of the grid.
- `ny`, `nx` (int): Dimensions of the new larger grid to be generated.
- `INPUT_FILE` (str): Path to the input NetCDF file containing source data.
- `OUTPUT_FILE` (str): Path to the output NetCDF file with the regridded data.

Workflow:
1. Open the input NetCDF file and extract geospatial variables.
2. Define the bounds and generate a new larger grid.
3. Perform regridding using `xESMF`, including extrapolation with the "nearest_s2d" method.
4. Save the regridded data, along with the new grid, to a NetCDF file.

Dependencies:
- `os`: For file handling and path operations.
- `numpy`: For numerical operations and grid generation.
- `xarray`: For handling labeled multidimensional arrays.
- `xesmf`: For regridding and extrapolation of geospatial data.

Example Usage:
    1. Update the `clon`, `clat`, `ny`, `nx`, and `INPUT_FILE` parameters.
    2. Run the script to generate a regridded output file.
    3. The output will be saved in the specified `OUTPUT_FILE`.

Notes:
- Ensure that the input file (`INPUT_FILE`) exists and contains the required variables:
  `XLAT`, `XLONG`, `TER`, and `LANDMASK`.
- The script overwrites any existing output file at the specified `OUTPUT_FILE` path.
"""
import os
import numpy as np
import xarray as xr
import xesmf as xe

# === Parameters ===
clon, clat = 120.9465, 23.6745  # Center latitude and longitude
ny, nx = 288, 224               # New larger grid dimensions

# === Input file ===
INPUT_FILE = './TReAD_wrf_d02_info.nc'
nc_in = xr.open_dataset(INPUT_FILE)

# Extract variables from the input dataset
lat = nc_in["XLAT"]
lon = nc_in["XLONG"]
ter = nc_in["TER"]
lmask = nc_in["LANDMASK"]

# === Generate New Grid ===
lat_min, lat_max = lat.min().item(), lat.max().item()
lon_min, lon_max = lon.min().item(), lon.max().item()

new_lat = np.linspace(lat_min, lat_max, ny)
new_lon = np.linspace(lon_min, lon_max, nx)
new_lat_grid, new_lon_grid = np.meshgrid(new_lat, new_lon, indexing='ij')

# Create new grid as xarray.Dataset
new_grid = xr.Dataset(
    {
        "lat": (["south_north", "west_east"], new_lat_grid),
        "lon": (["south_north", "west_east"], new_lon_grid),
    }
)

# === Regrid Using xESMF ===
regridder = xe.Regridder(
    nc_in,
    new_grid,
    method="bilinear",  # Bilinear interpolation
    extrap_method="nearest_s2d",  # Nearest neighbor extrapolation
    periodic=False,
)

# Regrid each variable
ter_regridded = regridder(nc_in["TER"])
lmask_regridded = regridder(nc_in["LANDMASK"])

# === Output file ===
OUTPUT_FILE = f"./wrf_{ny}x{nx}_grid_coords.nc"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

with xr.Dataset() as nc_out:
    # Assign regridded variables
    nc_out["XLAT"] = (["south_north", "west_east"], new_lat_grid)
    nc_out["XLONG"] = (["south_north", "west_east"], new_lon_grid)
    nc_out["TER"] = ter_regridded
    nc_out["LANDMASK"] = lmask_regridded

    # Add metadata
    nc_out["XLAT"].attrs["units"] = "degrees_north"
    nc_out["XLONG"].attrs["units"] = "degrees_east"
    nc_out["TER"].attrs["units"] = "meters"
    nc_out["LANDMASK"].attrs["units"] = "land mask"
    nc_out.attrs["coordinates"] = "XLAT XLONG"
    nc_out.attrs["description"] = f"New CorrDiff Training REF grid {ny}x{nx}"
    print(nc_out)

    # Save to NetCDF
    nc_out.to_netcdf(OUTPUT_FILE)

print(f"Output written to => {OUTPUT_FILE}")
