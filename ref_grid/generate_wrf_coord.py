"""
NetCDF Grid Extraction and Resampling Tool
==========================================

This script extracts a subgrid or resamples a WRF-style NetCDF coordinate file
(e.g., `XLAT`, `XLONG`, `LANDMASK`, terrain fields) to a new grid dimension
centered at a specified latitude and longitude. It supports both downscaling
(cropping) and upscaling (bilinear regridding with xESMF), while preserving
coordinate structure and available metadata fields.

Features
--------
- **Flexible Variable Handling**:
  Variables are defined using a unified `VARS` specification that lists
  variable names, units, and whether each field must be present.
  Missing optional fields (e.g., `TER`, `SLOPE`, `ASPECT`) are handled gracefully.

- **Cropping (Downscaling)**:
  If the requested grid size `(ny, nx)` is smaller than the input grid,
  the script cuts out a subregion centered on (`clon`, `clat`).

- **Regridding (Upscaling)**:
  If the target grid is larger than the original, the script applies
  bilinear interpolation via **xESMF**, using `nearest_s2d` extrapolation
  for edge points.

- **Metadata Preservation**:
  Units are taken from the `VARS` table; output variables are written
  only if they exist in the input.

- **Automatic File Handling**:
  Existing output files are replaced automatically.

Inputs
------
- `INPUT_FILE`: Path to source NetCDF file containing coordinate grids.
- `VARS`: Specification of all variables to extract/crop/regrid.
- `clon`, `clat`: Geographic center for cropping.
- `ny`, `nx`: Desired output grid size.

Outputs
-------
- `OUTPUT_FILE`: A new NetCDF file containing:
    - `XLAT`, `XLONG` grids (cropped or regridded)
    - All available fields defined in `VARS`
    - Attributes describing coordinates and grid size

Dependencies
------------
- numpy
- netCDF4
- xarray
- xESMF
- pathlib

Usage
-----
1. Edit the parameters (`clon`, `clat`, `ny`, `nx`, INPUT_FILE`).
2. Run the script:
       python generate_wrf_coord.py
3. Output file will be written to:
       ./wrf_{ny}x{nx}_grid_coords.nc

Notes
-----
- If a required variable (e.g., XLAT/XLONG) is missing, an exception is raised.
- Optional variables are skipped if absent.
- Works for any WRF-like horizontal grid structure.
"""
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import xesmf as xe
import xarray as xr

# === Parameters ===
clon, clat = 120.9465, 23.6745  # Center latitude and longitude
ny, nx = 304, 304               # Desired grid dimensions

# === Input / Output ===
INPUT_FILE = './TAIESM_tw3.5km_coord2d.nc'
OUTPUT_FILE = f"./wrf_{ny}x{nx}_grid_coords.nc"
VARS = {
    'XLAT':     {'unit': 'degrees_north',   'required': True },
    'XLONG':    {'unit': 'degrees_east',    'required': True },
    'LANDMASK': {'unit': 'land mask',       'required': True },
    'TER':      {'unit': 'meters',          'required': False },
    'SLOPE':    {'unit': 'slope',           'required': False },
    'ASPECT':   {'unit': 'degree',          'required': False },
}

# -----------------------------------------------------------
# Helper function
# -----------------------------------------------------------
def load_var(nc: xr.Dataset, var_name: str):
    """Load a variable; error if required and missing, else return None."""
    spec = VARS[var_name]
    if var_name in nc.variables:
        return nc.variables[var_name][:]
    if spec["required"]:
        raise KeyError(f"Required variable '{var_name}' not found in input file.")
    return None

def write_var(var_name: str, data: xr.Dataset):
    """Create and write a 2D variable to the NetCDF output file."""
    unit = VARS[var_name]["unit"]
    var = ncfile.createVariable(var_name, "f4", ("south_north", "west_east"))
    var[:, :] = data
    var.units = unit

# -------------------------------------------------------------------
# Load variables
# -------------------------------------------------------------------
nc_in = Dataset(INPUT_FILE, mode='r')

# Load all variables defined in VARS
var_data = { name: load_var(nc_in, name) for name in VARS }
lat, lon = var_data["XLAT"], var_data["XLONG"]
layer_data = { name: var_data[name] for name in VARS if name not in ("XLAT", "XLONG") }

# Report found (non-coordinate) layers
found_layers = [name for name, arr in layer_data.items() if arr is not None]
print(f"Input file: {INPUT_FILE}\nFound layers: {found_layers}\n")

# -------------------------------------------------------------------
# Crop or regrid
# -------------------------------------------------------------------
print(f"Input grid (lat, lon) = ({lat.shape[0]}, {lon.shape[1]})")
need_regrid = ny > lat.shape[0] or nx > lon.shape[1]

if need_regrid:
    print("Extrapolating to larger grid...")

    # Create new lat/lon grid
    new_lat = np.linspace(lat.min(), lat.max(), ny)
    new_lon = np.linspace(lon.min(), lon.max(), nx)
    new_grid = xr.Dataset({
        "lat": (["south_north", "west_east"], np.meshgrid(new_lat, new_lon, indexing='ij')[0]),
        "lon": (["south_north", "west_east"], np.meshgrid(new_lat, new_lon, indexing='ij')[1])
    })

    # Use xESMF for extrapolation
    regridder = xe.Regridder(
        xr.Dataset({"lat": (["south_north", "west_east"], lat),
                    "lon": (["south_north", "west_east"], lon)}),
        new_grid, method="bilinear", extrap_method="nearest_s2d"
    )

    lat_grid, lon_grid = new_grid["lat"].values, new_grid["lon"].values

    # Regrid all existing fields
    for name, arr in layer_data.items():
        if arr is not None:
            layer_data[name] = regridder(xr.DataArray(arr))
else:
    print("Cropping to smaller grid ...")

    # Find center indices
    idy, idx = np.abs(lat[:, 0] - clat).argmin(), np.abs(lon[0, :] - clon).argmin()

    # Calculate slicing indices
    slat, elat = max(0, idy - ny // 2), min(lat.shape[0], idy + ny // 2)
    slon, elon = max(0, idx - nx // 2), min(lon.shape[1], idx + nx // 2)
    print(f'  slice (lat, lon) = [{slat}:{elat}, {slon}:{elon}]')

    # Crop the grid
    lat_grid, lon_grid = lat[slat:elat, slon:elon], lon[slat:elat, slon:elon]

    # Regrid all existing fields
    for name, arr in layer_data.items():
        if arr is not None:
            layer_data[name] = arr[slat:elat, slon:elon]

# -------------------------------------------------------------------
# Save to output file
# -------------------------------------------------------------------
output_path = Path(OUTPUT_FILE)
if output_path.exists():
    output_path.unlink(missing_ok=True)

with Dataset(OUTPUT_FILE, mode="w", format="NETCDF4") as ncfile:
    # Create dimensions
    ncfile.createDimension("south_north", lat_grid.shape[0])
    ncfile.createDimension("west_east", lon_grid.shape[1])

    # Coordinates
    write_var("XLAT", lat_grid)
    write_var("XLONG", lon_grid)

    # All other fields, only if present
    for name, arr in layer_data.items():
        if arr is not None:
            write_var(name, arr)

    ncfile.setncattr("coordinates", "XLAT XLONG")
    ncfile.setncattr("description", f"CorrDiff REF grid {ny}x{nx}")

print(f"Output written to => {OUTPUT_FILE}")
