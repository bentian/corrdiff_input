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
VARS = [
    {'name': 'XLAT', 'unit': 'degrees_north', 'required': True },
    {'name': 'XLONG', 'unit': 'degrees_east', 'required': True },
    {'name': 'LANDMASK', 'unit': 'land mask', 'required': True },
    {'name': 'TER', 'unit': 'meters', 'required': False },
    {'name': 'SLOPE', 'unit': 'slope', 'required': False },
    {'name': 'ASPECT', 'unit': 'degree', 'required': False },
]

# -----------------------------------------------------------
# Helper function
# -----------------------------------------------------------
def load_var(nc, spec):
    """Load a variable; error if required and missing, else return None."""
    name = spec["name"]
    if name in nc.variables:
        return nc.variables[name][:]
    if spec["required"]:
        raise KeyError(f"Required variable '{name}' not found in input file.")
    return None

# -------------------------------------------------------------------
# Load variables
# -------------------------------------------------------------------
nc_in = Dataset(INPUT_FILE, mode='r')
var_data = {spec["name"]: load_var(nc_in, spec) for spec in VARS}
spec_by_name = {spec["name"]: spec for spec in VARS}

lat = var_data["XLAT"]
lon = var_data["XLONG"]

# All non-coordinate fields (LANDMASK + optional ones)
field_names = [v["name"] for v in VARS if v["name"] not in ("XLAT", "XLONG")]
field_data = {name: var_data[name] for name in field_names}

# Report detected (non-coordinate) fields
print(f"Input file:\n  {INPUT_FILE}")
print("Detected variables:")
for name in field_names:
    status = "FOUND" if field_data[name] is not None else "MISSING"
    print(f"  {name} - {status}")
print()

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
    for name, arr in field_data.items():
        if arr is not None:
            field_data[name] = regridder(xr.DataArray(arr))
else:
    print("Cropping to smaller grid...")

    # Find center indices
    idy, idx = np.abs(lat[:, 0] - clat).argmin(), np.abs(lon[0, :] - clon).argmin()

    # Calculate slicing indices
    slat, elat = max(0, idy - ny // 2), min(lat.shape[0], idy + ny // 2)
    slon, elon = max(0, idx - nx // 2), min(lon.shape[1], idx + nx // 2)

    # Crop the grid
    lat_grid, lon_grid = lat[slat:elat, slon:elon], lon[slat:elat, slon:elon]

    # Regrid all existing fields
    for name, arr in field_data.items():
        if arr is not None:
            field_data[name] = arr[slat:elat, slon:elon]

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

    def write_var(name, data):
        unit = spec_by_name[name]["unit"]
        var = ncfile.createVariable(name, "f4", ("south_north", "west_east"))
        var[:, :] = data
        var.units = unit

    # Coordinates
    write_var("XLAT", lat_grid)
    write_var("XLONG", lon_grid)

    # All other fields, only if present
    for name, arr in field_data.items():
        if arr is not None:
            write_var(name, arr)

    ncfile.setncattr("coordinates", "XLAT XLONG")
    ncfile.setncattr("description", f"CorrDiff REF grid {ny}x{nx}")

print(f"Output written to => {OUTPUT_FILE}")
