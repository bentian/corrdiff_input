"""
NetCDF Grid Extraction and Resampling Tool
==========================================

This script extracts a subgrid or resamples a WRF-style NetCDF coordinate file
(e.g., `XLAT`, `XLONG`, `LANDMASK`, terrain fields) to a new grid dimension
centered at a specified latitude and longitude. It supports both downscaling
(cropping) and upscaling (bilinear regridding with xESMF), while preserving
coordinate structure and available metadata fields.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from netCDF4 import Dataset
import xarray as xr
import xesmf as xe

# === Default parameters ===
DEFAULT_CLAT, DEFAULT_CLON = 23.6745, 120.9465  # Center latitude / longitude
DEFAULT_NY, DEFAULT_NX = 304, 304               # Desired grid dimensions

# === Input / Output ===
INPUT_FILE = "./TAIESM_tw3.5km_coord2d.nc"
VARS = {
    "XLAT":     {"unit": "degrees_north", "required": True},
    "XLONG":    {"unit": "degrees_east",  "required": True},
    "LANDMASK": {"unit": "land mask",     "required": True},
    "TER":      {"unit": "meters",        "required": False},
    "SLOPE":    {"unit": "slope",         "required": False},
    "ASPECT":   {"unit": "degree",        "required": False},
}


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def load_var(nc: Dataset, var_name: str):
    """Load a variable; error if required and missing, else return None."""
    spec = VARS[var_name]
    if var_name in nc.variables:
        return nc.variables[var_name][:]
    if spec["required"]:
        raise KeyError(f"Required variable '{var_name}' not found in input file.")
    return None


def write_var(ncfile: Dataset, var_name: str, data) -> None:
    """Create and write a 2D variable to the NetCDF output file."""
    unit = VARS[var_name]["unit"]
    var = ncfile.createVariable(var_name, "f4", ("south_north", "west_east"))
    var[:, :] = data
    var.units = unit


def load_input_fields(input_path: str):
    """Load all fields from INPUT_FILE according to VARS spec."""
    nc_in = Dataset(input_path, mode="r")

    var_data = {name: load_var(nc_in, name) for name in VARS}
    lat, lon = var_data["XLAT"], var_data["XLONG"]
    layer_data = {
        name: var_data[name] for name in VARS if name not in ("XLAT", "XLONG")
    }

    found_layers = [name for name, arr in layer_data.items() if arr is not None]
    print(f"Input file: {input_path}\nFound layers: {found_layers}\n")
    print(f"Input grid (lat, lon) = ({lat.shape[0]}, {lon.shape[1]})")

    return lat, lon, layer_data


def regrid_to_larger(
    lat,
    lon,
    layer_data: Dict,
    img_size: Tuple[int, int],
):
    """Regrid input fields to a larger (ny, nx) grid using xESMF."""
    ny, nx = img_size
    print("Extrapolating to larger grid...")

    # Create new lat/lon grid
    new_lat = np.linspace(lat.min(), lat.max(), ny)
    new_lon = np.linspace(lon.min(), lon.max(), nx)
    new_lat2d, new_lon2d = np.meshgrid(new_lat, new_lon, indexing="ij")
    new_grid = xr.Dataset(
        {
            "lat": (["south_north", "west_east"], new_lat2d),
            "lon": (["south_north", "west_east"], new_lon2d),
        }
    )

    # Use xESMF for extrapolation
    src = xr.Dataset(
        {
            "lat": (["south_north", "west_east"], lat),
            "lon": (["south_north", "west_east"], lon),
        }
    )
    regridder = xe.Regridder(
        src,
        new_grid,
        method="bilinear",
        extrap_method="nearest_s2d",
    )

    lat_grid = new_grid["lat"].values
    lon_grid = new_grid["lon"].values

    # Regrid all existing fields
    for name, arr in layer_data.items():
        if arr is not None:
            layer_data[name] = regridder(xr.DataArray(arr))

    return lat_grid, lon_grid, layer_data


def crop_to_smaller(
    lat,
    lon,
    layer_data: Dict,
    center: Tuple[float, float],
    img_size: Tuple[int, int],
):
    """Crop input fields to a smaller (ny, nx) grid centered at (clat, clon)."""
    ny, nx = img_size
    clat, clon = center

    print("Cropping to smaller grid ...")

    # Find center indices
    idy = np.abs(lat[:, 0] - clat).argmin()
    idx = np.abs(lon[0, :] - clon).argmin()

    # Calculate slicing indices
    slat = max(0, idy - ny // 2)
    elat = min(lat.shape[0], idy + ny // 2)
    slon = max(0, idx - nx // 2)
    elon = min(lon.shape[1], idx + nx // 2)
    print(f"  slice (lat, lon) = [{slat}:{elat}, {slon}:{elon}]")

    # Crop the grid
    lat_grid = lat[slat:elat, slon:elon]
    lon_grid = lon[slat:elat, slon:elon]

    # Crop all existing fields
    for name, arr in layer_data.items():
        if arr is not None:
            layer_data[name] = arr[slat:elat, slon:elon]

    return lat_grid, lon_grid, layer_data


def save_output(
    output_path: str,
    lat_grid,
    lon_grid,
    layer_data: dict,
    img_size: Tuple[int]
) -> None:
    """Write cropped/regridded grid + layers to OUTPUT_FILE."""
    out_path = Path(output_path)
    if out_path.exists():
        out_path.unlink(missing_ok=True)

    with Dataset(output_path, mode="w", format="NETCDF4") as ncfile:
        # Create dimensions
        ncfile.createDimension("south_north", lat_grid.shape[0])
        ncfile.createDimension("west_east", lon_grid.shape[1])

        # Coordinates
        write_var(ncfile, "XLAT", lat_grid)
        write_var(ncfile, "XLONG", lon_grid)

        # All other fields, only if present
        for name, arr in layer_data.items():
            if arr is not None:
                write_var(ncfile, name, arr)

        ny, nx = img_size
        ncfile.setncattr("coordinates", "XLAT XLONG")
        ncfile.setncattr("description", f"CorrDiff REF grid {ny}x{nx}")

    print(f"Output written to => {output_path}")


def process_grid(
    input_file: str,
    output_file: str,
    center: Tuple[float, float],
    img_size: Tuple[int, int],
) -> None:
    """High-level driver: load → crop/regrid → save."""
    lat, lon, layers = load_input_fields(input_file)

    ny, nx = img_size
    need_regrid = ny > lat.shape[0] or nx > lon.shape[1]

    lat_grid, lon_grid, layer_data = (
        regrid_to_larger(lat, lon, layers, img_size) if need_regrid
        else crop_to_smaller(lat, lon, layers, center, img_size)
    )

    save_output(output_file, lat_grid, lon_grid, layer_data, img_size)


# -----------------------------------------------------------
# main
# -----------------------------------------------------------
def main() -> None:
    """
    CLI wrapper.

    Usage
    -----
    - With defaults:
        python generate_wrf_coord.py

    - With custom input file and size:
        python generate_wrf_coord.py <input_file> <ny> <nx>
    """
    argc = len(sys.argv)

    if argc == 1:
        input_file = INPUT_FILE
        ny, nx = DEFAULT_NY, DEFAULT_NX
    elif argc == 4:
        try:
            input_file = sys.argv[1]
            ny = int(sys.argv[2])
            nx = int(sys.argv[3])
        except ValueError:
            print("Error: <ny>, <nx> must be int.")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python generate_wrf_coord.py")
        print("  python generate_wrf_coord.py <input_file> <ny> <nx>")
        sys.exit(1)

    output_file = f"./wrf_{ny}x{nx}_grid_coords.nc"
    process_grid(input_file, output_file,
                 (DEFAULT_CLAT, DEFAULT_CLON), # center lat / lon
                 (ny, nx)) # image size


if __name__ == "__main__":
    main()
