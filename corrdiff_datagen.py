"""
CorrDiff Dataset Generation and Zarr Storage.

This script processes TReAD and ERA5 datasets to generate a consolidated dataset,
verify its integrity, and save it in Zarr format. It integrates multiple data processing
modules and performs spatial regridding, variable aggregation, and data compression.

Features:
- Processes TReAD and ERA5 datasets for a specified date range.
- Regrids datasets to a reference grid.
- Generates a consolidated dataset with key variables and metrics:
  - Center (mean)
  - Scale (standard deviation)
  - Validity masks
- Verifies the structure and integrity of the dataset.
- Saves the dataset in compressed Zarr format.

Functions:
- `generate_output_dataset`: Combines processed TReAD and ERA5 data into a consolidated dataset.
- `write_to_zarr`: Writes the consolidated dataset to Zarr format with compression.
- `get_ref_grid`: Loads the reference grid dataset and extracts the required coordinates
   and terrain data.
- `generate_corrdiff_zarr`: Orchestrates the generation, verification, and saving of the dataset.
- `main`: Parses command-line arguments and triggers the dataset generation process.

Dependencies:
- `sys`: For command-line argument parsing.
- `zarr`: For handling Zarr storage format.
- `xarray`: For multi-dimensional labeled data operations.
- `numpy`: For numerical operations.
- `dask.diagnostics.ProgressBar`: For monitoring progress during dataset writing.
- Modules:
  - `tread`: For TReAD dataset processing.
  - `era5`: For ERA5 dataset processing.
  - `util`: For utility functions like dataset verification and regridding.

Usage:
    python corrdiff_datagen.py <start_date> <end_date>

    Example:
        python corrdiff_datagen.py 20180101 20180103

Notes:
- Ensure that the `REF_GRID_NC` file exists and contains valid reference grid data.
- The script handles both local and remote environments based on the presence of specific folders.

"""
import sys
from typing import Tuple

import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from tread import generate_tread_output
from era5 import generate_era5_output
from util import is_local_testing, verify_dataset, dump_regrid_netcdf

DEBUG = False  # Set to True to enable debugging
REF_GRID_NC = "./ref_grid/wrf_208x208_grid_coords.nc"
GRID_COORD_KEYS = ["XLAT", "XLONG"]

def get_ref_grid() -> Tuple[xr.Dataset, dict, dict]:
    """
    Load the reference grid dataset and extract its coordinates and terrain-related variables.

    This function reads a predefined reference grid NetCDF file and extracts:
    - A dataset containing latitude (`lat`) and longitude (`lon`) grids.
    - A dictionary of coordinate arrays specified by `GRID_COORD_KEYS`.
    - A dictionary of terrain-related variables (`TER`, `SLOPE`, `ASPECT`) for use in
      regridding and terrain processing.

    Returns:
        tuple:
            - grid (xarray.Dataset): A dataset containing the latitude ('lat') and
              longitude ('lon') grids for spatial alignment.
            - grid_coords (dict): A dictionary of extracted coordinate arrays defined
              by `GRID_COORD_KEYS` for downstream processing.
            - terrain_layers (dict): A dictionary containing terrain-related variables
              ('ter', 'slope', 'aspect') from the reference grid.

    Notes:
        - The reference grid file path is defined by the global constant `REF_GRID_NC`.
        - The coordinate keys to extract are defined in `GRID_COORD_KEYS`.
        - The terrain-related variables are returned as a dictionary with lowercase keys
          for consistency in downstream processing.
    """
    ref = xr.open_dataset(REF_GRID_NC, engine='netcdf4')
    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }

    return grid, grid_coords, {key.lower(): ref[key] for key in ["TER", "SLOPE", "ASPECT"]}

def generate_output_dataset(start_date: str, end_date: str) -> xr.Dataset:
    """
    Generates a consolidated output dataset by processing TReAD and ERA5 data fields.

    Parameters:
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.

    Returns:
        xr.Dataset: A dataset containing consolidated and processed TReAD and ERA5 data fields.
    """
    # Get REF grid
    grid, grid_coords, layers = get_ref_grid()

    # Generate CWB (TReAD) and ERA5 output fields
    tread_outputs = generate_tread_output(grid, start_date, end_date)
    era5_outputs = generate_era5_output(grid, layers, start_date, end_date)

    # Group outputs into dictionaries
    tread_data = {
        "cwb": tread_outputs[0],
        "cwb_variable": tread_outputs[1],
        "cwb_center": tread_outputs[2],
        "cwb_scale": tread_outputs[3],
        "cwb_valid": tread_outputs[4],
        "pre_regrid": tread_outputs[5],
        "post_regrid": tread_outputs[6],
    }
    era5_data = {
        "era5": era5_outputs[0],
        "era5_center": era5_outputs[1],
        "era5_scale": era5_outputs[2],
        "era5_valid": era5_outputs[3],
        "pre_regrid": era5_outputs[4],
        "post_regrid": era5_outputs[5],
    }

    # Create the output dataset
    out = xr.Dataset(
        coords={
            **{key: grid_coords[key] for key in GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-02-08 16:00:00", "ns"),  # Placeholder for timestamp
            "time": tread_data["cwb"].time,
            "cwb_variable": tread_data["cwb_variable"],
            "era5_scale": ("era5_channel", era5_data["era5_scale"].data),
        }
    )

    # Assign CWB and ERA5 data variables
    out = out.assign({
        "cwb": tread_data["cwb"],
        "cwb_center": tread_data["cwb_center"],
        "cwb_scale": tread_data["cwb_scale"],
        "cwb_valid": tread_data["cwb_valid"],
        "era5": era5_data["era5"],
        "era5_center": era5_data["era5_center"],
        "era5_valid": era5_data["era5_valid"],
    }).drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices
    if DEBUG:
        dump_regrid_netcdf(
            f"{start_date}_{end_date}",
            tread_data["pre_regrid"],
            tread_data["post_regrid"],
            era5_data["pre_regrid"],
            era5_data["post_regrid"],
        )

    return out

def write_to_zarr(out_path: str, out_ds: xr.Dataset) -> None:
    """
    Writes the given dataset to a Zarr storage format with compression.

    Parameters:
        out_path (str): The file path where the Zarr dataset will be saved.
        out_ds (xr.Dataset): The dataset to be written to Zarr format.

    Returns:
        None
    """
    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print(f"\nSaving data to {out_path}:")
    with ProgressBar():
        out_ds.to_zarr(out_path, mode='w', encoding=encoding, compute=True)

    print(f"Data successfully saved to [{out_path}]")

def generate_corrdiff_zarr(start_date: str, end_date: str) -> None:
    """
    Generates and verifies a consolidated dataset for TReAD and ERA5 data,
    then writes it to a Zarr file format.

    Parameters:
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.

    Returns:
        None
    """
    # Generate the output dataset.
    out = generate_output_dataset(start_date, end_date)
    print(f"\nZARR dataset =>\n {out}")

    # Verify the output dataset.
    passed, message = verify_dataset(out)
    if not passed:
        print(f"\nDataset verification failed => {message}")
        return

    # Write the output dataset to ZARR.
    write_to_zarr(f"corrdiff_dataset_{start_date}_{end_date}.zarr", out)

def main():
    """
    Main entry point for the script. Parses command-line arguments to generate
    a Zarr dataset for a specified date range.

    Command-line Usage:
        python corrdiff_datagen.py <start_date> <end_date>

    Example:
        python corrdiff_datagen.py 20180101 20180103

    Returns:
        None
    """
    if len(sys.argv) < 3:
        print("Usage: python corrdiff_datagen.py <start_date> <end_date>")
        print("  e.g., $python corrdiff_datagen.py 20180101 20180103")
        sys.exit(1)

    generate_corrdiff_zarr(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
