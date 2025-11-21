"""
CorrDiff Dataset Generation and Zarr Storage.

This script processes TaiESM 3.5km & 100km datasets to generate a consolidated dataset,
verify its integrity, and save it in Zarr format. It integrates multiple data processing
modules and performs spatial regridding, variable aggregation, and data compression.

Features:
- Processes TaiESM 3.5km & 100km datasets for a specified date range.
- Regrids datasets to a reference grid.
- Generates a consolidated dataset with key variables and metrics:
  - Center (mean)
  - Scale (standard deviation)
  - Validity masks
- Verifies the structure and integrity of the dataset.
- Saves the dataset in compressed Zarr format.

Functions:
- `generate_output_dataset`: Combines processed TaiESM 100km & 3.5km data into a consolidated dataset.
- `write_to_zarr`: Writes the consolidated dataset to Zarr format with compression.
- `get_ref_grid`: Loads the reference grid dataset and extracts the required coordinates.
- `generate_corrdiff_zarr`: Orchestrates the generation, verification, and saving of the dataset.
- `main`: Parses command-line arguments and triggers the dataset generation process.

Dependencies:
- `sys`: For command-line argument parsing.
- `zarr`: For handling Zarr storage format.
- `xarray`: For multi-dimensional labeled data operations.
- `numpy`: For numerical operations.
- `dask.diagnostics.ProgressBar`: For monitoring progress during dataset writing.
- Modules:
  - `taiesmep5`: For TaiESM 3.5km dataset processing.
  - `taiesm100`: For TaiESM 100km dataset processing.
  - `util`: For utility functions like dataset verification and regridding.

Usage:
    python corrdiff_sspgen.py <start_date> <end_date>

    Example:
        python corrdiff_sspgen.py 20180101 20180103

Notes:
- Ensure that the `REF_GRID_NC` file exists and contains valid reference grid data.
- The script handles both local and remote environments based on the presence of specific folders.

"""
import sys
from typing import Tuple
from enum import Enum

import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

from taiesm3p5 import generate_output as generate_taiesm3p5_output
# from taiesm100 import generate_output as generate_taiesm100_output
from util import verify_dataset, dump_regrid_netcdf

DEBUG = False  # Set to True to enable debugging
REF_GRID_NC = "./ref_grid/wrf_304x304_grid_coords.nc"
GRID_COORD_KEYS = ["XLAT", "XLONG"]

class SSP(str, Enum):
    historical = "historical"
    ssp126 = "ssp126"
    ssp245 = "ssp245"
    ssp370 = "ssp370"
    ssp585 = "ssp585"

def get_ref_grid() -> Tuple[xr.Dataset, dict, dict]:
    """
    Load the reference grid dataset and extract its coordinates

    This function reads a predefined reference grid NetCDF file and extracts:
    - A dataset containing latitude (`lat`) and longitude (`lon`) grids.
    - A dictionary of coordinate arrays specified by `GRID_COORD_KEYS`.

    Returns:
        tuple:
            - grid (xarray.Dataset): A dataset containing the latitude ('lat') and
              longitude ('lon') grids for spatial alignment.
            - grid_coords (dict): A dictionary of extracted coordinate arrays defined
              by `GRID_COORD_KEYS` for downstream processing.

    Notes:
        - The reference grid file path is defined by the global constant `REF_GRID_NC`.
        - The coordinate keys to extract are defined in `GRID_COORD_KEYS`.
    """
    ref = xr.open_dataset(REF_GRID_NC, engine='netcdf4')
    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }

    return grid, grid_coords

def generate_output_dataset(start_date: str, end_date: str) -> xr.Dataset:
    """
    Generates a consolidated output dataset by processing TaiESM 3.5km & 100km data fields.

    Parameters:
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.

    Returns:
        xr.Dataset: A dataset containing consolidated and processed TaiESM 3.5km & 100km data fields.
    """
    # Get REF grid
    grid, grid_coords = get_ref_grid()

    # Generate TaiESM 3.5km & 100km output fields
    taiesm3p5_outputs = generate_taiesm3p5_output(grid, start_date, end_date, SSP.historical)
    return
    taiesm100_outputs = generate_taiesm100_output(grid, start_date, end_date)

    # Group outputs into dictionaries
    ground_truth_data = {
        "cwb": taiesm3p5_outputs[0],
        "cwb_variable": taiesm3p5_outputs[1],
        "cwb_center": taiesm3p5_outputs[2],
        "cwb_scale": taiesm3p5_outputs[3],
        "cwb_valid": taiesm3p5_outputs[4],
        "pre_regrid": taiesm3p5_outputs[5],
        "post_regrid": taiesm3p5_outputs[6],
    }
    input_data = {
        "era5": taiesm100_outputs[0],
        "era5_center": taiesm100_outputs[1],
        "era5_scale": taiesm100_outputs[2],
        "era5_valid": taiesm100_outputs[3],
        "pre_regrid": taiesm100_outputs[4],
        "post_regrid": taiesm100_outputs[5],
    }

    # Create the output dataset
    out = xr.Dataset(
        coords={
            **{key: grid_coords[key] for key in GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-02-08 16:00:00", "ns"),  # Placeholder for timestamp
            "time": ground_truth_data["cwb"].time,
            "cwb_variable": ground_truth_data["cwb_variable"],
            "era5_scale": ("era5_channel", input_data["era5_scale"].data),
        }
    )

    # Assign CWB and ERA5 data variables
    out = out.assign({
        "cwb": ground_truth_data["cwb"],
        "cwb_center": ground_truth_data["cwb_center"],
        "cwb_scale": ground_truth_data["cwb_scale"],
        "cwb_valid": ground_truth_data["cwb_valid"],
        "era5": input_data["era5"],
        "era5_center": input_data["era5_center"],
        "era5_valid": input_data["era5_valid"],
    }).drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices
    if DEBUG:
        dump_regrid_netcdf(
            f"{start_date}_{end_date}",
            ground_truth_data["pre_regrid"],
            ground_truth_data["post_regrid"],
            input_data["pre_regrid"],
            input_data["post_regrid"],
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
    Generates and verifies a consolidated dataset for TaiESM 3.5km & 100km  data,
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
    write_to_zarr(f"ssp_dataset_{start_date}_{end_date}.zarr", out)

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
