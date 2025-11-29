"""
CorrDiff Dataset Generation and Zarr Storage.

This script processes low-resolution (low-res) and high-resolution (high-res) datasets
to generate a consolidated dataset, verify its integrity, and save it in Zarr format.
It integrates multiple data processing modules and performs spatial regridding,
variable aggregation, and data compression.

Features:
- Processes low-res and high-res datasets for a specified date range.
- Regrids datasets to a reference grid.
- Generates a consolidated dataset with key variables and metrics:
  - Center (mean)
  - Scale (standard deviation)
  - Validity masks
- Verifies the structure and integrity of the dataset.
- Saves the dataset in compressed Zarr format.

Functions:
- `generate_output_dataset`: Combines processed low-res and high-res data into
                             a consolidated dataset.
- `write_to_zarr`: Writes the consolidated dataset to Zarr format with compression.
- `get_ref_grid`: Loads the reference grid dataset and extracts the required coordinates
   & optionally terrain data.
- `generate_corrdiff_zarr`: Orchestrates the generation, verification, and saving of the dataset.
- `main`: Parses command-line arguments and triggers the dataset generation process.

Dependencies:
- `sys`: For command-line argument parsing.
- `zarr`: For handling Zarr storage format.
- `xarray`: For multi-dimensional labeled data operations.
- `numpy`: For numerical operations.
- `dask.diagnostics.ProgressBar`: For monitoring progress during dataset writing.
- Modules:
  - `cwa_data`: CWA mode related functions.
  - `ssp_data`: SSP mode related functions.
  - `util`: For utility functions like dataset verification and regridding.

Usage:
    python corrdiff_datagen.py <start_date> <end_date>

    Example:
        python corrdiff_datagen.py 20180101 20180103
"""
import sys

import zarr
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

import cwa_data as cwa
import ssp_data as ssp
from util import verify_dataset, dump_regrid_netcdf

DEBUG = False  # Set to True to enable debugging

def generate_output_dataset(mode: str, start_date: str, end_date: str,
                            ssp_level: str) -> xr.Dataset:
    """
    Generates a consolidated output dataset by processing low-res and high-res data fields.

    Parameters:
        mode (str): Processing mode, either 'CWA' or 'SSP'.
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.
        ssp_level (str): SSP level used to select the TaiESM dataset directory
                            (e.g., 'historical', 'ssp126', 'ssp245').

    Returns:
        xr.Dataset: A dataset containing consolidated and processed
                    low-res and high-res data fields.
    """
    # Generate high-res and low-res output datasets
    hr_outputs, lr_outputs, grid_coords = (
        cwa.generate_output_dataset(start_date, end_date) if mode == "CWA"
        else ssp.generate_output_dataset(start_date, end_date, ssp_level)
    )

    # Group outputs into dictionaries
    hr_data = dict(zip(
        ["cwb", "cwb_variable", "cwb_center", "cwb_scale", "cwb_valid",
         "pre_regrid", "post_regrid"],
        hr_outputs,
    ))
    lr_data = dict(zip(
        ["era5", "era5_center", "era5_scale", "era5_valid",
         "pre_regrid", "post_regrid"],
        lr_outputs,
    ))

    # Create the output dataset
    out = xr.Dataset(
        data_vars={
            "cwb":         hr_data["cwb"],
            "cwb_center":  hr_data["cwb_center"],
            "cwb_scale":   hr_data["cwb_scale"],
            "cwb_valid":   hr_data["cwb_valid"],
            "era5":        lr_data["era5"],
            "era5_center": lr_data["era5_center"],
            "era5_valid":  lr_data["era5_valid"],
        },
        coords={
            **{key: grid_coords[key] for key in cwa.GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-11-27 10:00:00", "ns"),  # Placeholder for timestamp
            "time": hr_data["cwb"].time,
            "cwb_variable": hr_data["cwb_variable"],
            "era5_scale": ("era5_channel", lr_data["era5_scale"].data),
        },
    ).drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices
    if DEBUG:
        dump_regrid_netcdf(
            f"{start_date}_{end_date}",
            hr_data["pre_regrid"],
            hr_data["post_regrid"],
            lr_data["pre_regrid"],
            lr_data["post_regrid"],
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

def generate_corrdiff_zarr(mode: str, start_date: str, end_date: str, ssp_level: str = '') -> None:
    """
    Generates and verifies a consolidated dataset for low-res and high-res data,
    then writes it to a Zarr file format.

    Parameters:
        mode (str): Processing mode, either 'CWA' or 'SSP'.
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.
        ssp_level (str, optional): SSP level used to select the TaiESM dataset directory
                                    (e.g., 'historical', 'ssp126', 'ssp245').

    Returns:
        None
    """
    # Generate the output dataset.
    out = generate_output_dataset(mode, start_date, end_date, ssp_level)
    print(f"\nZARR dataset =>\n {out}")

    # Verify the output dataset.
    passed, message = verify_dataset(out)
    if not passed:
        print(f"\nDataset verification failed => {message}")
        return

    # Write the output dataset to ZARR.
    write_to_zarr(f"corrdiff_{ssp_level}_dataset_{start_date}_{end_date}.zarr", out)

def validate_ssp_level(raw: str) -> str:
    """
    Validate and normalize an SSP suffix string.
    """
    allowed_ssp_levels = {"historical", "ssp126", "ssp245", "ssp370", "ssp585"}
    if raw not in allowed_ssp_levels:
        raise ValueError(f"ssp_level must be one of {allowed_ssp_levels}")

    return raw

def main():
    """
    Main entry point for the script. Parses command-line arguments to generate
    a Zarr dataset for a specified date range.

     Usage
    -----
        CWA / TReAD+ERA5 mode:
            python corrdiff_datagen.py <start_date> <end_date>

        SSP / TaiESM mode:
            python corrdiff_datagen.py <start_date> <end_date> <ssp_level>

    Examples
    --------
        python corrdiff_datagen.py 20180101 20180103
        python corrdiff_datagen.py 20180101 20180103 ssp126
    """
    argc = len(sys.argv)
    if argc not in (3, 4):
        print("Usage:")
        print("  CWA : python corrdiff_datagen.py <start> <end>")
        print("  SSP : python corrdiff_datagen.py <start> <end> <ssp_level>")
        sys.exit(1)

    if argc == 3:
        generate_corrdiff_zarr('CWA', sys.argv[1], sys.argv[2])
    elif argc == 4:
        generate_corrdiff_zarr('SSP', sys.argv[1], sys.argv[2], validate_ssp_level(sys.argv[3]))

if __name__ == "__main__":
    main()
