"""
CorrDiff dataset generator: build, validate, and export multi-source inputs to Zarr.

This script orchestrates the end-to-end pipeline for creating CorrDiff-ready
datasets from high-resolution (TReAD / TaiESM 3.5 km) and low-resolution
(ERA5 / TaiESM 100 km) sources on a common WRF-style reference grid.

Main responsibilities
---------------------
- Drive data loading and preprocessing via `data_builder`:
    * `generate_cwa_outputs()`  - TReAD + ERA5.
    * `generate_ssp_outputs()`  - TaiESM 3.5 km + 100 km.
- Combine high-res and low-res tensors into a single `xarray.Dataset`
  (`generate_output_dataset`).
- Optionally dump pre- and post-regrid NetCDF snapshots for debugging
  (`dump_regrid_netcdf` when `DEBUG` is True).
- Verify that the final dataset meets CorrDiff geometry and metadata
  requirements (`verify_dataset`).
- Write the consolidated dataset to compressed Zarr storage
  (`write_to_zarr`).

Usage
-----
Run from the command line:

    CWA / TReAD + ERA5 mode:
        python corrdiff_datagen.py <start_date> <end_date>

    SSP / TaiESM mode:
        python corrdiff_datagen.py <start_date> <end_date> <ssp_level>

Examples
--------
    python corrdiff_datagen.py 20180101 20180103
    python corrdiff_datagen.py 20180101 20180103 ssp126

The output is a validated CorrDiff-ready Zarr dataset named:

    corrdiff_dataset_<start_date>_<end_date>.zarr
    or
    <ssp_level>_dataset_<start_date>_<end_date>.zarr
"""
import sys
from pathlib import Path

import xarray as xr
import numpy as np
from numcodecs import Blosc
from dask.diagnostics import ProgressBar

from data_builder import (
    GRID_COORD_KEYS, generate_cwa_outputs, generate_ssp_outputs, validate_ssp_level
)

DEBUG = False  # Set to True to enable debugging

def dump_regrid_netcdf(
    subdir: str,
    hr_pre_regrid: xr.Dataset,
    hr_post_regrid: xr.Dataset,
    lr_pre_regrid: xr.Dataset,
    lr_post_regrid: xr.Dataset
) -> None:
    """
    Saves the provided datasets to NetCDF files within a specified subdirectory.

    Parameters:
        subdir (str): The subdirectory path where the NetCDF files will be saved.
        hr_pre_regrid (xr.Dataset): The high-resolution dataset before regridding.
        hr_post_regrid (xr.Dataset): The high-resolution dataset after regridding.
        lr_pre_regrid (xr.Dataset): The low-resolution dataset before regridding.
        lr_post_regrid (xr.Dataset): The low-resolution dataset after regridding.

    Returns:
        None
    """
    folder = Path(f"./nc_dump/{subdir}")
    folder.mkdir(parents=True, exist_ok=True)

    for dataset, name in [
        (hr_pre_regrid, "highres_pre_regrid.nc"),
        (hr_post_regrid, "highres_post_regrid.nc"),
        (lr_pre_regrid, "lowres_pre_regrid.nc"),
        (lr_post_regrid, "lowres_post_regrid.nc")
    ]:
        dataset.to_netcdf(folder / name)


def generate_output_dataset(start_date: str, end_date: str, ssp_level: str) -> xr.Dataset:
    """
    Generates a consolidated output dataset by processing low-res and high-res data fields.

    Parameters:
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
        generate_ssp_outputs(start_date, end_date, ssp_level)
        if ssp_level else generate_cwa_outputs(start_date, end_date)
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
            **{key: grid_coords[key] for key in GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-12-10 09:00:00", "ns"),  # Placeholder for timestamp
            "time": hr_data["cwb"].time,
            "cwb_variable": hr_data["cwb_variable"],
            "era5_scale": ("era5_channel", lr_data["era5_scale"].data),
        },
    ).drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices
    if DEBUG:
        dump_regrid_netcdf(
            f"{start_date}_{end_date}",
            hr_data["pre_regrid"], hr_data["post_regrid"],
            lr_data["pre_regrid"], lr_data["post_regrid"],
        )

    return out


def verify_dataset(ds: xr.Dataset) -> tuple[bool, str]:
    """
    Verifies an xarray.Dataset to ensure:
    1. Dimensions 'south_north' and 'west_east' are equal and both are multiples of 16.
    2. The dataset includes all specified coordinates and data variables.

    Parameters:
    - dataset: xarray.Dataset to verify.

    Returns:
    - A tuple (bool, str) where:
      - bool: True if the dataset passes all checks, False otherwise.
      - str: A message describing the result.
    """
    # Required dimensions, coordinates and data variables
    required_dims = [
        "time", "south_north", "west_east", "cwb_channel", "era5_channel"
    ]
    required_coords = [
        "time", "XLONG", "XLAT", "cwb_pressure", "cwb_variable",
        "era5_scale", "era5_pressure", "era5_variable"
    ]
    required_vars = [
        "cwb", "cwb_center", "cwb_scale", "cwb_valid",
        "era5", "era5_center", "era5_valid"
    ]

    # Check required dimensions
    missing_dims = [dim for dim in required_dims if dim not in ds.dims]
    if missing_dims:
        return False, f"Missing required dimensions: {', '.join(missing_dims)}."
    if ds.sizes["south_north"] != ds.sizes["west_east"]:
        return False, "Dimensions 'south_north' and 'west_east' are not equal."
    if ds.sizes["south_north"] % 16 != 0:
        return False, "Dimensions 'south_north' and 'west_east' are not multiples of 16."

    # Check coordinates
    missing_coords = [coord for coord in required_coords if coord not in ds.coords]
    if missing_coords:
        return False, f"Missing required coordinates: {', '.join(missing_coords)}."

    # Check data variables
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        return False, f"Missing required data variables: {', '.join(missing_vars)}."

    # All checks passed
    return True, "Dataset verification passed successfully."


def write_to_zarr(out_path: str, out_ds: xr.Dataset) -> None:
    """
    Writes the given dataset to a Zarr storage format with compression.

    Parameters:
        out_path (str): The file path where the Zarr dataset will be saved.
        out_ds (xr.Dataset): The dataset to be written to Zarr format.

    Returns:
        None
    """
    comp = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print(f"\nSaving data to {out_path}:")
    with ProgressBar():
        out_ds.to_zarr(out_path, mode='w', encoding=encoding, compute=True, zarr_format=2)

    print(f"Data successfully saved to [{out_path}]")


def generate_corrdiff_zarr(start_date: str, end_date: str, ssp_level: str = '') -> None:
    """
    Generates and verifies a consolidated dataset for low-res and high-res data,
    then writes it to a Zarr file format.

    Parameters:
        start_date (str): Start date of the data range in 'YYYYMMDD' format.
        end_date (str): End date of the data range in 'YYYYMMDD' format.
        ssp_level (str, optional): SSP level used to select the TaiESM dataset directory
                                    (e.g., 'historical', 'ssp126', 'ssp245').

    Returns:
        None
    """
    # Generate the output dataset.
    out = generate_output_dataset(start_date, end_date, ssp_level)
    print(f"\nZARR dataset =>\n {out}")

    # Verify the output dataset.
    passed, message = verify_dataset(out)
    if not passed:
        print(f"\nDataset verification failed => {message}")
        return

    # Write the output dataset to ZARR.
    suffix = f"_{ssp_level}" if ssp_level else ""
    write_to_zarr(f"corrdiff_dataset_{start_date}_{end_date}{suffix}.zarr", out)


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
        generate_corrdiff_zarr(sys.argv[1], sys.argv[2])
    elif argc == 4:
        generate_corrdiff_zarr(sys.argv[1], sys.argv[2], validate_ssp_level(sys.argv[3]))


if __name__ == "__main__":
    main()
