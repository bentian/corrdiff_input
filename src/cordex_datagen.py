"""
CorrDiff CORDEX Zarr dataset generator.

This script assembles, verifies, and writes CORDEX-based training datasets
for the CorrDiff framework. It combines high-resolution (HR) and low-resolution
(LR) climate fields, applies normalization and regridding, and outputs a single
consolidated dataset in Zarr format suitable for model training.

Main responsibilities
---------------------
- Generate standardized HR and LR datasets via `generate_cordex_train_outputs`
- Assemble CorrDiff training tensors and metadata into a single xarray.Dataset
- Validate dataset structure, dimensions, and required variables
- Persist the final dataset to Zarr with compression
- Optionally dump intermediate NetCDF files for debugging

Supported configurations
------------------------
- Multiple experiment domains (e.g. ALPS, NZ)
- Multiple training configurations (e.g. ESD_pseudo_reality, Emulator_hist_future)

Entry points
------------
- `generate_corrdiff_zarr`: generate and write a single Zarr dataset
- `main`: batch-generate Zarr datasets for all domain/config combinations

Notes
-----
- Output datasets are chunked and compressed for efficient training-time access.
- Verification enforces spatial dimensions to be square and multiples of 16.
- Debug mode (`DEBUG = True`) enables NetCDF dumps of intermediate regridding steps.
"""
from pathlib import Path

import xarray as xr
import numpy as np
from numcodecs import Blosc
from dask.diagnostics import ProgressBar

from data_builder import (
    GRID_COORD_KEYS, generate_cordex_train_outputs
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


def generate_output_dataset(exp_domain: str, train_config: str) -> xr.Dataset:
    """
    Generates a consolidated output dataset by processing low-res and high-res data fields.

    Parameters:
        exp_domain (str): Experiment domain identifier (e.g., "ALPS", "NZ").
        train_config (str): Training configuration identifier
                            (e.g., "ESD_pseudo_reality", "Emulator_hist_future").

    Returns:
        xr.Dataset: A dataset containing consolidated and processed
                    low-res and high-res data fields.
    """
    # Generate high-res and low-res output datasets
    hr_outputs, lr_outputs, grid_coords = generate_cordex_train_outputs(exp_domain, train_config)

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
            "XTIME": np.datetime64("2026-01-09 17:00:00", "ns"),  # Placeholder for timestamp
            "time": hr_data["cwb"].time,
            "cwb_variable": hr_data["cwb_variable"],
            "era5_scale": ("era5_channel", lr_data["era5_scale"].data),
        },
    ).drop_vars(
        ["y", "x", "south_north", "west_east", "cwb_channel", "era5_channel"],
        errors="ignore"     # ignore error if (y, x) is absent
    )

    # [DEBUG] Dump data pre- & post-regridding, and print output data slices
    if DEBUG:
        dump_regrid_netcdf(
            f"{exp_domain}_{train_config[:3]}",
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


def generate_corrdiff_zarr(exp_domain: str, train_config: str) -> None:
    """
    Generates and verifies a consolidated dataset for low-res and high-res data,
    then writes it to a Zarr file format.

    Parameters:
        exp_domain (str): Experiment domain identifier (e.g., "ALPS", "NZ").
        train_config (str): Training configuration identifier
                            (e.g., "ESD_pseudo_reality", "Emulator_hist_future").

    Returns:
        None
    """
    # Generate the output dataset.
    out = generate_output_dataset(exp_domain, train_config)
    print(f"\nZARR dataset =>\n {out}")

    # Verify the output dataset.
    passed, message = verify_dataset(out)
    if not passed:
        print(f"\nDataset verification failed => {message}")
        return

    # Write the output dataset to ZARR.
    write_to_zarr(f"cordex_train_{exp_domain}_{train_config[:3]}.zarr", out)


def main():
    """
    Generate CorrDiff Zarr training datasets for all CORDEX combinations.

    Iterates over the specified experiment domains and training configurations,
    invoking `generate_corrdiff_zarr` for each Cartesian combination. This serves
    as the entry point for batch generation of CORDEX-based training data.
    """
    for exp_domain in ["ALPS", "NZ"]:
        for train_config in ["ESD_pseudo_reality", "Emulator_hist_future"]:
            generate_corrdiff_zarr(exp_domain, train_config)


if __name__ == "__main__":
    main()
