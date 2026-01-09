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
import xarray as xr
import numpy as np

from corrdiff_datagen import verify_dataset, write_to_zarr, dump_regrid_netcdf
from data_builder import GRID_COORD_KEYS, generate_cordex_train_outputs

DEBUG = False  # Set to True to enable debugging


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
