"""
CorrDiff dataset generation and serialization script.

This module builds, validates, and writes CorrDiff-compatible datasets
to Zarr format for multiple experiment modes, including:

- CWA / TReAD + ERA5 historical data
- SSP scenario simulations
- CORDEX downscaling experiments (train and test configurations)

The script orchestrates data generation via functions in ``data_builder``,
assembles high-resolution (HR) and low-resolution (LR) components into a
unified xarray.Dataset, verifies dataset integrity, and writes the result
to compressed Zarr stores.

Supported command-line modes
----------------------------
CWA mode:
    python corrdiff_datagen.py <start_date> <end_date>

SSP mode:
    python corrdiff_datagen.py <start_date> <end_date> <ssp_level>

CORDEX mode:
    python corrdiff_datagen.py cordex

Key responsibilities
--------------------
- Generate HR/LR outputs for different experiment configurations
- Assemble CorrDiff datasets with consistent dimensions and coordinates
- Validate datasets against CorrDiff schema requirements
- Serialize datasets to Zarr with compression
- Optionally dump intermediate NetCDF files for debugging

This script is intended to be run as a standalone entry point in the
CorrDiff data preparation workflow.
"""
import sys
from pathlib import Path
from itertools import product

import xarray as xr
import numpy as np
from numcodecs import Blosc
from dask.diagnostics import ProgressBar

from data_builder import (
    GRID_COORD_KEYS, generate_cwa_outputs,
    generate_ssp_outputs, validate_ssp_level,
    generate_cordex_train_outputs, generate_cordex_test_outputs
)

DEBUG = False  # Set to True to enable debugging
XTIME = np.datetime64("2026-01-09 17:00:00", "ns")  # placeholder timestamp


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


def build_out(hr_outputs, lr_outputs, grid_coords, tag: str) -> xr.Dataset:
    """
    Assemble the final CorrDiff output dataset from HR and LR components.

    This function combines preprocessed high-resolution (HR) and low-resolution (LR)
    outputs into a single xarray.Dataset that conforms to the CorrDiff training
    and evaluation schema. It merges normalized data variables, attaches shared
    grid coordinates, and drops intermediate dimensions not required by the
    CorrDiff model.

    Parameters
    ----------
    hr_outputs : tuple
        Tuple of HR outputs in CorrDiff order, containing:
        (fields, variable metadata, normalization center, normalization scale,
         validity mask, pre-regrid dataset, post-regrid dataset).
    lr_outputs : tuple
        Tuple of LR outputs in CorrDiff order, containing:
        (fields, normalization center, normalization scale, validity mask,
         pre-regrid dataset, post-regrid dataset).
    grid_coords : xr.Dataset
        Dataset containing grid coordinate arrays (e.g., XLAT, XLONG) defining
        the spatial domain.
    tag : str
        Identifier used for optional debugging output (e.g., NetCDF dumps).

    Returns
    -------
    xr.Dataset
        Consolidated CorrDiff dataset containing HR and LR variables, coordinates,
        and metadata, ready for validation and serialization.
    """
    hr_keys = ["cwb", "cwb_variable", "cwb_center", "cwb_scale", "cwb_valid",
               "pre_regrid", "post_regrid"]
    lr_keys = ["era5", "era5_center", "era5_scale", "era5_valid",
               "pre_regrid", "post_regrid"]
    hr, lr = dict(zip(hr_keys, hr_outputs)), dict(zip(lr_keys, lr_outputs))

    out = (
        xr.Dataset(
            data_vars={
                "cwb": hr["cwb"], "cwb_center": hr["cwb_center"],
                "cwb_scale": hr["cwb_scale"], "cwb_valid": hr["cwb_valid"],

                "era5": lr["era5"], "era5_center": lr["era5_center"],
                "era5_valid": lr["era5_valid"],
            },
            coords={
                **{k: grid_coords[k] for k in GRID_COORD_KEYS},
                "XTIME": XTIME,
                "time": hr["cwb"].time,
                "cwb_variable": hr["cwb_variable"],
                "era5_scale": ("era5_channel", lr["era5_scale"].data),
            },
        )
        .drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])
    )

    if DEBUG:
        dump_regrid_netcdf(tag, hr["pre_regrid"], hr["post_regrid"],
                           lr["pre_regrid"], lr["post_regrid"])

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
    comp = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    encoding = { var: {'compressor': comp} for var in out_ds.data_vars }

    print(f"\nSaving data to {out_path}:")
    with ProgressBar():
        out_ds.to_zarr(out_path, mode='w', encoding=encoding, compute=True, zarr_format=2)

    print(f"Data successfully saved to [{out_path}]")


def create_corrdiff_zarr(prefix: str, tag: str, outputs) -> None:
    """
    Build, validate, and write a CorrDiff dataset to a Zarr store.

    This function:
      1) Builds an output dataset using ``build_out``
      2) Prints a summary of the dataset
      3) Verifies dataset integrity using ``verify_dataset``
      4) Writes the dataset to a Zarr directory if verification succeeds

    Parameters
    ----------
    prefix : str
        Filename prefix for the output Zarr store (e.g. ``"corrdiff_dataset"``).
    tag : str
        Tag identifying the dataset configuration (used in dataset metadata
        and output filename).
    outputs :
        Tuple of outputs returned by a generator function
        (e.g. ``generate_cwa_outputs`` or ``generate_cordex_train_outputs``).

    Returns
    -------
    None
        The dataset is written to disk as ``<prefix>_<tag>.zarr``.
    """
    ds = build_out(*outputs, tag=tag)
    print(f"\nZARR dataset =>\n{ds}")

    ok, msg = verify_dataset(ds)
    if not ok:
        print(f"\nDataset verification failed => {msg}")
        return

    write_to_zarr(f"{prefix}_{tag}.zarr", ds)


def create_cordex_zarrs() -> None:
    """Create CorrDiff zarr files for CORDEX experiments."""
    exp_domains = ["ALPS", "NZ", "SA"]
    train_cfgs = ["ESD_pseudo_reality", "Emulator_hist_future"]
    gcm_sets = ["TG", "OOSG"]

    for exp_domain in exp_domains:
        # train
        for train_cfg in train_cfgs:
            create_corrdiff_zarr("cordex_train", f"{exp_domain}_{train_cfg[:3]}",
                                    generate_cordex_train_outputs(exp_domain, train_cfg))

        # test (TG / OOSG) x (perfect / imperfect)
        for test_cfg, perfect in product(gcm_sets, [False, True]):
            suffix = "perfect" if perfect else "imperfect"
            create_corrdiff_zarr(
                "cordex_test", f"{exp_domain}_{test_cfg}_{suffix}",
                generate_cordex_test_outputs(exp_domain, train_cfgs[0], test_cfg, perfect)
            )


def main():
    """
    Command-line entry point for CorrDiff dataset generation.

    Supports three modes:
      - CWA:
          python corrdiff_datagen.py <start_date> <end_date>
      - SSP:
          python corrdiff_datagen.py <start_date> <end_date> <ssp_level>
      - CORDEX:
          python corrdiff_datagen.py cordex

    Depending on the mode, this function generates training and/or test
    datasets, validates them, and writes the results to Zarr stores.
    """
    argc = len(sys.argv)
    if argc not in (2, 3, 4):
        print("Usage:")
        print("  CWA : python corrdiff_datagen.py <start> <end>")
        print("  SSP : python corrdiff_datagen.py <start> <end> <ssp_level>")
        print("  CORDEX : python corrdiff_datagen.py cordex")
        sys.exit(1)

    # CORDEX
    if sys.argv[1] == "cordex":
        create_cordex_zarrs()
        return

    # CWA / SSP
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    ssp_level = validate_ssp_level(sys.argv[3]) if argc == 4 else ''

    if argc == 3:   # CWA
        create_corrdiff_zarr("corrdiff_dataset", f"{start_date}_{end_date}",
                             generate_cwa_outputs(start_date, end_date))
    elif argc == 4: # SSP
        create_corrdiff_zarr("corrdiff_dataset", f"{start_date}_{end_date}_{ssp_level}",
                             generate_ssp_outputs(start_date, end_date, ssp_level))


if __name__ == "__main__":
    main()
