"""
CorrDiff Dataset Generation and Zarr Storage.

This script processes low-resolution (low-res) and high-resolution (high-res) datasets to generate a consolidated dataset,
verify its integrity, and save it in Zarr format. It integrates multiple data processing
modules and performs spatial regridding, variable aggregation, and data compression.

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
- `generate_output_dataset`: Combines processed low-res and high-res data into a consolidated dataset.
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
  - CWA
    - `tread`: For TReAD dataset processing.
    - `era5`: For ERA5 dataset processing.
  - SSP
    - `taiesmep5`: For TaiESM 3.5 km dataset processing.
    - `taiesm100`: For TaiESM 100 km dataset processing.
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
from util import verify_dataset, dump_regrid_netcdf

DEBUG = False  # Set to True to enable debugging
GRID_COORD_KEYS = ["XLAT", "XLONG"]

def get_ref_grid(mode: str) -> Tuple[xr.Dataset, dict, dict]:
    """
    Load the reference grid dataset and extract its coordinates and terrain-related variables.

    This function reads a predefined reference grid NetCDF file and extracts:
    - A dataset containing latitude (`lat`) and longitude (`lon`) grids.
    - A dictionary of coordinate arrays specified by `GRID_COORD_KEYS`.
    - A dictionary of terrain-related variables (`TER`, `SLOPE`, `ASPECT`) for use in
      regridding and terrain processing.

    Parameters:
        mode (str): Processing mode, either 'CWA' or 'SSP'.

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
    # Reference grid paths
    ref_grid_path = (
        "./ref_grid/wrf_208x208_grid_coords.nc" if mode == "CWA"  # CWA domain
        else "./ref_grid/wrf_304x304_grid_coords.nc"  # SSP domain
    )
    ref = xr.open_dataset(ref_grid_path, engine='netcdf4')

    grid = xr.Dataset({ "lat": ref.XLAT, "lon": ref.XLONG })
    grid_coords = { key: ref.coords[key] for key in GRID_COORD_KEYS }
    layers = (
        { key.lower(): ref[key] for key in ["TER", "SLOPE", "ASPECT"] if key in ref }
        if mode == 'CWA' else {}
    )

    return grid, grid_coords, layers

def generate_hr_lr_outputs(
    mode: str,
    grid: xr.Dataset,
    layers: dict,
    start_date: str,
    end_date: str,
    ssp_level: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Generate paired high-resolution (HR) and low-resolution (LR) datasets
    according to the selected processing mode.

    Parameters
    ----------
    mode : str
        Processing mode. Must be one of:
            - "CWA": Use TReAD as high-res and ERA5 as low-res.
            - "SSP": Use TaiESM 3.5 km as high-res and TaiESM 100 km as low-res.
    grid : xr.Dataset
        Reference grid used for spatial alignment and regridding.
    layers : dict
        Optional terrain-related fields ('ter', 'slope', 'aspect').
        Required only in "CWA" mode; ignored for "SSP".
    start_date : str
        Start date in 'YYYYMMDD' format.
    end_date : str
        End date in 'YYYYMMDD' format.
    ssp_level : str
        SSP level used *only* in "SSP" mode (e.g., "historical", "ssp126", "ssp245").

    Returns
    -------
    (xr.Dataset, xr.Dataset)
        A tuple (hr_ds, lr_ds), where:
        - hr_ds: High-resolution dataset for the selected mode.
        - lr_ds: Low-resolution dataset for the selected mode.

    Notes
    -----
    - In "CWA" mode, this function returns (TReAD_output, ERA5_output).
    - In "SSP" mode, this function returns (TaiESM_3.5km_output, TaiESM_100km_output).
    - Import statements are inside the mode branches to reduce unnecessary
      dependency loading for modes that do not use those modules.
    """
    if mode == 'CWA':   # CWA
        from tread import generate_tread_output
        from era5 import generate_era5_output
        return (
            generate_tread_output(grid, start_date, end_date),
            generate_era5_output(grid, layers, start_date, end_date)
        )

    # SSP
    from taiesm3p5 import generate_output as generate_taiesm3p5_output
    from taiesm100 import generate_output as generate_taiesm100_output
    return (
        generate_taiesm3p5_output(grid, start_date, end_date, ssp_level),
        generate_taiesm100_output(grid, start_date, end_date, ssp_level)
    )

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
        xr.Dataset: A dataset containing consolidated and processed low-res and high-res data fields.
    """
    # Get REF grid
    grid, grid_coords, layers = get_ref_grid(mode)

    # Generate high-res and low-res output datasets
    hr_outputs, lr_outputs = generate_hr_lr_outputs(mode, grid, layers, start_date, end_date, ssp_level)

    # Group outputs into dictionaries
    hr_data = {
        "cwb": hr_outputs[0],
        "cwb_variable": hr_outputs[1],
        "cwb_center": hr_outputs[2],
        "cwb_scale": hr_outputs[3],
        "cwb_valid": hr_outputs[4],
        "pre_regrid": hr_outputs[5],
        "post_regrid": hr_outputs[6],
    }
    lr_data = {
        "era5": lr_outputs[0],
        "era5_center": lr_outputs[1],
        "era5_scale": lr_outputs[2],
        "era5_valid": lr_outputs[3],
        "pre_regrid": lr_outputs[4],
        "post_regrid": lr_outputs[5],
    }

    # Create the output dataset
    out = xr.Dataset(
        coords={
            **{key: grid_coords[key] for key in GRID_COORD_KEYS},
            "XTIME": np.datetime64("2025-02-08 16:00:00", "ns"),  # Placeholder for timestamp
            "time": hr_data["cwb"].time,
            "cwb_variable": hr_data["cwb_variable"],
            "era5_scale": ("era5_channel", lr_data["era5_scale"].data),
        }
    )

    # Assign CWB and ERA5 data variables
    out = out.assign({
        "cwb": hr_data["cwb"],
        "cwb_center": hr_data["cwb_center"],
        "cwb_scale": hr_data["cwb_scale"],
        "cwb_valid": hr_data["cwb_valid"],
        "era5": lr_data["era5"],
        "era5_center": lr_data["era5_center"],
        "era5_valid": lr_data["era5_valid"],
    }).drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])

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
    write_to_zarr(f"corrdiff_{ssp_level}dataset_{start_date}_{end_date}.zarr", out)

def validate_ssp_level(raw: str) -> str:
    """
    Validate and normalize an SSP suffix string.
    """
    ALLOWED_SSP = {"historical", "ssp126", "ssp245", "ssp370", "ssp585"}
    if raw not in ALLOWED_SSP:
        raise ValueError(f"ssp_level must be one of {ALLOWED_SSP}")

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
