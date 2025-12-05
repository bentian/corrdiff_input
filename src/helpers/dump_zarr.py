"""
Zarr Dataset Inspection and Field Dump Utility.

This script provides functionality for inspecting and analyzing the content of Zarr datasets.
It includes methods to:
- Print slices of specific fields ('cwb' and 'era5') over time.
- Dump detailed information about Zarr dataset fields, including their structure and content.

Functions:
- print_slices_over_time: Prints time-series slices of 'cwb' and 'era5' data from a dataset.
- dump_zarr_fields: Inspects a Zarr file and prints details of specific fields and their structure.
- main: Parses the Zarr file path from command-line arguments and triggers the dump process.

Usage:
    python zarr_dump.py <file_path>

    Example:
        python zarr_dump.py corrdiff_dataset_20180101_20180103.zarr

Dependencies:
- `sys`: For command-line argument parsing.
- `zarr`: For interacting with Zarr datasets.
- `xarray`: For handling and visualizing labeled multi-dimensional arrays.

Example Usage:
    # Open and inspect a Zarr file
    dump_zarr_fields("path_to_zarr_file.zarr")

    # Print slices of data over time
    ds = xr.open_zarr("path_to_zarr_file.zarr")
    print_slices_over_time(ds, limit=5)

Notes:
- Ensure the Zarr file path is valid and accessible before running the script.
- The script expects the dataset to include 'cwb' and 'era5' variables for meaningful inspection.

"""
import sys
import zarr
import xarray as xr

def print_slices_over_time(ds: xr.Dataset, limit=10) -> None:
    """
    Prints slices of 'cwb' and 'era5' data over time from the given dataset.

    Parameters:
        ds (xr.Dataset): The dataset containing 'cwb' and 'era5' variables with a 'time' dimension.
        limit (int, optional): The maximum number of time steps to print. Defaults to 10.

    Returns:
        None
    """
    print("\n" + "-"*40)

    start = 0
    end = min(ds.time.size, start + limit)
    for t in ds.time[start:end]:
        # Select the data for the current time step
        cwb_data = ds['cwb'].sel(time=t)
        era5_data = ds['era5'].sel(time=t)
        half_shape_x = 208 // 2

        print(f"\nTime: {t.values} Half_Shape_X: {half_shape_x}")
        print("CWB Data Slice:")
        print(cwb_data[0, half_shape_x].values[half_shape_x - 10: half_shape_x + 10])
        print("ERA5 Data Slice:")
        print(era5_data[0, half_shape_x].values[half_shape_x - 10: half_shape_x + 10])

    print("\n" + "-"*40 + "\n")

def dump_zarr_fields(zarr_path: str) -> None:
    """
    Dumps specified fields from a Zarr file and prints their content and structure.

    Parameters:
        zarr_path (str): Path to the Zarr file.
    """
    # Open the Zarr file in consolidated mode
    group = zarr.open_consolidated(zarr_path)
    num_cwb_channels = len(group['cwb_variable'])
    num_era5_channels = len(group['era5_variable'])

    print("Zarr Group Tree Structure:")
    print(group.tree())
    print("\nField Details:\n")

    # Iterate over the specified folders and dump the content
    for field_group, count in [
        ("cwb_center", num_cwb_channels),
        ("cwb_pressure", num_cwb_channels),
        ("cwb_scale", num_cwb_channels),
        ("cwb_variable", num_cwb_channels),
        ("era5_center", num_era5_channels),
        ("era5_pressure", num_era5_channels),
        ("era5_scale", num_era5_channels),
        ("era5_variable", num_era5_channels)
    ]:
        print(f"\n{field_group}:")
        print(group[field_group].info)
        for i in range(count):
            print(f"  Index {i}: {group[field_group][i]}")

    # Print general information about all groups
    # print("\nAll Groups Information:\n")
    # for folder in group:
    #     print(f"{folder}:")
    #     print(group[folder].info)

    print("\nZarr Dataset Structure:")
    ds = xr.open_zarr(zarr_path)
    print(ds)

    print_slices_over_time(ds)

def main():
    """
    Main function to parse positional argument for Zarr file path.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    zarr_path = sys.argv[1]  # First positional argument
    dump_zarr_fields(zarr_path)

if __name__ == "__main__":
    main()
