"""
Zarr Dataset Merging and Preprocessing Script

This module merges multiple Zarr datasets containing atmospheric and meteorological 
variables, processes key statistics, and saves the combined dataset as a new Zarr file.

Functionality:
--------------
- Loads multiple Zarr files (`corrdiff_*.zarr`) from the current directory.
- Concatenates datasets along the `time` dimension.
- Computes statistical aggregations for ERA5 and CWB data:
    - `era5_center`: Mean values of ERA5 variables across time and spatial dimensions.
    - `era5_scale`: Standard deviation of ERA5 variables across time and spatial dimensions.
    - `cwb_center`: Mean values of CWB variables across time and spatial dimensions.
    - `cwb_scale`: Standard deviation of CWB variables across time and spatial dimensions.
- Generates validity indicators (`cwb_valid`, `era5_valid`) marking all time steps as valid.
- Drops the `era5_channel` variable for simplification.
- Saves the final processed dataset as `combined.zarr`.

Dependencies:
-------------
- `xarray` for dataset manipulation.
- `dask.array` for handling large datasets efficiently.
- `glob` for file discovery.

Output:
-------
- The processed dataset is stored in `./combined.zarr`.
"""
import glob
import xarray as xr
import dask.array as da

def recompute_fields(ds: xr.Dataset) -> xr.Dataset:
    """
    Recomputes normalization statistics (center and scale) for ERA5 and CWB datasets
    and updates validity indicators.

    Parameters:
        ds (xr.Dataset):
            An xarray Dataset containing:
            - "era5" (time, era5_channel, south_north, west_east): ERA5 data
            - "cwb" (time, cwb_channel, south_north, west_east): CWB data
            - "era5_channel" (era5_channel): List of ERA5 channels
            - "era5_pressure" (era5_channel): Corresponding ERA5 pressures
            - "era5_variable" (era5_channel): Corresponding ERA5 variable names
            - "time" (time): Time coordinate

    Returns:
        xr.Dataset:
            - Updated dataset with:
                - **"era5_center"**: Mean of ERA5 data over time and spatial dimensions.
                - **"era5_scale"**: Standard deviation of ERA5 data over time and space.
                - **"cwb_center"**: Mean of CWB data over time and space.
                - **"cwb_scale"**: Standard deviation of CWB data over time and space.
                - **"cwb_valid"**: Boolean array indicating valid CWB data for each time step.
                - **"era5_valid"**: Boolean array indicating valid ERA5 data
                                    per time step and channel.
            - The dataset **drops the "era5_channel" variable** after recomputation.

    Notes:
        - The validity indicators (`cwb_valid` and `era5_valid`) are set to `True` for all entries.
        - `era5_valid` maintains ERA5 metadata (`era5_pressure`, `era5_variable`).
    """
    # Recompute era5_center and era5_scale
    ds['era5_center'] = ds['era5'].mean(dim=['time', 'south_north', 'west_east'])
    ds['era5_scale'] = ds['era5'].std(dim=['time', 'south_north', 'west_east'])

    # Recompute cwb_center and cwb_scale
    ds['cwb_center'] = ds['cwb'].mean(dim=['time', 'south_north', 'west_east'])
    ds['cwb_scale'] = ds['cwb'].std(dim=['time', 'south_north', 'west_east'])

    # Compute validity indicators
    ds['cwb_valid'] = xr.DataArray(
        data=da.ones(len(ds["time"]), dtype=bool, chunks=len(ds["time"])),
        dims=["time"],
        coords={"time": ds["time"]},
        name="cwb_valid"
    )

    ds['era5_valid'] = xr.DataArray(
        data=da.ones((len(ds["time"]), len(ds["era5_channel"])), dtype=bool,
                     chunks=(len(ds["time"]), len(ds["era5_channel"]))),
        dims=["time", "era5_channel"],
        coords={
            "time": ds["time"],
            "era5_channel": ds["era5_channel"],
            "era5_pressure": ds["era5_pressure"],
            "era5_variable": ds["era5_variable"]
        },
        name="era5_valid"
    )

    return ds.drop_vars(["era5_channel"])

def main():
    """
    Processes and combines multiple Zarr datasets along the time dimension.

    Workflow:
        1. **List & Sort Zarr Files:**
           - Finds all files matching the pattern `./corrdiff_*.zarr`.
           - Sorts them to ensure correct time order.
        2. **Load Datasets with Dask:**
           - Opens each Zarr dataset with `xarray.open_dataset()`, enabling lazy loading.
        3. **Concatenate Along Time:**
           - Merges datasets along the `"time"` dimension using `xr.concat()`.
        4. **Recompute Derived Fields:**
           - Calls `recompute_fields()` to update dataset statistics
             (e.g., means, scales, validity flags).
        5. **Save to a New Zarr File:**
           - Writes the combined dataset to `./combined.zarr`.

    Outputs:
        - A new combined Zarr dataset stored as `./combined.zarr`.

    Notes:
        - Uses **Dask** to handle large datasets efficiently.
        - Assumes all Zarr files have **matching dimensions and variables**.
        - The function prints the merged dataset for verification.
    """
    # List and sort Zarr files
    zarr_files = sorted(glob.glob('./corrdiff_*.zarr'))
    for zarr_file in zarr_files:
        print(zarr_file)

    # Open datasets with Dask
    datasets = [xr.open_dataset(zarr_file, engine='zarr', chunks={}) for zarr_file in zarr_files]

    # Concatenate datasets along the time dimension
    combined = xr.concat(datasets, dim='time')

    # Recompute *_center, *_scale, *_valid fields
    combined = recompute_fields(combined)
    print(combined)

    # Save the combined dataset to a new Zarr file
    combined.to_zarr('./combined.zarr')

if __name__ == "__main__":
    main()
