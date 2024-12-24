import xesmf as xe
import xarray as xr

def regrid_dataset(ds, grid):
    # Regrid data over the spatial dimensions for all timestamps, based on CWA coordinates.
    remap = xe.Regridder(ds, grid, method="bilinear")

    print("Grid =>")
    print(grid)

    print("\nDataset =>")
    print(ds)

    print("\nBEFORE regrid_dataset =>")
    for t in range(1):
        print(ds.isel(time=t)['precipitation'].values[1][:10])

    # Regrid each time step while keeping the original coordinates and dimensions
    ds_regrid = xr.concat(
        [remap(ds.isel(time=i)).assign_coords(time=ds.time[i])
            for i in range(ds.sizes["time"])],
        dim="time"
    )

    print("\nAFTER regrid_dataset =>")
    for t in range(1):
        print(ds_regrid.isel(time=t)['precipitation'].values[1][:10])
    print("===")

    return ds_regrid

def print_slices_over_time(ds, limit=10):
    print("\n" + "-"*40)

    end = min(ds.time.size, limit)
    for t in ds.time[:end]:
        # Select the data for the current time step
        cwb_data = ds['cwb'].sel(time=t)
        era5_data = ds['era5'].sel(time=t)

        print(f"\nTime: {t.values}")
        print("CWB Data Slice:")
        print(cwb_data[0, 0].values[:20])
        print("ERA5 Data Slice:")
        print(era5_data[0, 0].values[:20])

    print("\n" + "-"*40 + "\n")