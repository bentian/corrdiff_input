# 📌 Overview

`corrdiff_input` is a data–preparation pipeline for constructing CorrDiff-ready Zarr datasets from multiple atmospheric data sources, including:
- ERA5 (pressure-level & surface-level fields)
- TReAD (CWB Taiwan reanalysis)
- TaiESM 3.5 km / TaiESM 100 km (SSP climate scenarios)

The pipeline:
1. Loads raw NetCDF files
2. Regrids all datasets onto a unified WRF-style reference grid
3. Normalizes, centers & stacks channels into CorrDiff-compatible tensors
4. Outputs the consolidated dataset as a compressed Zarr store

# 🗂 Features
✅ Unified Reference Grid

Creates or loads a consistent **208×208 WRF-style grid**.

✅ Multi-source Dataset Loading

Supports:
- ERA5 surface (SFC)
- ERA5 pressure-level (PRS)
- TReAD high-resolution fields
- TaiESM 3.5 km / 100 km (SSP scenarios)

✅ Regridding & Spatial Alignment

Built-in utilities for:
- Bilinear interpolation
- Nearest-cell extrapolation
- Optional custom grid generation

✅ CorrDiff-Ready Tensor Construction

Creates:
- `cwb_*` (TReAD / TaiESM 3.5km)
- `era5_*` (ERA5 / TaiESM 100km)
- Associated metadata: center, scale, variable names, masks

✅ Zarr Export & Validation

- Outputs compressed Zarr datasets
- Tools for merging, slicing, and inspecting Zarr stores


# 📦 Installation

Before using the project, install the required dependencies:

```
conda env create -f yml/corrdiff_input.yml
```

# 🚀 Usage

## 1️⃣ Generate a Single CorrDiff Dataset

- CWA mode (TReAD + ERA5):
  ```
  python src/corrdiff_datagen.py <start_date> <end_date>
  ```
  Example: `python src/corrdiff_datagen.py 20180101 20180131`

- SSP mode (TaiESM 3.5 km + 100 km):
  ```
  python src/corrdiff_datagen.py <start_date> <end_date> <ssp_level>
  ```
  Example: `python src/corrdiff_datagen.py 20180101 20180131 ssp585`

  This will:
  - Load and regrid the input datasets
  - Construct CorrDiff-formatted tensors
  - Saves output to
    ```
    corrdiff_dataset_<start_date>_<end_date>.zarr
    # or
    corrdiff_dataset_<start_date>_<end_date>_<ssp_level>.zarr
    ```

### 🔍 Debugging Regridding Artifacts
Enable NetCDF dumps in `src/corrdiff_datagen.py`:
```
DEBUG = True  # Set to True to enable debugging
```

This writes:
```
nc_dump/<start_date>_<end_date>/
   highres_pre_regrid.nc
   highres_post_regrid.nc
   lowres_pre_regrid.nc
   lowres_post_regrid.nc
```

## 2️⃣ Generate Multi-Year Datasets (avoid OOM)

For long time ranges (> 8 years):
```
./datagen_n_merge.sh <start_date> <end_date>
```

The script:
- Splits the time range into periods <= 8 years
- Generates Zarr files per period
- Merges them into a single consolidated Zarr store

The reason is to avoid OOM on BIG server given dataset with > 8-year time range.

## 3️⃣ Inspect or Slice Zarr Files

### Inspect structure and preview slices:

```
python src/helpers/dump_zarr.py <input_zarr_file>
```

### Filter variables / time ranges:

_Revise file paths in `src/helpers/filter_zarr.py` and run_
```
python src/helpers/filer_zarr.py
```

### Merge many Zarr files:

_Put all `corrdiff_*.zarr` under `src/helpers/` and run_
```
python src/helpers/merge_zarr.py
```

## 4️⃣ Generate Reference Grid

### Create a new 208×208 WRF-style grid:
```
cd ref_grid
python generate_wrf_coord.py
```

### Adjust grid size:

1. Modify `ny` and `nx` in `generate_wrf_coord.py`:

```
ny, nx = 208, 208               # Desired grid dimensions
```

2. Revise `*_REF_GRID` in `src/data_builder.py` acccordingly:

```
CWA_REF_GRID = "../ref_grid/wrf_208x208_grid_coords.nc"
SSP_REF_GRID = "../ref_grid/ssp_208x208_grid_coords.nc"
```

# 📂 Project Structure

## Prepare NetCDF files (for local testing only)

### CWA mode
- Put TReAD file below under `data/tread/`.
  - `wrfo2D_d02_{yyyymm}.nc`
- Put ERA5 files below under `data/era5/`.
  - `ERA5_PRS_*_{yyyymm}_r1440x721_day.nc`
  - `ERA5_SFC_*_{yyyymm}_r1440x721_day.nc`

### SSP mode
- Put TaiESM 3.5 km file below under `data/taiesm3p5/`.
  - `wrfday_d01_201801.nc`
- Put TaiESM 100 km below under `data/taiesm100/`.
  - `TaiESM1_SFC_*_201801_r1440x721_day.nc`
  - `TaiESM1_PRS_*_201801_r1440x721_day.nc`

## Example
```
📦 corrdiff_input
 ┣ 📂 yml/      # YML files to create conda environment
 ┣ 📂 data/     # Input data (NetCDF files of low- and high-resolution datasets)
   ┣ 📂 tread/
     ┗ 📜 wrfo2D_d02_201801.nc
   ┣ 📂 era5/
     ┣ 📜 ERA5_SFC_*_201801_r1440x721_day.nc
     ┗ 📜 ERA5_PRS_*_201801_r1440x721_day.nc
   ┣ 📂 taiesm3p5/
     ┗ 📜 wrfday_d01_201801.nc
   ┣ 📂 taiesm100/
     ┣ 📜 TaiESM1_SFC_*_201801_r1440x721_day.nc
     ┗ 📜 TaiESM1_PRS_*_201801_r1440x721_day.nc
   ┗ 📂 extreme_dates/
     ┣ 📜 extreme_dates.txt
     ┗ 📜 extreme_dates_histogram.png
 ┣ 📂 ref_grid/
   ┣ 📜 generate_wrf_coord.py       # REF grid generation script
   ┣ 📜 TReAD_wrf_d02_info.nc       # TReAD grid used to generate REF grid
   ┣ 📜 wrf_208x208_grid_coords.nc  # CWA 208x208 REF grid
   ┣ 📜 TAIESM_tw3.5km_coord2d.nc   # TaiESM 3.5 km grid used to generate REF grid
   ┗ 📜 ssp_208x208_grid_coords.nc  # SSP 208x208 REF grid
 ┣ 📂 src/
   ┣ 📂 helpers/              # Zarr utilities & Geographic region plotting
   ┣ 📂 data_loader/          # TReAD / ERA5 / TaiESM data loaders
   ┣ 📜 corrdiff_datagen.py   # Zarr generation script
   ┣ 📜 data_builder.py       # Low- and high-resolution dataset builder script
   ┗ 📜 tensor_fields.py      # Tensor fields constructor for Zarr
 ┣ 📜 datagen_n_merge.sh      # Shell script to create and merge datasets
 ┗ 📜 README.md               # Project documentation
```

📘 Script Overview

🔹 `src/corrdiff_datagen.py`

Main driver:
- Loads datasets
- Regrids to REF grid
- Builds CorrDiff-ready tensors
- Saves to Zarr

🔹 `src/data_builder.py`

CorrDiff data assembly module:
- Loads REF grid
- Builds low- and high-resolution datasets
- Converts datasets into CorrDiff-ready tensors

🔹 `src/tensor_fields.py`

Tensor creation logic:
- Stacks channels
- Normalizes data (center/scale)
- Creates metadata

🔹 `src/data_loaders/`

- Loads datasets for:
  - TReAD / ERA5
  - TaiESM 3.5 km / 100 km
- Provides utilities:
  - Regridding (xesmf)
  - File data format validation

🔹 `src/helpers/`

Post-processing & debugging:
- Inspects single Zarr file
- Merges multiple Zarr files
- Filters data in Zarr by dates
- Plots geographic region

🔹 `ref_grid/`

WRF-style reference grid creation.
- Allows easy grid size customization (e.g., 208x208 -> any `ny` x `nx`)
- Supports CWA and SSP modes for different configurations

# 🎯 Why Use This Project?

- ✅ Automates multi-source climate dataset preparation
- ✅ Ensures spatial & temporal consistency
- ✅ Outputs CorrDiff-ready tensors
- ✅ Handles large datasets via Dask & Zarr
- ✅ Provides debugging & inspection tools
- ✅ Extensible to new climate models

# 🤝 Contributing

PRs and issues are welcome! If integrating new datasets or grids, please document them clearly.
