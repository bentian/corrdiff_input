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

- Basic usage (TReAD + ERA5 mode):

  `python src/corrdiff_datagen.py <start_date> <end_date>`

  Example: `python src/corrdiff_datagen.py 20180101 20180131`

- SSP mode (TaiESM 3.5 km + 100 km):

  `python src/corrdiff_datagen.py <start_date> <end_date> <ssp_level>`

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

`./datagen_n_merge.sh <start_date> <end_date>`

The script:
- Splits the time range into periods <= 8 years
- Generates Zarr files per period
- Merges them into a single consolidated Zarr store

The reason is to avoid OOM on BIG server given dataset with > 8-year time range.

## 3️⃣ Inspect or Slice Zarr Files

Inspect structure and preview slices:

```
python src/helpers/dump_zarr.py <input_zarr_file>
```

Filter variables / time ranges:

Revise `src/helpers/filter_zarr.py` and run
```
python src/helpers/filer_zarr.py
```

Run `ref_grid/generate_wrf_coord.py` to generate REF grid for regridding ERA5 and TReAD datasets:

```
cd ref_grid
python generate_wrf_coord.py
```

### Adjust REF grid size

1. Modify `ny` and `nx` in `generate_wrf_coord.py` to customize REF grid size:

```
ny, nx = 208, 208               # Desired grid dimensions
```

2. Revise `REF_GRID_NC` in `corrdiff_datagen.py` acccordingly:

```
REF_GRID_NC = "./ref_grid/wrf_208x208_grid_coords.nc"
```

# 📂 Project Structure

## Prepare TReAD and ERA5 netcdf files (local testing only)
- Put TReAD file below under `data/tread/`.
  - `wrfo2D_d02_{yyyymm}.nc`
- Put ERA5 files below under `data/era5/`.
  - `ERA5_PRS_*_{yyyymm}_r1440x721_day.nc`
  - `ERA5_SFC_*_{yyyymm}_r1440x721_day.nc`

## Example

```
📦 corrdiff_input
 ┣ 📂 data/                  # Input data (NetCDF files of ERA5 and TReAD datasets)
   ┣ 📂 tread/
     ┗ 📜 wrfo2D_d02_201801.nc
   ┣ 📂 era5/
     ┣ 📜 ERA5_PRS_q_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_r_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_t_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_u_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_v_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_w_201801_r1440x721_day.nc
     ┣ 📜 ERA5_PRS_z_201801_r1440x721_day.nc
     ┣ 📜 ERA5_SFC_msl_201801_r1440x721_day.nc
     ┣ 📜 ERA5_SFC_t2m_201801_r1440x721_day.nc
     ┣ 📜 ERA5_SFC_tp_201801_r1440x721_day.nc
     ┣ 📜 ERA5_SFC_u10_201801_r1440x721_day.nc
     ┗ 📜 ERA5_SFC_v10_201801_r1440x721_day.nc
   ┗ 📂 extreme_dates/
     ┣ 📜 extreme_dates.txt
     ┗ 📜 extreme_dates_histogram.png
 ┣ 📂 ref_grid/
   ┣ 📜 generate_wrf_coord.py       # Generates REF grid
   ┣ 📜 TReAD_wrf_d02_info.nc       # TReAD grid used to generate REF grid
   ┗ 📜 wrf_208x208_grid_coords.nc  # Default 208 x 208 REF grid
 ┣ 📂 helpers/
   ┣ 📜 dump_zarr.py          # Zarr data inspection
   ┣ 📜 filter_zarr.py        # Zarr data extraction
   ┗ 📜 merge_zarr.py         # Zarr file combination
 ┣ 📜 corrdiff_datagen.py     # Dataset generation script
 ┣ 📜 era5.py                 # ERA5 data processing
 ┣ 📜 tread.py                # TReAD data processing
 ┣ 📜 util.py                 # Utility functions for data transformation
 ┣ 📜 datagen_n_merge.sh      # Shell script to create and merge datasets
 ┗ 📜 README.md               # Project documentation
```

📜 Script Descriptions

🔹 `corrdiff_datagen.py` - Generate Processed Datasets
  - Fetches datasets from multiple sources
  - Regrids them to match a common grid
  - Saves final dataset in Zarr format

🔹 `era5.py` - ERA5 Data Processing
  - Loads ERA5 dataset
  - Performs regridding and data aggregation
  - Computes mean, standard deviation, and validity

🔹 `tread.py` - TReAD Data Processing
  - Loads TReAD dataset
  - Computes daily aggregated variables
  - Regrids dataset for analysis

🔹 `util.py` - General Utilities
  - Provides data transformation, regridding, and verification utilities

🔹 `datagen_n_merge.sh` - Create and Merge Datasets
  - Splits time range by 8-year interval, creates datasets per interval, and merges them into one

🔹 `helpers/dump_zarr.py` - Inspect Zarr Datasets
  - Inspects structure and data slices in Zarr files.

🔹 `helpers/filter_zarr.py` - Extract Zarr Datasets
  - Extracts data slices from one Zarr file and saves to another.

🔹 `helpers/merge_zarr.py` - Combine Zarr Datasets
  - Combines and saves multiple Zarr files into one Zarr file.

🔹 `ref_grid/generate_wrf_coord.py` - Extract Grid Coordinates
  - Extracts and saves grid coordinates from datasets

# 🎯 Why Use This Project?

- ✅ Automates Data Extraction and Processing
- ✅ Generates Ready-to-Use Datasets
- ✅ Handles Large NetCDF & Zarr Datasets Efficiently
- ✅ Supports Regridding, Verification, and Data Export

# ⚡ Contributing

Feel free to submit pull requests or open issues for improvements!
