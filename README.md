# 📌 Overview
This project provides a set of tools to processing, merging, and exporting scientific datasets.. It includes:

- Data extraction and conversion from ERA5 and TReAD datasets
- Processing and regridding for dataset compatibility
- Exporting processed data into structured formats (Zarr, NetCDF)

# 📦 Installation

Before using the project, install the required dependencies:

```
conda env create -f corrdiff_input.yml
```

# 🚀 Usage

## 1️⃣ Generate Processed Datasets

Run the dataset generation script:

`python corrdiff_datagen.py <start_date> <end_date>`

Example

`python corrdiff_datagen.py 20180101 20180105`

- Fetches data from various sources
- Regrids datasets to match reference grid
- Saves output in Zarr format

### Dump data for debugging
Set `DEBUG=True` in `corrdiff_datagen.py` to dump ERA5 & TReAD pre- and post-regridding data into netcdf files.

```
DEBUG = True  # Set to True to enable debugging
```

### Generate > 8-year dataset

Run `datagen_n_merge.sh` to create multiple datasets and merge into one:

`./datagen_n_merge.sh <start_date> <end_date>`

The reason is to avoid OOM on BIG server given dataset with > 8-year time range.

## 2️⃣ Extract and Dump Zarr Data

Run `helpers/dump_zarr.py` to dump Zarr data and extract data slices:

`python helpers/dump_zarr.py <input_zarr_file>`

## 3️⃣ Generate REF grid file for output Zarr

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
 ┣ 📂 data/                   # Input data (NetCDF files of ERA5 and TReAD datasets)
   ┣ 📂 tread/
     ┗ 📜 wrfo2D_d02_201801.nc
   ┗ 📂 era5/
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
 ┣ 📂 ref_grid/
   ┣ 📜 generate_wrf_coord.py       # Generates REF grid
   ┣ 📜 TReAD_wrf_d02_info.nc       # TReAD grid used to generate REF grid
   ┗ 📜 wrf_208x208_grid_coords.nc  # Default 208 x 208 REF grid
 ┣ 📂 helpers/
   ┣ 📜 dump_zarr.py          # Zarr data extraction
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
  - Extracts and saves data slices from Zarr files

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
