"""
NetCDF Metrics Processing and CSV Export.

This module processes meteorological metrics from a NetCDF dataset, computes statistical summaries,
and exports the results to a CSV file. The main functionality includes calculating mean values of
metrics and grouping metrics by month.

Features:
- Renames standard meteorological variables for easier processing.
- Computes the mean values for all metrics across the time dimension.
- Groups metrics by month and computes monthly means.
- Outputs results as a CSV file, appending grouped data to the same file.

Functions:
- process_metrics: Processes metrics from the input NetCDF file and writes results to a CSV file.

Command-Line Usage:
    python script.py <input_file> <output_file>

    Example:
        python script.py ./input.nc ./output.csv

Dependencies:
- `sys`: For command-line argument handling.
- `xarray`: For handling NetCDF datasets.
- `pandas`: Used internally for CSV export and DataFrame operations.

Notes:
- Ensure that the input file exists and contains valid NetCDF data.
- The script renames specific variables in the dataset to standardized names.
- The output CSV file is overwritten for mean metrics and appended for grouped metrics.
"""
import sys
import xarray as xr

def process_metrics(input_file, output_file):
    """
    Processes the metrics from the input NetCDF file and writes results to a CSV file.

    Parameters:
        input_file (str): Path to the input NetCDF file.
        output_file (str): Path to the output CSV file.
    """
    # Load the dataset
    ds = xr.open_dataset(input_file, engine='netcdf4').rename({
        # Baseline
        "precipitation" : "prcp",
        "temperature_2m" : "t2m",
        "eastward_wind_10m" : "u10m",
        "northward_wind_10m": "v10m",
        # "windspeed_10m": "uv10m",
        # "relative_humidity_2m" : "rh2m",
        # "sea_level_pressure": "slp",
        # "maximum_temperature_2m": "t2max",
        # "minimum_temperature_2m": "t2min",
    })

    # 1. Compute the mean of all 4 metrics
    metric_mean = ds.mean(dim="time")
    df_metric_mean = metric_mean.to_dataframe().round(2)

    # Print and save metric mean
    print("\nMetric Mean =>")
    print(df_metric_mean)
    df_metric_mean.to_csv(output_file, float_format="%.2f")

    # 2. Group all metrics by month and compute mean
    ds_grouped = ds.groupby("time.month").mean(dim="time")
    df_grouped_mae = ds_grouped.sel(metric="mae").to_dataframe()
    df_grouped_rmse = ds_grouped.sel(metric="rmse").to_dataframe()

    # Print grouped metrics
    print("\nGrouped by Month =>")
    print(df_grouped_mae)
    print(df_grouped_rmse)

    # Append grouped metrics to the output file
    df_grouped_mae.to_csv(output_file, mode='a', float_format="%.2f")
    df_grouped_rmse.to_csv(output_file, mode='a', float_format="%.2f")

if __name__ == "__main__":
    # Check if the required arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    # Process metrics
    process_metrics(sys.argv[1], sys.argv[2])
