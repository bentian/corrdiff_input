"""
NetCDF monthly time-axis checker.

This script walks a directory tree, finds NetCDF files whose names contain a
`YYYYMM` month token (e.g.
    TaiESM1_ssp126_r1i1p1f1_t_EA_201501_day.nc
    TaiESM1-WRF_tw3.5_ssp126_wrfday_d01_201501.nc
), and verifies that the `time` dimension has the expected number of daily
entries for that calendar month.

Rules / assumptions:
- Days per month are fixed by a simple table and DO NOT consider leap years
  (i.e. February is always expected to have 28 days, even in leap years).
- A valid file name must contain a 4-digit year followed immediately by a
  2-digit month (YYYYMM), and end with either:
    * "...YYYYMM.nc"          or
    * "...YYYYMM_day.nc"
- The NetCDF file must contain a `time` coordinate/variable. If it contains
  `Times` instead, this script will rename `Times` -> `time`.
- Time values are expected to be strings with the format "%Y-%m-%d_%H:%M:%S".
  These are converted to pandas Timestamps for checking.

For each file the script:
1. Counts the number of time steps and compares it with the expected number of
   days in that month.
2. On mismatch, it diagnoses the problem by:
   - Normalizing timestamps to dates.
   - Listing missing dates in that month (if any).
   - Listing extra dates not belonging to that month (if any).
   - Listing duplicate dates (same date appearing more than once) with counts.

Usage:
    python check_nc_time_counts.py <top_folder>

Arguments:
    <top_folder>  Top-level directory under which all *.nc files are searched.

Exit codes:
    0  Success (script ran; individual files may still fail checks).
    1  Invalid arguments or <top_folder> is not a directory.
"""
import sys
from pathlib import Path
import re
from collections import Counter

import xarray as xr
import pandas as pd


# Days per month, ignoring leap years
DAYS_PER_MONTH = {
    1: 31,
    2: 28,  # always 28, as requested
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def find_nc_files(root: Path):
    """Yield all *.nc files whose name ends with MM.nc where 01 <= MM <= 12."""
    pattern = re.compile(r".*?(\d{4})(\d{2})(?=(_day)?\.nc$)")
    for path in root.rglob("*.nc"):
        m = pattern.match(path.name)
        if not m:
            continue
        mm = int(m.group(2))
        if 1 <= mm <= 12:
            yield path, mm


def check_file(path: Path, month: int):
    """Check that a NetCDF file has the expected number of daily time steps for its month."""

    expected_days = DAYS_PER_MONTH[month]

    try:
        with xr.open_dataset(path) as ds:
            # Choose time source
            src = "time" if "time" in ds else "Times" if "Times" in ds else None
            if src is None:
                print(f"[WARN] {path}: no 'time' or 'Times' coordinate/variable found")
                return

            fmt = "%Y-%m-%d_%H:%M:%S" if src == "Times" else None
            times = pd.to_datetime(ds[src].astype(str), format=fmt)
    except Exception as e:
        print(f"[ERROR] Cannot open {path}: {e}")
        return

    # Assume daily data; each entry is one day
    n_time = len(times)
    if n_time == expected_days:
        print(f"[OK]   {path}  month={month:02d}  time_count={n_time}")
        return

    print(f"[FAIL] {path}  month={month:02d}  time_count={n_time}, expected={expected_days}")

    # Diagnose missing / duplicate dates (use normalized dates, ignore hour)
    dates = times.normalize()
    date_counter = Counter(dates)
    if not date_counter:
        print("       No valid datetime values parsed.")
        return

    unique_dates = sorted(date_counter.keys())
    year = unique_dates[0].year  # assume single month, single year

    expected_dates = [
        pd.Timestamp(year=year, month=month, day=d)
        for d in range(1, expected_days + 1)
    ]

    expected_set = set(expected_dates)
    actual_set = set(unique_dates)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    duplicates = [d for d, c in date_counter.items() if c > 1]

    if missing:
        print("       Missing dates:")
        for d in missing:
            print("         ", d.strftime("%Y-%m-%d"))

    if extra:
        print("       Extra dates (not expected for that month):")
        for d in extra:
            print("         ", d.strftime("%Y-%m-%d"))

    if duplicates:
        print("       Duplicate dates (appear more than once):")
        for d in duplicates:
            print(f"         {d.strftime('%Y-%m-%d')} (count={date_counter[d]})")

    print()


def main():
    """Verify time count for NetCDF files under <top_folder>."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <top_folder>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    any_files = False
    for path, month in sorted(find_nc_files(root)):
        any_files = True
        check_file(path, month)

    if not any_files:
        print("No matching NetCDF files found under", root)


if __name__ == "__main__":
    main()
