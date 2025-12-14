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
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import re

import xarray as xr
import pandas as pd


# Days per month, ignoring leap years
DAYS_PER_MONTH = {
    1: 31, 2: 28,  # always 28, as requested
    3: 31, 4: 30,
    5: 31, 6: 30,
    7: 31, 8: 31,
    9: 30, 10: 31,
    11: 30, 12: 31,
}

@dataclass
class HourState:
    last_hour: int | None = None
    last_path: Path | None = None


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


def load_times(ds: xr.Dataset, path: Path):
    """Return datetime64 index from 'time' or 'Times' coordinate."""
    if "time" in ds:
        src = "time"
        fmt = None
    elif "Times" in ds:
        src = "Times"
        fmt = "%Y-%m-%d_%H:%M:%S"
    else:
        print(f"[WARN] {path}: no 'time' or 'Times' coordinate found")
        return None

    try:
        strings = ds[src].astype(str)
        return pd.to_datetime(strings, format=fmt)
    except (ValueError, TypeError) as e:
        print(f"[ERROR] Failed converting time in {path}: {e}")
        return None


def check_file(path: Path, month: int, state: HourState):
    """Check time count for a single NetCDF file."""

    expected_days = DAYS_PER_MONTH[month]

    try:
        with xr.open_dataset(path) as ds:
            times = load_times(ds, path)
    except (OSError, IOError) as e:
        print(f"[ERROR] Cannot open {path}: {e}")
        return

    # Ensure `times`` is a non-empty list
    if times is None or len(times) == 0:
        print(f"[FAIL] {path.name} (month={month:02d}) -> no valid time values")
        return state

    # Date-level diagnosis
    n_time = len(times)
    dates = times.normalize()
    counts = Counter(dates)

    expected = {
        pd.Timestamp(year=dates[0].year, month=month, day=d)
        for d in range(1, expected_days + 1)
    }
    actual = set(counts)

    missing = expected - actual
    extra = actual - expected
    dup_dates = sum(c > 1 for c in counts.values())

    # Hour consistency
    hours = sorted({t.hour for t in times})

    issues = []
    if n_time != expected_days:
        issues.append(f"count={n_time}/{expected_days}")
    if missing:
        issues.append(f"missing={len(missing)}")
    if extra:
        issues.append(f"extra={len(extra)}")
    if dup_dates:
        issues.append(f"dup_dates={dup_dates}")
    if len(hours) > 1:
        issues.append(f"hours={hours}")

    # Cross-file hour consistency (only compare when this file itself is single-hour)
    if len(hours) == 1:
        hr = hours[0]
        if state.last_hour is not None and hr != state.last_hour:
            issues.append(
                f"hour_change={state.last_hour:02d}->{hr:02d} "
                f"(prev={(state.last_path.name if state.last_path else 'unknown')})"
            )
        state = HourState(last_hour=hr, last_path=path)

    # Print issues if not OK
    if issues:
        print(f"[FAIL] {path.name} (month={month:02d}) -> {" | ".join(issues)}")

    return state


def main():
    """Verify time count for NetCDF files under <top_folder>."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <top_folder>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    # Initial state
    state = HourState()
    last_dir = None
    files = sorted(find_nc_files(root))

    # Main loop to check files
    for path, month in files:
        curr_dir = path.parent
        if curr_dir != last_dir:
            print(f"\n=== {curr_dir} ===")
            last_dir = curr_dir

        state = check_file(path, month, state)

    if not files:
        print(f"No matching NetCDF files found under {root}")


if __name__ == "__main__":
    main()
