#!/bin/bash

# Check arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage:"
    echo "  CWA / TReAD+ERA5 : $0 <start_date> <end_date>"
    echo "  SSP / TaiESM     : $0 <start_date> <end_date> <ssp_level>"
    exit 1
fi

START_DATE=$1
END_DATE=$2
SSP_LEVEL=${3:-""}

# Convert dates to year-only for interval calculation
START_YEAR=$(echo "$START_DATE" | cut -c1-4)
END_YEAR=$(echo "$END_DATE" | cut -c1-4)

INTERVAL=8
CURRENT_YEAR=$START_YEAR
cd src || exit 1

run_ssp_job() {
    local LEVEL=$1
    local current_year=$START_YEAR

    echo "=== Running SSP level [$LEVEL] from $START_DATE to $END_DATE ==="

    # Generate datasets for each 8-year interval
    while [ "$current_year" -le "$END_YEAR" ]; do
        local next_year=$((current_year + INTERVAL - 1))
        if [ "$next_year" -gt "$END_YEAR" ]; then
            next_year=$END_YEAR
        fi

        local interval_start_date=${current_year}0101
        local interval_end_date=${next_year}1231

        echo "Running corrdiff_datagen.py for $interval_start_date to $interval_end_date (SSP=$LEVEL) ..."

        python corrdiff_datagen.py "$interval_start_date" "$interval_end_date" "$LEVEL" || exit 1

        current_year=$((next_year + 1))
    done

    # Name merged dataset for this SSP level
    local merged_zarr="merged_dataset_${START_DATE}_${END_DATE}_${LEVEL}.zarr"

    echo "Merging all datasets into [$merged_zarr] ..."
    python helpers/merge_zarr.py || exit 1
    mv combined.zarr "$merged_zarr"

    # Move all zarrs for this run into ../<SSP_LEVEL>
    local dest_dir="../$LEVEL"
    mkdir -p "$dest_dir"
    mv ./*.zarr "$dest_dir"/
    echo "Moved all .zarr files to $dest_dir"
}

run_cwa_job() {
    local current_year=$START_YEAR

    echo "=== Running CWA mode from $START_DATE to $END_DATE ==="

    # Generate datasets for each 8-year interval
    while [ "$current_year" -le "$END_YEAR" ]; do
        local next_year=$((current_year + INTERVAL - 1))
        if [ "$next_year" -gt "$END_YEAR" ]; then
            next_year=$END_YEAR
        fi

        local interval_start_date=${current_year}0101
        local interval_end_date=${next_year}1231

        echo "Running corrdiff_datagen.py for $interval_start_date to $interval_end_date ..."
        python corrdiff_datagen.py "$interval_start_date" "$interval_end_date" || exit 1

        current_year=$((next_year + 1))
    done

    # Name merged dataset (no SSP level)
    local merged_zarr="merged_dataset_${START_DATE}_${END_DATE}.zarr"

    echo "Merging all datasets into [$merged_zarr] ..."
    python helpers/merge_zarr.py || exit 1
    mv combined.zarr "$merged_zarr"
}

if [ -z "$SSP_LEVEL" ]; then
    # CWA / TReAD+ERA5 mode
    run_cwa_job
else
    # SSP / TaiESM mode
    if [ "$SSP_LEVEL" = "all" ]; then
        # Iterate through all configured SSP levels
        for level in ssp126 ssp245 ssp370 ssp585; do
            run_ssp_job "$level"
        done
    else
        # Single specified SSP level
        run_ssp_job "$SSP_LEVEL"
    fi
fi
