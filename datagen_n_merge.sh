#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_date> <end_date>"
    exit 1
fi

START_DATE=$1
END_DATE=$2

# Convert dates to year-only for interval calculation
START_YEAR=$(echo $START_DATE | cut -c1-4)
END_YEAR=$(echo $END_DATE | cut -c1-4)

INTERVAL=8
CURRENT_YEAR=$START_YEAR

# Generate datasets for each 8-year interval
while [ $CURRENT_YEAR -le $END_YEAR ]; do
    NEXT_YEAR=$((CURRENT_YEAR + INTERVAL - 1))
    if [ $NEXT_YEAR -gt $END_YEAR ]; then
        NEXT_YEAR=$END_YEAR
    fi

    INTERVAL_START_DATE=${CURRENT_YEAR}0101
    INTERVAL_END_DATE=${NEXT_YEAR}1231

    echo "Running datagen.py for $INTERVAL_START_DATE to $INTERVAL_END_DATE ..."
    python corrdiff_datagen.py $INTERVAL_START_DATE $INTERVAL_END_DATE

    CURRENT_YEAR=$((NEXT_YEAR + 1))
done

# Merge all generated datasets
MERGED_ZARR=merged_dataset_${START_DATE}_${END_DATE}.zarr
echo "Merging all datasets into [$MERGED_ZARR] ..."

cd helpers
mv ../corrdiff*.zarr .
python merge_zarr.py
mv combined.zarr $MERGED_ZARR