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
SSP_LEVEL=""

if [ "$#" -eq 3 ]; then
    SSP_LEVEL=$3
fi

# Convert dates to year-only for interval calculation
START_YEAR=$(echo "$START_DATE" | cut -c1-4)
END_YEAR=$(echo "$END_DATE" | cut -c1-4)

INTERVAL=8
CURRENT_YEAR=$START_YEAR
cd src || exit 1

# Generate datasets for each 8-year interval
while [ "$CURRENT_YEAR" -le "$END_YEAR" ]; do
    NEXT_YEAR=$((CURRENT_YEAR + INTERVAL - 1))
    if [ "$NEXT_YEAR" -gt "$END_YEAR" ]; then
        NEXT_YEAR=$END_YEAR
    fi

    INTERVAL_START_DATE=${CURRENT_YEAR}0101
    INTERVAL_END_DATE=${NEXT_YEAR}1231

    echo "Running corrdiff_datagen.py for $INTERVAL_START_DATE to $INTERVAL_END_DATE ..."

    if [ -z "$SSP_LEVEL" ]; then
        # CWA / TReAD+ERA5 mode
        python corrdiff_datagen.py "$INTERVAL_START_DATE" "$INTERVAL_END_DATE"
    else
        # SSP / TaiESM mode
        python corrdiff_datagen.py "$INTERVAL_START_DATE" "$INTERVAL_END_DATE" "$SSP_LEVEL"
    fi

    CURRENT_YEAR=$((NEXT_YEAR + 1))
done

# Name merged dataset (include SSP level if provided)
if [ -z "$SSP_LEVEL" ]; then
    MERGED_ZARR="merged_dataset_${START_DATE}_${END_DATE}.zarr"
else
    MERGED_ZARR="merged_dataset_${START_DATE}_${END_DATE}_${SSP_LEVEL}.zarr"
fi

echo "Merging all datasets into [$MERGED_ZARR] ..."

python helpers/merge_zarr.py
mv combined.zarr "$MERGED_ZARR"
