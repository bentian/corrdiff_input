# Run the code

## Set up environment via conda

```
conda env create -f corrdiff_input.yml
```

## Generate CorrDiff dataset

```
python corrdiff_datagen.py <start_date> <end_date>
```
Example: `python corrdiff_datagen.py 20180101 20180105`

## Dump zarr file

```
python zarr_dump.py <zarr_file>
```
Example: `python zarr_dump.py corrdiff_dataset_test.zarr`


## Run tests

```
python -m unittest <test_file>
```
Example: `python -m unittest test_corrdiff_datagen.py`

# Files

## Code
- `corrdiff_datagen.py`: Main functions that take start & end dates, generate output dataset, and write to zarr file.
- `tread.py`: Functions that handle TReAD data.
- `era5.py`: Functions that handle ERA5 data.
- `zarr_dump.py`: Function to dump the generated zarr file.
- `test_*.py`: Test files generated by ChatGPT.

## Data
- `data/cwa_dataset_example.zarr`: Sample dataset based on original CorrDiff dataset, for grid and coordinates reference.
- `archive/*`: initial version of python code by Lama, and the channel list accordingly.
