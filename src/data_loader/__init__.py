"""
Unified import surface for CorrDiff data loaders.

This package-level module re-exports the primary dataset constructors and
channel definitions for all data sources used in CorrDiff training and
evaluation. It provides a stable, compact API so downstream code can depend
on a single import location rather than individual loader implementations.

Supported data sources
----------------------
- TReAD (observations):
    * `get_tread_dataset`
    * `get_tread_channels`

- ERA5 (reanalysis):
    * `get_era5_dataset`
    * `get_era5_channels`

- TaiESM 3.5 km (regional climate model):
    * `get_taiesm3p5_dataset`
    * `get_taiesm3p5_channels`

- TaiESM 100 km (global climate model):
    * `get_taiesm100_dataset`
    * `get_taiesm100_channels`

- CORDEX (training datasets):
    * `get_cordex_train_datasets`
    * `get_cordex_test_datasets`
    * `get_cordex_train_hr_channels`
    * `get_cordex_train_lr_channels`

Example
-------
Importing from this module allows callers to rely on a unified interface:

    from data_loaders import (
        get_era5_dataset,
        get_tread_dataset,
        get_cordex_train_datasets,
    )

rather than importing from individual submodules.
"""
from .tread import get_tread_dataset, get_tread_channels
from .era5 import get_era5_dataset, get_era5_channels
from .taiesm3p5 import get_taiesm3p5_dataset, get_taiesm3p5_channels
from .taiesm100 import get_taiesm100_dataset, get_taiesm100_channels
from .cordex import (
    get_train_datasets as get_cordex_train_datasets,
    get_test_datasets as get_cordex_test_datasets,
    get_hr_channels as get_cordex_hr_channels,
    get_lr_channels as get_cordex_lr_channels
)


__all__ = [
    # CWA
    "get_tread_dataset",            "get_tread_channels",
    "get_era5_dataset",             "get_era5_channels",

    # SSP
    "get_taiesm3p5_dataset",        "get_taiesm3p5_channels",
    "get_taiesm100_dataset",        "get_taiesm100_channels",

    # Cordex
    "get_cordex_train_datasets",    "get_cordex_test_datasets",
    "get_cordex_hr_channels",       "get_cordex_lr_channels"
]
