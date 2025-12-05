"""
Unified import surface for CorrDiff data loaders.

This package-level module re-exports the main dataset and channel
constructors for all supported sources:

- TReAD:
    * `get_tread_dataset`
    * `get_tread_channels`
- ERA5:
    * `get_era5_dataset`
    * `get_era5_channels`
- TaiESM 3.5 km:
    * `get_taiesm3p5_dataset`
    * `get_taiesm3p5_channels`
- TaiESM 100 km:
    * `get_taiesm100_dataset`
    * `get_taiesm100_channels`

Importing from this module allows callers to depend on a stable, compact
API instead of individual loader modules, for example:

    from data_loaders import get_era5_dataset, get_tread_dataset
"""
from .tread import get_tread_dataset, get_tread_channels
from .era5 import get_era5_dataset, get_era5_channels
from .taiesm3p5 import get_taiesm3p5_dataset, get_taiesm3p5_channels
from .taiesm100 import get_taiesm100_dataset, get_taiesm100_channels

__all__ = [
    "get_tread_dataset",        "get_tread_channels",
    "get_era5_dataset",         "get_era5_channels",
    "get_taiesm3p5_dataset",    "get_taiesm3p5_channels",
    "get_taiesm100_dataset",    "get_taiesm100_channels",
]
