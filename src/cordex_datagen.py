"""
CorrDiff CORDEX Zarr dataset generator.

Builds consolidated CorrDiff-ready datasets from CORDEX HR/LR sources (train + test),
optionally dumps intermediate regrid NetCDFs for debugging, verifies schema, and
writes the final result to Zarr.
"""
from itertools import product
import numpy as np
import xarray as xr

from corrdiff_datagen import verify_dataset, write_to_zarr, dump_regrid_netcdf
from data_builder import \
    GRID_COORD_KEYS, generate_cordex_train_outputs, generate_cordex_test_outputs

DEBUG = False                                       # Set to True to enable debugging
XTIME = np.datetime64("2026-01-09 17:00:00", "ns")  # placeholder timestamp


def build_out(hr_outputs, lr_outputs, grid_coords, tag: str) -> xr.Dataset:
    """
    Assemble the final CorrDiff output dataset from HR and LR components.

    This function combines preprocessed high-resolution (HR) and low-resolution (LR)
    outputs into a single xarray.Dataset that conforms to the CorrDiff training
    and evaluation schema. It merges normalized data variables, attaches shared
    grid coordinates, and drops intermediate dimensions not required by the
    CorrDiff model.

    Parameters
    ----------
    hr_outputs : tuple
        Tuple of HR outputs in CorrDiff order, containing:
        (fields, variable metadata, normalization center, normalization scale,
         validity mask, pre-regrid dataset, post-regrid dataset).
    lr_outputs : tuple
        Tuple of LR outputs in CorrDiff order, containing:
        (fields, normalization center, normalization scale, validity mask,
         pre-regrid dataset, post-regrid dataset).
    grid_coords : xr.Dataset
        Dataset containing grid coordinate arrays (e.g., XLAT, XLONG) defining
        the spatial domain.
    tag : str
        Identifier used for optional debugging output (e.g., NetCDF dumps).

    Returns
    -------
    xr.Dataset
        Consolidated CorrDiff dataset containing HR and LR variables, coordinates,
        and metadata, ready for validation and serialization.
    """
    hr_keys = ["cwb", "cwb_variable", "cwb_center", "cwb_scale", "cwb_valid",
               "pre_regrid", "post_regrid"]
    lr_keys = ["era5", "era5_center", "era5_scale", "era5_valid",
               "pre_regrid", "post_regrid"]
    hr, lr = dict(zip(hr_keys, hr_outputs)), dict(zip(lr_keys, lr_outputs))

    out = (
        xr.Dataset(
            data_vars={
                "cwb": hr["cwb"], "cwb_center": hr["cwb_center"],
                "cwb_scale": hr["cwb_scale"], "cwb_valid": hr["cwb_valid"],

                "era5": lr["era5"], "era5_center": lr["era5_center"],
                "era5_valid": lr["era5_valid"],
            },
            coords={
                **{k: grid_coords[k] for k in GRID_COORD_KEYS},
                "XTIME": XTIME,
                "time": hr["cwb"].time,
                "cwb_variable": hr["cwb_variable"],
                "era5_scale": ("era5_channel", lr["era5_scale"].data),
            },
        )
        .drop_vars(["south_north", "west_east", "cwb_channel", "era5_channel"])
    )

    if DEBUG:
        dump_regrid_netcdf(tag,
                           hr["pre_regrid"], hr["post_regrid"],
                           lr["pre_regrid"], lr["post_regrid"])

    return out


def write_corrdiff_zarr(ds: xr.Dataset, path: str) -> None:
    """Verify and write a CorrDiff dataset to Zarr format."""
    print(f"\nZARR dataset =>\n{ds}")

    ok, msg = verify_dataset(ds)
    if not ok:
        print(f"\nDataset verification failed => {msg}")
        return

    write_to_zarr(path, ds)


def main():
    """
    Generate and write CorrDiff Zarr datasets for all CORDEX train and test configurations.

    This function iterates over supported experiment domains and:
    - Builds CorrDiff training datasets for all training configurations
      (e.g., pseudo-reality, historical/future emulator)
    - Builds CorrDiff test datasets for all GCM test sets (TG, OOSG),
      for both perfect and imperfect predictor cases
    - Serializes each resulting dataset to Zarr format using standardized
      naming conventions

    The first training configuration is used as the reference source of
    static fields (e.g., orography) for all test datasets within a domain.

    This function serves as the main entry point for batch generation of
    CORDEX-based CorrDiff datasets.
    """
    domains = ["ALPS", "NZ", "SA"]
    train_configs = ["ESD_pseudo_reality", "Emulator_hist_future"]
    gcm_sets = ["TG", "OOSG"]

    for exp_domain in domains:
        # train
        for train_config in train_configs:
            ds = build_out(*generate_cordex_train_outputs(exp_domain, train_config),
                           tag=f"{exp_domain}_{train_config[:3]}")
            write_corrdiff_zarr(ds, f"cordex_train_{exp_domain}_{train_config[:3]}.zarr")

        # test (TG / OOSG) x (perfect / imperfect)
        for test_config, perfect in product(gcm_sets, [False, True]):
            perfect_suffix = "perfect" if perfect else "imperfect"

            ds = build_out(
                *generate_cordex_test_outputs(exp_domain, train_configs[0], test_config, perfect),
                tag=f"{exp_domain}_{test_config}_{perfect_suffix}"
            )
            write_corrdiff_zarr(
                ds, f"cordex_test_{exp_domain}_{test_config}_{perfect_suffix}.zarr"
            )


if __name__ == "__main__":
    main()
