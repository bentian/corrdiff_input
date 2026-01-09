"""
CorrDiff CORDEX Zarr dataset generator.

Builds consolidated CorrDiff-ready datasets from CORDEX HR/LR sources (train + test),
optionally dumps intermediate regrid NetCDFs for debugging, verifies schema, and
writes the final result to Zarr.
"""
import numpy as np
import xarray as xr
from itertools import product

from corrdiff_datagen import verify_dataset, write_to_zarr, dump_regrid_netcdf
from data_builder import \
    GRID_COORD_KEYS, generate_cordex_train_outputs, generate_cordex_test_outputs

DEBUG = False
XTIME = np.datetime64("2026-01-09 17:00:00", "ns")  # placeholder timestamp


def build_out(hr_outputs, lr_outputs, grid_coords, tag: str) -> xr.Dataset:
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
    for exp_domain in ["ALPS", "NZ"]:
        # train
        for train_config in ["ESD_pseudo_reality", "Emulator_hist_future"]:
            ds = build_out(*generate_cordex_train_outputs(exp_domain, train_config),
                           tag=f"{exp_domain}_{train_config[:3]}")
            write_corrdiff_zarr(ds, f"cordex_train_{exp_domain}_{train_config[:3]}.zarr")

            # test (TG / OOSG) x (perfect / imperfect)
            # for test_config, perfect in product(["TG", "OOSG"], [False, True]):
            #     perfect_suffix = "perfect" if perfect else "imperfect"

            #     ds = build_out(
            #         *generate_cordex_test_outputs(exp_domain, train_config, test_config, perfect),
            #         tag=f"{exp_domain}_{test_config}_{perfect_suffix}"
            #     )
            #     write_corrdiff_zarr(
            #         ds, f"cordex_test_{exp_domain}_{test_config}_{perfect_suffix}.zarr"
            #     )


if __name__ == "__main__":
    main()
