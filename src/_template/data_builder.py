from typing import Tuple, Optional
import xarray as xr

# TODO: Modify data_loader/__init__.py to expose these methods
from data_loader import get_hr_dataset, get_lr_dataset, get_hr_channels, get_lr_channels
from tensor_fields import get_cwb_fields, get_era5_fields


def get_ref_grid(
    scenario: Optional[str] = None,
) -> Tuple[xr.Dataset, xr.Dataset, Optional[xr.Dataset]]:
    """
    SAMPLE TEMPLATE: build the reference grid used by CorrDiff.

    This mirrors the two existing patterns in data_builder.py:

    1) CWA-style (ERA5 + TReAD):
       - Grid comes from a fixed WRF domain file
       - Terrain (TER/slope/aspect) is required and returned

    2) SSP-style (TaiESM):
       - Grid comes from a scenario-specific WRF domain
       - Terrain is NOT required downstream → return None

    You should adapt:
      - grid file path
      - coordinate names
      - whether terrain layers are needed
    """

    # ------------------------------------------------------------
    # 1) Load grid definition
    # ------------------------------------------------------------
    grid = xr.open_dataset("path/to/your_ref_grid.nc")

    # ------------------------------------------------------------
    # 2) Normalize grid coordinates
    # ------------------------------------------------------------
    # CorrDiff expects a consistent spatial layout.
    # Normalize naming here so HR/LR loaders don’t need to care.
    #
    # Typical WRF grid:
    #   dims: south_north, west_east
    #   coords: XLAT, XLONG
    #
    if "XLAT" in grid:
        grid = grid.rename({"XLAT": "lat", "XLONG": "lon"})


    # ------------------------------------------------------------
    # 3) Grid coordinates (returned separately)
    # ------------------------------------------------------------
    # data_builder.py typically returns grid_coords for
    # writing metadata / NetCDF headers later.
    grid_coords = xr.Dataset(
        {
            "lat": grid["lat"],
            "lon": grid["lon"],
        }
    )

    # ------------------------------------------------------------
    # 4) Optional terrain layers
    # ------------------------------------------------------------
    # Needed for ERA5-style LR conditioning.
    # SSP-style workflows usually return None here.
    #
    terrain = None
    if scenario is None:
        # ---- CWA / ERA5-style terrain ----
        terrain = xr.open_dataset("path/to/hr_terrain_layers.nc")
        # Expected variables (example):
        #   - TER
        #   - slope
        #   - aspect

        # Safety: ensure spatial dims match grid
        terrain = terrain.transpose(*grid.dims)

    # ------------------------------------------------------------
    # 5) Return
    # ------------------------------------------------------------
    return grid, grid_coords, terrain


def generate_outputs(
    start_date: str,
    end_date: str,
    *,
    scenario: Optional[str] = None,
) -> Tuple[tuple, tuple, xr.Dataset]:
    """
    TEMPLATE: generate CorrDiff-ready outputs for a new workflow.

    Pattern is identical to:
      - generate_cwa_outputs(): (HR=TReAD, LR=ERA5, terrain needed)
      - generate_ssp_outputs(): (HR=TaiESM3p5, LR=TaiESM100, scenario needed)

    You should replace:
      - get_ref_grid(...) arguments (do you need scenario? terrain? none?)
      - get_hr_dataset / get_lr_dataset
      - get_hr_channels / get_lr_channels
      - optional terrain injection behavior (if LR needs terrain like ERA5)

    Returns
    -------
    (hr_outputs, lr_outputs, grid_coords)

    Where:
      hr_outputs = (
          *get_cwb_fields(hr_out, hr_channels),
          hr_pre_regrid,
          hr_out,
      )

      lr_outputs = (
          *get_era5_fields(lr_out, lr_channels),
          lr_pre_regrid,
          lr_out,
      )
    """

    # ------------------------------------------------------------
    # 1) Reference grid
    # ------------------------------------------------------------
    # If your workflow needs a scenario-specific grid (SSP-style), pass it here.
    # If your workflow needs terrain layers (CWA/ERA5-style), keep `terrain`.
    #
    # Examples:
    #   grid, grid_coords, terrain = get_ref_grid()                 # CWA
    #   grid, grid_coords, _       = get_ref_grid(ssp_level)        # SSP
    #
    # TODO: edit to your needs:
    grid, grid_coords, terrain = get_ref_grid(scenario)

    # ------------------------------------------------------------
    # 2) HR dataset (the “target” side)
    # ------------------------------------------------------------
    # TODO: implement these functions in your HR loader module:
    #   - get_hr_dataset(grid, start_date, end_date, scenario?)
    #   - get_hr_channels()
    #
    # Should return: (pre_regrid_ds, regridded_ds)
    hr_pre_regrid, hr_out = get_hr_dataset(grid, start_date, end_date, scenario)
    print(f"\nHR dataset [{scenario}] =>\n {hr_out}")

    hr_outputs = (
        *get_cwb_fields(hr_out, get_hr_channels()),
        hr_pre_regrid,
        hr_out,
    )

    # ------------------------------------------------------------
    # 3) LR dataset (the “conditioning” side)
    # ------------------------------------------------------------
    # TODO: implement these functions in your LR loader module:
    #   - get_lr_dataset(grid, terrain?, start_date, end_date, scenario?)
    #   - get_lr_channels()
    #
    # Notes:
    # - If LR needs terrain injection like ERA5, pass `terrain`.
    # - If LR is scenario-based like TaiESM100, pass `scenario`.
    #
    # Example signatures:
    #   lr_pre_regrid, lr_out = get_lr_dataset(grid, terrain, start_date, end_date)
    #   lr_pre_regrid, lr_out = get_lr_dataset(grid, start_date, end_date, scenario)
    #
    lr_pre_regrid, lr_out = get_lr_dataset(grid, terrain, start_date, end_date, scenario)
    print(f"\nLR dataset [{scenario}] =>\n {lr_out}")

    lr_outputs = (
        *get_era5_fields(lr_out, get_lr_channels()),
        lr_pre_regrid,
        lr_out,
    )

    # ------------------------------------------------------------
    # 4) Return
    # ------------------------------------------------------------
    return hr_outputs, lr_outputs, grid_coords
