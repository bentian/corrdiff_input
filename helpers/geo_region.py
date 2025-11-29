"""
Utility module for visualizing a geographic bounding box using Cartopy.

This script plots a rectangular region defined by fixed latitude and longitude
bounds and centers the map view on that region. The bounding box is highlighted
in red, and standard map features (land, ocean, borders, coastlines) are added
for context.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Region bounds
lat_min, lat_max = -9.895, 69.27
lon_min, lon_max = 40, 190

def plot_geo_region() -> None:
    """Plot the geographic region defined by global lat/lon bounds."""
    # Center longitude of the box
    center_lon = (lon_min + lon_max) / 2.0

    # Create map
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center_lon))
    ax.set_extent(
        [lon_min - 10, lon_max + 10, lat_min - 10, lat_max + 10],
        crs=ccrs.PlateCarree()
    )

    # Add map layers
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Draw bounding box
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color='red', linewidth=2, transform=ccrs.PlateCarree())

    ax.set_title(f"Region: lat [{lat_min}, {lat_max}] lon [{lon_min}, {lon_max}]")
    plt.show()

def main() -> None:
    """Run the geographic region plot."""
    plot_geo_region()

if __name__ == "__main__":
    main()
