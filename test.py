import matplotlib.pyplot as plt
import pandas as pd
import pymap3d as pm
import contextily as ctx
import geopandas as gpd

from IMU_reader import read_ground_truth_csv

if __name__ == "__main__":
    file_path = r"data\run2_groundtruth.txt"
    df = read_ground_truth_csv(file_path)

    # Convert ECEF → lat/lon
    lat, lon, _ = pm.ecef2geodetic(
        df['X-ECEF'].values,
        df['Y-ECEF'].values,
        df['Z-ECEF'].values
    )

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({"lat": lat, "lon": lon})

    # Convert to Web Mercator (required for map tiles)
 
    gdf = gpd.GeoDataFrame(
        plot_df,
        geometry=gpd.points_from_xy(plot_df.lon, plot_df.lat),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax, linewidth=2)

    # Add aerial/satellite basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik  # aerial imagery
    )

    ax.set_axis_off()
    plt.title("Trajectory on Aerial Map")
    plt.show()