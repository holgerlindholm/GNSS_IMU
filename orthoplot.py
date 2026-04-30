# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:00:03 2026

@author: mathi
"""
import matplotlib.pyplot as plt
import pandas as pd
import pymap3d as pm
import contextily as ctx
import geopandas as gpd


def orthoplot(SPP_sol):
    """
    
    Takes SPP solution and plots over Area of Interest

    Parameters
    ----------
    SPP_sol: Solution of position (X,Y,Z-matrix)

    Returns
    -------
    None.

    """
    
    # Convert SPP to dataframe
    df = pd.read_csv(SPP_sol)
    X = df['X']
    Y = df['Y']
    Z = df['Z']
    
    # Convert ECEF → lat/lon
    lat, lon, _ = pm.ecef2geodetic(X,Y,Z)

    # Create DataFrame for plotting
    df['lat'] = lat
    df['lon'] = lon
    # plot_df = pd.DataFrame({"lat": lat, "lon": lon})
    

    # Convert to Web Mercator (required for map tiles)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax)

    # Add aerial/satellite basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik  # aerial imagery
    )

    ax.set_axis_off()
    plt.title("Rover GNSS position")
    plt.show()

orthoplot("data/run4_spp_solution.csv")