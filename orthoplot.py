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
import os


def orthoplot(SPP_sol):
    """
    
    Takes SPP solution and plots over Area of Interest

    Parameters
    ----------
    SPP_sol: Solution of position (X,Y,Z-matrix) as csv-file
        Note name of headers should be the same

    Returns
    -------
    None.

    """
    # Ensure input is a list
    if isinstance(SPP_sol, str):
        SPP_sol = [SPP_sol]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Run for each file
    for i, file in enumerate(SPP_sol):
        df = pd.read_csv(file)

        X = df['X-ECEF']
        Y = df['Y-ECEF']
        Z = df['Z-ECEF']

        # Convert ECEF → lat/lon
        lat, lon, _ = pm.ecef2geodetic(X, Y, Z)

        df['lat'] = lat
        df['lon'] = lon

        # Convert to Web Mercator
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        label = os.path.splitext(os.path.basename(file))[0]

        # Plot with different color per dataset
        gdf.plot(ax=ax, label=label, markersize=5)

    # Add basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax.set_axis_off()
    plt.title("Rover GNSS position")

    # Add legend only if multiple datasets
    if len(SPP_sol) > 1:
        plt.legend(title="Datasets")

    plt.show()

orthoplot(["data/run2_spp_solution.csv", "data/run3_spp_solution.csv", "data/run4_spp_solution.csv"])