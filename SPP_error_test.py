import matplotlib.pyplot as plt
import pandas as pd
import pymap3d as pm
import contextily as ctx
import geopandas as gpd
import numpy as np

from IMU_reader import read_ground_truth_csv, read_imu_csv,read_spp_csv,read_pos_file

def plot_ground_truth(X_ECEF, Y_ECEF, Z_ECEF, ax=None, label=None, color=None):
    # Drop NaN before converting
    mask = ~(pd.isna(X_ECEF) | pd.isna(Y_ECEF) | pd.isna(Z_ECEF))
    X_ECEF, Y_ECEF, Z_ECEF = X_ECEF[mask], Y_ECEF[mask], Z_ECEF[mask]

    # Convert ECEF → lat/lon
    lat, lon, _ = pm.ecef2geodetic(X_ECEF.values, Y_ECEF.values, Z_ECEF.values)

    # Create GeoDataFrame in Web Mercator
    gdf = gpd.GeoDataFrame(
        {"lat": lat, "lon": lon},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot as a connected LINE, not scatter points
    ax.plot(
        gdf.geometry.x,
        gdf.geometry.y,
        linewidth=2,
        label=label,
        color=color
    )

    return ax

def plot_multiple_tracks(list_of_xyz, names_of_tracks):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, ((X, Y, Z), name) in enumerate(zip(list_of_xyz, names_of_tracks)):
        plot_ground_truth(X, Y, Z, ax=ax, label=name, color=colors[i % len(colors)])

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    ax.legend()
    plt.title("Multiple Tracks Comparison")
    plt.tight_layout()
    plt.show()


# Load data
spp_file        = read_spp_csv(r"data/run2_spp_solution.csv")
spp_file_tropo  = read_spp_csv(r"data/run2_spp_solution_tropo_filter.csv")
ground_truth_file = read_ground_truth_csv(r"data/run2_groundtruth.txt",gps_week=2415)
rtk_file = read_pos_file(r"data\run2_BUDD.pos")

# Sort (required)
spp_file = spp_file.sort_values("GPSTime")
spp_file_tropo = spp_file_tropo.sort_values("GPSTime")
ground_truth_file = ground_truth_file.sort_values("GPSTime")
rtk_file = rtk_file.sort_values("GPSTime")

merged = pd.merge_asof(
    spp_file,
    spp_file_tropo,
    on="GPSTime",
    direction="nearest",
    tolerance=0.1,
    suffixes=("_spp", "_tropo")
)

merged = pd.merge_asof(
    merged,
    ground_truth_file,
    on="GPSTime",
    direction="nearest",
    tolerance=0.1
)

merged = pd.merge_asof(
    merged,
    rtk_file,
    on="GPSTime",
    direction="nearest",
    tolerance=0.1,
    suffixes=("", "_rtk")
)

ecef_cols = [
    "X-ECEF_spp", "Y-ECEF_spp", "Z-ECEF_spp",
    "X-ECEF_tropo", "Y-ECEF_tropo", "Z-ECEF_tropo",
    "X-ECEF", "Y-ECEF", "Z-ECEF",               # ground truth
    "X-ECEF_rtk", "Y-ECEF_rtk", "Z-ECEF_rtk"
]

merged.dropna(subset=ecef_cols, inplace=True)

print(f"Rows after merge: {len(merged)}")   # sanity check — should be > 0

list_of_xyz = [
    (merged["X-ECEF_spp"],   merged["Y-ECEF_spp"],   merged["Z-ECEF_spp"]),
    (merged["X-ECEF_tropo"], merged["Y-ECEF_tropo"], merged["Z-ECEF_tropo"]),
    (merged["X-ECEF_rtk"],   merged["Y-ECEF_rtk"],   merged["Z-ECEF_rtk"]),
    (merged["X-ECEF"],       merged["Y-ECEF"],       merged["Z-ECEF"]),
]

names = ["SPP", "SPP + Tropo + Filter", "RTK", "Ground Truth"]

plot_multiple_tracks(list_of_xyz, names)


# # -----------------------
# # ECEF residuals
# # -----------------------
# dx = spp_file["X-ECEF"] - ground_truth_file["X-ECEF"]
# dy = spp_file["Y-ECEF"] - ground_truth_file["Y-ECEF"]
# dz = spp_file["Z-ECEF"] - ground_truth_file["Z-ECEF"]

# # -----------------------
# # Reference point (origin for ENU)
# # -----------------------
# # Use mean ground truth position
# x_ref = ground_truth_file["X-ECEF"].mean()
# y_ref = ground_truth_file["Y-ECEF"].mean()
# z_ref = ground_truth_file["Z-ECEF"].mean()

# # -----------------------
# # Plot everything
# # -----------------------
# plt.figure(figsize=(12,8))

# # ECEF subplot
# plt.subplot(2,1,1)
# plt.plot(dx, label="dX")
# plt.plot(dy, label="dY")
# plt.plot(dz, label="dZ")
# plt.title("ECEF Residuals")
# plt.ylabel("Error [m]")
# plt.legend()
# plt.grid(True)


# plt.tight_layout()
# plt.show()