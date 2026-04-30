import matplotlib.pyplot as plt
import pandas as pd
import pymap3d as pm
import contextily as ctx
import geopandas as gpd
import numpy as np
from datetime import datetime

from IMU_reader import read_ground_truth_csv,read_imu_csv

def plot_ground_truth(X_ECEF,Y_ECEF,Z_ECEF):
   # Convert ECEF → lat/lon
    lat, lon, _ = pm.ecef2geodetic(X_ECEF,Y_ECEF,Z_ECEF)

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
    plt.title("Ground Truth trajectory Aerial Map")
    plt.show()

def plot_gyro_accel(df):
    """
    Plot gyroscope and accelerometer data
    Columns:
    'Gyro_X', 'Gyro_Y', 'Gyro_Z',
    'Accel_X', 'Accel_Y', 'Accel_Z'
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Gyroscope ---
    ax[0].plot(df["datetime"], df["Gyro_X"], label="Gyro X")
    ax[0].plot(df["datetime"], df["Gyro_Y"], label="Gyro Y")
    ax[0].plot(df["datetime"], df["Gyro_Z"], label="Gyro Z")
    ax[0].set_title("Gyroscope")
    ax[0].set_ylabel("Angular velocity (deg/s)")
    ax[0].legend()
    ax[0].grid()

    # --- Accelerometer ---
    ax[1].plot(df["datetime"], df["Accel_X"], label="Accel X")
    ax[1].plot(df["datetime"], df["Accel_Y"], label="Accel Y")
    ax[1].plot(df["datetime"], df["Accel_Z"], label="Accel Z")
    ax[1].set_title("Accelerometer")
    ax[1].set_ylabel("Acceleration (m/s²)")
    ax[1].set_xlabel("UTC Time")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()

def return_bias_std(static_data_column):
    """
    return mean + std of static data column
    """
    bias = np.mean(static_data_column)
    std = np.std(static_data_column)
    return bias,std

def plot_imu_histogram(df:pd.DataFrame) -> None:
    """
    Plot histogram of IMU data columns
    """
    imu_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z','Accel_X', 'Accel_Y', 'Accel_Z']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(imu_cols):
        bias,std = return_bias_std(df_imu_stationary[col])
        axes[i].hist(df[col], bins=50, color='blue', alpha=0.7)
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].text(0.05, 0.95, f"Bias: {bias:.4f}\nStd: {std:.4f}", 
                     transform=axes[i].transAxes, horizontalalignment='left', 
                     verticalalignment='top', fontsize=14, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        axes[i].grid()

    plt.tight_layout()
    plt.show()

def plot_single_axis_deadreackoning(df:pd.DataFrame, axis_name:str) -> None:
    """
    Plot dead reckoning trajectory for a single axis
    """
    sample_rate = 100  # Hz
    plot_time = 10  # seconds

    time_axis = df['Time'] - df['Time'][0]
    time_axis = time_axis[:sample_rate * plot_time]  # Limit to first 3 minutes
    dt = time_axis.diff().fillna(0)
    accel = df[axis_name][time_axis.index]  # Limit accel to same length as time_axis
    velocity = (accel * dt).cumsum()
    position = (velocity * dt).cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, accel, label=f"{axis_name} acceleration", color='blue')
    plt.plot(time_axis, velocity, label=f"{axis_name} velocity", color='orange')
    plt.plot(time_axis, position, label=f"{axis_name} position", color='green')
    plt.title(f"Dead Reckoning")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Plot ground truth track
    # file_path = r"data/run2_groundtruth.txt"
    # df_truth = read_ground_truth_csv(file_path,gps_week=2415)
    # X_ECEF,Y_ECEF,Z_ECEF = df_truth["X-ECEF"].values,df_truth["Y-ECEF"].values,df_truth["Z-ECEF"].values
    # plot_ground_truth(X_ECEF,Y_ECEF,Z_ECEF)

    # Plot gyroscope and acc. data
    #imu_file = r"data/run2_imu.txt"
    imu_file = r"data/static_imu.txt"
    df_imu = read_imu_csv(imu_file, gps_week=2415)    
    #print(df_imu)

    static_imu_file = r"data/static_imu.txt"
    df_imu_stationary = read_imu_csv(static_imu_file, gps_week=2415)


    for col in ['Gyro_X', 'Gyro_Y', 'Gyro_Z','Accel_X', 'Accel_Y', 'Accel_Z']:
        #print(df_imu_stationary[col].describe())
        bias,std = return_bias_std(df_imu_stationary[col])
        #print(f"{col} - Bias: {bias:.4f}, Std: {std:.4f}")

    plot_gyro_accel(df_imu)
    plot_imu_histogram(df_imu)
    plot_single_axis_deadreackoning(df_imu, "Accel_X")


    # Plot imu (dead reckoning trajectory)
    imu_trajectory = r"data/static_trajectory.csv"
    seconds_to_plot = 180
    df_imu_traj = pd.read_csv(imu_trajectory,dtype=float)
    df_imu_traj = df_imu_traj[df_imu_traj["time_s"]<seconds_to_plot]
    X_ECEF, Y_ECEF, Z_ECEF = df_imu_traj["ECEF_X_m"].values, df_imu_traj["ECEF_Y_m"].values, df_imu_traj["ECEF_Z_m"]
    #plot_ground_truth(X_ECEF[::10],Y_ECEF[::10],Z_ECEF[::10])

    # Plot SPP
    #spp_file = pd.read_csv(r"data\run2_spp_solution.csv")
    #print(spp_file.columns)
    #plot_ground_truth(spp_file["X"],spp_file["Y"],spp_file["Z"])



    