import numpy as np
import pandas as pd
import pymap3d as pm
from datetime import datetime

# --- Configuration & Data Loading ---
# (Assuming your IMU_reader functions are available)
from IMU_reader import read_imu_csv, read_ground_truth_csv

imu_file = r"data\run2_imu.txt"
ground_truth_file = r"data\run2_groundtruth.txt"

df_imu = read_imu_csv(imu_file, gps_week=2415)
df_imu = df_imu[df_imu["datetime"] > datetime(2026, 4, 23, 7, 33)]
df_truth = read_ground_truth_csv(ground_truth_file, gps_week=2415)

N = len(df_imu)
t = df_imu["Time"].to_numpy()
dt = np.diff(t, prepend=t[0])

# --- 1. Bias Correction (Using your provided static mean values) ---
# Gyro: deg/s -> rad/s
gyro_bias = np.deg2rad(np.array([0.0855, 0.0780, 0.2700]))
# Accel: m/s² (Note: Z-bias 9.8237 includes gravity)
accel_bias = np.array([-0.1679, -0.0173, 9.8237])

gyro = np.deg2rad(df_imu[["Gyro_X", "Gyro_Y", "Gyro_Z"]].to_numpy()) - gyro_bias
accel = df_imu[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy() - accel_bias

# --- 2. Initial State ---
r0_ecef = df_truth[["X-ECEF", "Y-ECEF", "Z-ECEF"]].iloc[0].to_numpy()
lat0, lon0, h0 = pm.ecef2geodetic(*r0_ecef)

# Initial Orientation (Euler angles in radians)
roll = np.deg2rad(0.9267)
pitch = np.deg2rad(-0.1970)
yaw = np.deg2rad(113.2533)

# Storage
vel_ned = np.zeros((N, 3))
pos_ned = np.zeros((N, 3))
g_ned = np.array([0, 0, 0]) # Gravity is already removed via your Z-bias calibration

def get_rotation_matrix(phi, theta, psi):
    """Body to NED rotation matrix (ZYX convention)"""
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

# --- 3. Integration Loop ---
for k in range(1, N):
    # Update Orientation (Rates to Angles)
    # Simple integration; for high precision, use Quaternions
    roll  += gyro[k, 0] * dt[k]
    pitch += gyro[k, 1] * dt[k]
    yaw   += gyro[k, 2] * dt[k]

    # Create Rotation Matrix (Body -> NED)
    R_nb = get_rotation_matrix(roll, pitch, yaw)

    # Transform Body Accel to Navigation Frame (NED)
    # We use the bias-corrected accel (where gravity was already subtracted in static test)
    a_nav = R_nb @ accel[k]

    # Integrate Velocity
    vel_ned[k] = vel_ned[k-1] + a_nav * dt[k]

    # Integrate Position
    pos_ned[k] = pos_ned[k-1] + vel_ned[k] * dt[k]

# --- 4. Coordinate Conversion (NED to ECEF) ---
positions = np.zeros((N, 3))
for k in range(N):
    positions[k] = pm.ned2ecef(pos_ned[k, 0], pos_ned[k, 1], pos_ned[k, 2], lat0, lon0, h0)

# --- 5. Output ---
results = pd.DataFrame({
    "time_s": t - t[0],
    "North_m": pos_ned[:, 0],
    "East_m": pos_ned[:, 1],
    "Down_m": pos_ned[:, 2],
    "ECEF_X_m": positions[:, 0],
    "ECEF_Y_m": positions[:, 1],
    "ECEF_Z_m": positions[:, 2]
})

print(f"Final 2D Displacement: {np.hypot(pos_ned[-1, 0], pos_ned[-1, 1]):.2f} meters")