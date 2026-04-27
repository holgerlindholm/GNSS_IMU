import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt

from IMU_reader import read_imu_csv, read_ground_truth_csv
from coordinate_converter import ned_rotation_matrix, yaw_rotation_matrix, GRAVITY_MAGNITUDE
from Kalman_filter import KF


def make_gravity_ecef(r0_ecef, lat0, lon0, alt0):
    gx, gy, gz = pm.ned2ecef(0.0, 0.0, GRAVITY_MAGNITUDE, lat0, lon0, alt0)
    g_e = np.array([gx, gy, gz]) - r0_ecef

    print("g_e:", g_e)
    print("|g_e|:", np.linalg.norm(g_e))

    return g_e


def main():
    gps_week = 2415
    dt = 0.01
    gnss_update_steps = 100
    start_time = pd.Timestamp("2026-04-23 07:33:00")

    imu_file = "data/run2_imu.txt"
    gt_file = "data/run2_groundtruth.txt"

    imu = read_imu_csv(imu_file, gps_week=gps_week)
    gt = read_ground_truth_csv(gt_file, gps_week=gps_week)

    # Crop BOTH to same start time
    imu = imu[imu["datetime"] > start_time].reset_index(drop=True)
    gt = gt[gt["datetime"] > start_time].reset_index(drop=True)

    print("IMU samples:", len(imu))
    print("GT samples:", len(gt))

    gyro_deg = imu[["Gyro_X", "Gyro_Y", "Gyro_Z"]].to_numpy()
    accel_raw = imu[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy()

    # Fixed pre-calibration
    gyro_bias_deg = np.array([0.0855, 0.0780, 0.2700])
    accel_bias = np.array([-0.3237, 0.0300, 0.0])

    gyro = np.deg2rad(gyro_deg - gyro_bias_deg)
    accel = accel_raw - accel_bias

    # Initial conditions after crop
    r0_ecef = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].iloc[0].to_numpy()
    v0_ecef = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].iloc[0].to_numpy()

    lat0, lon0, alt0 = pm.ecef2geodetic(*r0_ecef)
    g_e = make_gravity_ecef(r0_ecef, lat0, lon0, alt0)

    heading_deg = gt["Heading"].iloc[0] if "Heading" in gt.columns else 113.2533301520
    heading_rad = np.deg2rad(heading_deg)

    R_ned = ned_rotation_matrix(lat0, lon0, alt0, r0_ecef)
    C_neu_to_ned = np.diag([1.0, 1.0, -1.0])
    R_yaw = yaw_rotation_matrix(-heading_rad)
    C_b_e0 = R_ned.T @ R_yaw @ C_neu_to_ned
    #C_b_e0 = R_ned.T @ C_neu_to_ned

    # --- SANITY CHECK ---
    f0_e = C_b_e0 @ accel[0]

    print("\n--- SANITY CHECK ---")
    print("accel[0] (body):", accel[0])
    print("f0_e (ECEF):", f0_e)
    print("g_e:", g_e)
    print("f0_e + g_e:", f0_e + g_e)
    print("|f0_e + g_e|:", np.linalg.norm(f0_e + g_e))

    # Initial covariance
    P = np.zeros((15, 15))
    P[0:3, 0:3] = (5.0**2) * np.eye(3)
    P[3:6, 3:6] = (0.5**2) * np.eye(3)
    P[6:9, 6:9] = (np.deg2rad(1.0) ** 2) * np.eye(3)
    P[9:12, 9:12] = (0.005 ** 2) * np.eye(3)
    P[12:15, 12:15] = (np.deg2rad(0.005) ** 2) * np.eye(3)

    # Process noise
    Q = np.zeros((15, 15))
    Q[3:6, 3:6] = (0.3**2) * dt * np.eye(3)
    Q[6:9, 6:9] = (np.deg2rad(0.3)**2) * dt * np.eye(3)
    Q[9:12, 9:12] = (1e-4**2) * dt * np.eye(3)
    Q[12:15, 12:15] = (1e-5**2) * dt * np.eye(3)

    # Fake GNSS noise, because ground truth is being used as GNSS data
    # i did not compute the spp,DGPS solution here
    R = np.zeros((6, 6))
    R[0:3, 0:3] = (0.75 ** 2) * np.eye(3)  # position noise
    R[3:6, 3:6] = (0.08 ** 2) * np.eye(3)  # velocity noise,

    kf = KF(r0_ecef, v0_ecef, C_b_e0, P, Q, R, g_e)

    gt_pos = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].to_numpy()
    gt_vel = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].to_numpy()

    imu_time = np.arange(len(imu)) * dt

    out_time = []
    out_pos = []
    out_vel = []
    nis_vals = []
    nis_times = []

    for k in range(len(imu)):
        kf.predict(accel[k], gyro[k], dt)

        t = imu_time[k]

        # 1 Hz fake GNSS update
        if k % gnss_update_steps == 0:
            gnss_idx = min(k, len(gt_pos) - 1)

            y, S = kf.update(gt_pos[gnss_idx], gt_vel[gnss_idx])

            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)
            except np.linalg.LinAlgError:
                print("S became singular at k =", k)
                break

            nis_vals.append(nis)
            nis_times.append(t)

        # Divergence guard
        if (
            not np.all(np.isfinite(kf.r_e))
            or not np.all(np.isfinite(kf.v_e))
            or not np.all(np.isfinite(kf.P))
        ):
            print("Filter diverged at k =", k, "time =", t)
            break

        out_time.append(t)
        out_pos.append(kf.r_e.copy())
        out_vel.append(kf.v_e.copy())

    out_pos = np.asarray(out_pos)
    out_vel = np.asarray(out_vel)

    north, east, down = pm.ecef2ned(
        out_pos[:, 0], out_pos[:, 1], out_pos[:, 2],
        lat0, lon0, alt0
    )

    gt_n, gt_e, gt_d = pm.ecef2ned(
        gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
        lat0, lon0, alt0
    )

    results = pd.DataFrame({
        "time": out_time,
        "ECEF_X": out_pos[:, 0],
        "ECEF_Y": out_pos[:, 1],
        "ECEF_Z": out_pos[:, 2],
        "North": north,
        "East": east,
        "Down": down,
        "Vel_X": out_vel[:, 0],
        "Vel_Y": out_vel[:, 1],
        "Vel_Z": out_vel[:, 2],
    })

    results.to_csv("kf_test_results.csv", index=False)

    print("Saved results to kf_test_results.csv")
    print("Number of KF/GNSS updates:", len(nis_vals))
    print("Final KF North/East/Down:", north[-1], east[-1], down[-1])

    # Plot 1: XY trajectory
    plt.figure()
    plt.plot(results["East"], results["North"], label="KF")
    plt.plot(gt_e, gt_n, "--", label="Ground truth as GNSS")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("XY trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: ECEF X and Y
    plt.figure()
    plt.plot(results["time"], results["ECEF_X"] - results["ECEF_X"].iloc[0], label="KF ΔX")
    plt.plot(results["time"], results["ECEF_Y"] - results["ECEF_Y"].iloc[0], label="KF ΔY")
    plt.xlabel("Time [s]")
    plt.ylabel("ECEF change [m]")
    plt.title("KF ECEF ΔX and ΔY")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 3: NIS
    plt.figure()
    plt.plot(nis_times, nis_vals, label="NIS")
    plt.axhline(12.592, linestyle="--", label="95% bound, dof=6")
    plt.axhline(16.812, linestyle="--", label="99% bound, dof=6")
    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("NIS consistency check")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Position RMSE
    N = min(len(north), len(gt_n), len(out_time))

    pos_error = np.vstack((
        north[:N] - gt_n[:N],
        east[:N] - gt_e[:N],
        down[:N] - gt_d[:N]
    )).T

    pos_error_norm = np.sqrt(np.sum(pos_error ** 2, axis=1))

    rmse_cum = np.sqrt(
        np.cumsum(pos_error_norm ** 2) / np.arange(1, N + 1)
    )

    plt.figure()
    plt.plot(out_time[:N], pos_error_norm, label="Position error norm")
    plt.plot(out_time[:N], rmse_cum, label="Cumulative RMSE")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position RMSE")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Final cumulative position RMSE:", rmse_cum[-1])


if __name__ == "__main__":
    main()