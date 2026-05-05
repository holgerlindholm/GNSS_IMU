import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
from pathlib import Path

from IMU_reader import read_imu_csv, read_ground_truth_csv
from coordinate_converter import ned_rotation_matrix, yaw_rotation_matrix, GRAVITY_MAGNITUDE
from Kalman_filter import KF

#settings for outage (dropout)
#20 second dropout
enable_outage = True
outage_start = 150.0
outage_end = 155.0

# def make_gravity_ecef(r0_ecef, lat0, lon0, alt0):
#     gx, gy, gz = pm.ned2ecef(0.0, 0.0, GRAVITY_MAGNITUDE, lat0, lon0, alt0)
#     g_e = np.array([gx, gy, gz]) - r0_ecef

#     print("g_e:", g_e)
#     print("|g_e|:", np.linalg.norm(g_e))

#     return g_e

def make_gravity_ecef(lat0, lon0,alt0,r0_ecef):
    # gravity in NED frame (Down is positive)
    g_ned = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])

    R_ned = ned_rotation_matrix(lat0, lon0, alt0, r0_ecef)

    g_ned = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    g_e = R_ned.T @ g_ned

    return g_e

#function to handle predict update for ekf (with outages)
#this way easier to simualte with and without outages)
def run_kf_case(case_name, enable_outage, dt, accel, gyro, imu_time,
                spp_time, spp_pos, spp_vel,
                r0_ecef, v0_ecef, C_b_e0, P, Q, R, g_e):

    #generate KF object:
    kf = KF(r0_ecef, v0_ecef, C_b_e0.copy(), P.copy(), Q.copy(), R.copy(), g_e)

    out_time = []
    out_pos = []
    out_vel = []
    nis_vals = []
    nis_times = []

    next_gnss_idx = 0

    for k in range(len(imu_time)):
        kf.predict(accel[k], gyro[k], dt)

        t = imu_time[k]

        in_outage = enable_outage and (outage_start <= t <= outage_end) #setup dropout and duration of dropout

        if next_gnss_idx < len(spp_time) and t >= spp_time[next_gnss_idx]:
            #update like normal if not in outage
            if not in_outage:
                #update with spp solutions
                y, S = kf.update(spp_pos[next_gnss_idx], spp_vel[next_gnss_idx])

                nis = float(y.T @ np.linalg.inv(S) @ y)
                nis_vals.append(nis)
                nis_times.append(t)

            next_gnss_idx += 1

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

    return {
        "case_name": case_name,
        "enable_outage": enable_outage,
        "out_time": np.asarray(out_time),
        "out_pos": np.asarray(out_pos),
        "out_vel": np.asarray(out_vel),
        "nis_vals": np.asarray(nis_vals),
        "nis_times": np.asarray(nis_times),
    }

#function to plot the results (to make it easier to plot both versions)
def plot_kf_case(run, gt_pos, lat0, lon0, alt0, plot_dir):
    case_name = run["case_name"]
    out_time = run["out_time"]
    out_pos = run["out_pos"]
    out_vel = run["out_vel"]
    nis_vals = run["nis_vals"]
    nis_times = run["nis_times"]

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

    results.to_csv(plot_dir / f"kf_test_results_{case_name}.csv", index=False)

    print(f"\nSaved results to {plot_dir / f'kf_test_results_{case_name}.csv'}")
    print("Number of KF/GNSS updates:", len(nis_vals))
    print("Final KF North/East/Down:", north[-1], east[-1], down[-1])

    # Plot 1: XY trajectory
    plt.figure()
    plt.plot(results["East"], results["North"], label="KF")
    plt.plot(gt_e, gt_n, "--", label="Ground truth")
    #start end markers
    plt.scatter(results["East"].iloc[0], results["North"].iloc[0],
                marker="o", s=80, label="Start")
    plt.scatter(results["East"].iloc[-1], results["North"].iloc[-1],
                marker="x", s=100, label="End")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title(f"XY trajectory - {case_name}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / f"XY_Trajectory_{case_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: ECEF X and Y
    plt.figure()
    plt.plot(results["time"], results["ECEF_X"] - results["ECEF_X"].iloc[0], label="KF delta_X")
    plt.plot(results["time"], results["ECEF_Y"] - results["ECEF_Y"].iloc[0], label="KF delta_Y")
    if run["enable_outage"]:
        plt.axvspan(outage_start, outage_end, alpha=0.2, label="GNSS outage")  # adds shading for when dropout starts
    plt.xlabel("Time [s]")
    plt.ylabel("ECEF change [m]")
    plt.title(f"KF ECEF delta_X and delta_Y - {case_name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / f"ECEF_deltaXY_{case_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 3: NIS
    plt.figure()
    plt.plot(nis_times, nis_vals, label="NIS")
    plt.axhline(12.592, linestyle="--", label="95% bound, dof=6")
    plt.axhline(16.812, linestyle="--", label="99% bound, dof=6")
    if run["enable_outage"]:
        plt.axvspan(outage_start, outage_end, alpha=0.2, label="GNSS outage") #adds shading for when dropout starts
    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title(f"NIS consistency check - {case_name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / f"NIS_{case_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    #printed result for NIS
    nis_95 = 12.592
    nis_99 = 16.812

    if len(nis_vals) > 0:
        pct_below_95 = 100.0 * np.mean(nis_vals <= nis_95)
        pct_below_99 = 100.0 * np.mean(nis_vals <= nis_99)
        nis_mean = np.mean(nis_vals)

        print(f"\n--- NIS consistency ({case_name}) ---")
        print(f"Mean NIS: {nis_mean:.3f}")
        print(f"Expected mean NIS, dof=6: 6.000")
        print(f"Percent below 95% bound ({nis_95}): {pct_below_95:.2f}%")
        print(f"Percent below 99% bound ({nis_99}): {pct_below_99:.2f}%")
    else:
        print(f"\n--- NIS consistency ({case_name}) ---")
        print("No NIS values were saved")

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
    if run["enable_outage"]:
        plt.axvspan(outage_start, outage_end, alpha=0.2, label="GNSS outage")  # adds shading for when dropout starts
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title(f"Position RMSE - {case_name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / f"Position_RMSE_{case_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Final cumulative position RMSE:", rmse_cum[-1])


def main():
    gps_week = 2415
    dt = 0.01
    gnss_update_steps = 10 #(100/10 = 10 (gnns(10Hz) updates on every 10 time relative to IMU (100hz))


    #change run_id based on runs
    run_id = 2
    imu_file = f"data/run{run_id}_imu.txt"
    gt_file = f"data/run{run_id}_groundtruth.txt"
    spp_file = f"data/run{run_id}_spp_solution.csv"

    #added a suubfodler for spp and rtk ekf plots to declutter folder
    plot_dir = Path("Plots") / f"Run_{run_id}" / "SPP_KF"
    plot_dir.mkdir(parents=True, exist_ok=True)

    imu = read_imu_csv(imu_file, gps_week=gps_week)
    gt = read_ground_truth_csv(gt_file, gps_week=gps_week)
    spp = pd.read_csv(spp_file)

    spp["datetime"] = pd.to_datetime(spp["datetime"]) #spp datetime moved up for start_time comp

    #set start_time based on max val from files
    start_time = max(
        imu["datetime"].iloc[0],
        gt["datetime"].iloc[0],
        spp["datetime"].iloc[0]
    )

    # Crop BOTH to same start time
    imu = imu[imu["datetime"] > start_time].reset_index(drop=True)
    gt = gt[gt["datetime"] > start_time].reset_index(drop=True)

    print("IMU samples:", len(imu))
    print("GT samples:", len(gt))

    gyro_deg = imu[["Gyro_X", "Gyro_Y", "Gyro_Z"]].to_numpy()
    accel_raw = imu[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy()

    # Fixed pre-calibration
    # 0.085950     0.078210     0.274324    -0.167841    -0.016930     9.823505
    gyro_bias_deg = np.array([0.085950, 0.078210, 0.274324])
    accel_bias = np.array([-0.167841, -0.016930, 0.0])

    gyro = np.deg2rad(gyro_deg - gyro_bias_deg)
    accel = accel_raw - accel_bias

    # Initial conditions after crop
    r0_ecef = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].iloc[0].to_numpy()
    v0_ecef = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].iloc[0].to_numpy()

    lat0, lon0, alt0 = pm.ecef2geodetic(*r0_ecef)
    g_e = make_gravity_ecef(lat0, lon0,alt0,r0_ecef)

    heading_deg = gt["Heading"].iloc[0] if "Heading" in gt.columns else 113.2533301520
    heading_rad = np.deg2rad(heading_deg)

    R_ned = ned_rotation_matrix(lat0, lon0, alt0, r0_ecef)
    C_neu_to_ned = np.diag([1.0, 1.0, -1.0])
    R_yaw = yaw_rotation_matrix(-heading_rad)
    C_b_e0 = R_ned.T @ R_yaw @ C_neu_to_ned
    #C_b_e0 = R_ned.T @ C_neu_to_ned

    #SANITY CHECK
    f0_e = C_b_e0 @ accel[0]

    print("\n--- SANITY CHECK ---")
    print("accel[0] (body):", accel[0])
    print("f0_e (ECEF):", f0_e)
    print("g_e:", g_e)
    print("f0_e + g_e:", f0_e + g_e)
    print("|f0_e + g_e|:", np.linalg.norm(f0_e + g_e))

    # Initial covariance
    P = np.zeros((15, 15))
    P += 1e-6 * np.eye(15)
    P[0:3, 0:3] = (5.0**2) * np.eye(3)
    P[3:6, 3:6] = (0.5**2) * np.eye(3)
    P[6:9, 6:9] = (np.deg2rad(1.0) ** 2) * np.eye(3)
    P[9:12, 9:12] = (0.005 ** 2) * np.eye(3)
    P[12:15, 12:15] = (np.deg2rad(0.005) ** 2) * np.eye(3)

    # Process noise
    Q = np.zeros((15, 15))
    Q[3:6, 3:6] = (0.5**2) * dt * np.eye(3)
    Q[6:9, 6:9] = (np.deg2rad(0.5)**2) * dt * np.eye(3)
    Q[9:12, 9:12] = (1e-4**2) * dt * np.eye(3)
    Q[12:15, 12:15] = (1e-5**2) * dt * np.eye(3)


    gt_pos = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].to_numpy()
    gt_vel = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].to_numpy()


    # Crop SPP to same start time as IMU/GT
    spp = spp[spp["datetime"] > start_time].reset_index(drop=True)

    # Drop first row if velocity interpolation produced NaN
    spp = spp.dropna(subset=[
        "X-ECEF", "Y-ECEF", "Z-ECEF",
        "VX-ECEF", "VY-ECEF", "VZ-ECEF"
    ]).reset_index(drop=True)

    spp_pos = spp[["X-ECEF", "Y-ECEF", "Z-ECEF"]].to_numpy()
    spp_vel = spp[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].to_numpy()

    spp_time = (spp["datetime"] - start_time).dt.total_seconds().to_numpy()

    #getting R (variance for spp) for std in spp_solution csv
    #using the avaerage std
    pos_var = spp[["std_X", "std_Y", "std_Z"]].pow(2).mean().to_numpy()
    vel_var = spp[["std_VX", "std_VY", "std_VZ"]].pow(2).mean().to_numpy()
    pos_var = np.maximum(pos_var, 1.0**2)   # ≥ 1 m²
    vel_var = np.maximum(vel_var, 0.1**2)   # ≥ 0.01 (m/s)²

    R = np.zeros((6, 6))
    R[0:3, 0:3] = np.diag(pos_var) # position noise (variance)
    R[3:6, 3:6] = np.diag(vel_var) # velocity noise (variance)

    #allign imu time with same ref as SPP
    imu_time = (imu["datetime"] - start_time).dt.total_seconds().to_numpy()

    #run kf without outage
    case_name_no_outage = f"Run_{run_id}_SPP_NoOutage"
    run_no_outage = run_kf_case(
        case_name_no_outage, False, dt, accel, gyro, imu_time,
        spp_time, spp_pos, spp_vel,
        r0_ecef, v0_ecef, C_b_e0, P, Q, R, g_e
    )

    # run kf with outage
    case_name_outage = f"Run_{run_id}_SPP_Outage"
    run_with_outage = run_kf_case(
        case_name_outage, enable_outage, dt, accel, gyro, imu_time,
        spp_time, spp_pos, spp_vel,
        r0_ecef, v0_ecef, C_b_e0, P, Q, R, g_e
    )

    #plot without outage
    plot_kf_case(run_no_outage, gt_pos, lat0, lon0, alt0, plot_dir)
    #plot with outage
    plot_kf_case(run_with_outage, gt_pos, lat0, lon0, alt0, plot_dir)


if __name__ == "__main__":
    main()
