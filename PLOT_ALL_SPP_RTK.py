import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
from pathlib import Path

from IMU_reader import read_imu_csv, read_ground_truth_csv
from coordinate_converter import ned_rotation_matrix, yaw_rotation_matrix, GRAVITY_MAGNITUDE, ENU_to_NED
from Kalman_filter import KF

enable_outage = True
outage_start = 150.0
outage_end = 180.0


def make_gravity_ecef(lat0, lon0, alt0, r0_ecef):
    R_ned = ned_rotation_matrix(lat0, lon0, alt0, r0_ecef)
    g_ned = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    return R_ned.T @ g_ned


def run_kf_case(case_name, enable_outage, dt, accel, gyro, imu_time,
                gnss_time, gnss_pos, gnss_vel,
                r0_ecef, v0_ecef, C_b_e0, P, Q, R, g_e):

    kf = KF(r0_ecef, v0_ecef, C_b_e0.copy(), P.copy(), Q.copy(), R.copy(), g_e)

    out_time, out_pos, out_vel = [], [], []
    out_b_a, out_b_g = [], []
    nis_vals, nis_times = [], []
    next_gnss_idx = 0

    for k in range(len(imu_time)):
        kf.predict(accel[k], gyro[k], dt)
        t = imu_time[k]
        in_outage = enable_outage and (outage_start <= t <= outage_end)

        if next_gnss_idx < len(gnss_time) and t >= gnss_time[next_gnss_idx]:
            if not in_outage:
                y, S = kf.update(gnss_pos[next_gnss_idx], gnss_vel[next_gnss_idx])
                nis = float(y.T @ np.linalg.inv(S) @ y)
                nis_vals.append(nis)
                nis_times.append(t)
            next_gnss_idx += 1

        if (not np.all(np.isfinite(kf.r_e))
                or not np.all(np.isfinite(kf.v_e))
                or not np.all(np.isfinite(kf.P))):
            print("Filter diverged at k =", k, "time =", t)
            break

        out_time.append(t)
        out_pos.append(kf.r_e.copy())
        out_vel.append(kf.v_e.copy())
        out_b_a.append(kf.b_a.copy())
        out_b_g.append(kf.b_g.copy())

    return {
        "case_name":    case_name,
        "enable_outage": enable_outage,
        "out_time":     np.asarray(out_time),
        "out_pos":      np.asarray(out_pos),
        "out_vel":      np.asarray(out_vel),
        "out_b_a":      np.asarray(out_b_a),
        "out_b_g":      np.asarray(out_b_g),
        "nis_vals":     np.asarray(nis_vals),
        "nis_times":    np.asarray(nis_times),
    }


def plot_kf_cases(runs: list, gt_pos, lat0, lon0, alt0, plot_dir, save_plot=False):
    """
    Overlay an arbitrary number of KF runs on the same figures.
    Each entry in `runs` is a dict returned by run_kf_case().
    """
    # Pre-compute NED for ground truth and every run
    gt_n, gt_e, gt_d = pm.ecef2ned(
        gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], lat0, lon0, alt0
    )

    ned_runs = []
    for run in runs:
        n, e, d = pm.ecef2ned(
            run["out_pos"][:, 0], run["out_pos"][:, 1], run["out_pos"][:, 2],
            lat0, lon0, alt0
        )
        ned_runs.append((n, e, d))

    # Build a combined case label for file names
    combined_label = "_vs_".join(r["case_name"].replace(" ", "_") for r in runs)

    # ------------------------------------------------------------------
    # Figure 1: Horizontal trajectory + vertical position
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10),
                             gridspec_kw={"height_ratios": [2, 1]})

    # ── Top: East-North trajectory ─────────────────────────────────────
    ax = axes[0]
    ax.plot(gt_e, gt_n, "k--", lw=1.2, label="Ground truth")

    for run, (n, e, d) in zip(runs, ned_runs):
        ax.plot(e, n, lw=1.2, label=run["case_name"])

        if run["enable_outage"]:
            t = run["out_time"]
            idx_s = np.argmin(np.abs(t - outage_start))
            idx_e = np.argmin(np.abs(t - outage_end))
            ax.scatter(e[idx_s], n[idx_s], marker="s", s=100, zorder=5)
            ax.scatter(e[idx_e], n[idx_e], marker="D", s=100, zorder=5)
            ax.text(e[idx_s], n[idx_s], " outage start", fontsize=8)
            ax.text(e[idx_e], n[idx_e], " outage end",   fontsize=8)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Horizontal trajectory")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # ── Bottom: Vertical position (Up) ────────────────────────────────
    ax2 = axes[1]
    ax2.plot(runs[0]["out_time"][:len(gt_d)], -gt_d, "k--", lw=1.2,
             label="Ground truth")

    for run, (n, e, d) in zip(runs, ned_runs):
        up = -d
        ax2.plot(run["out_time"][:len(up)], up, lw=1.2, label=run["case_name"])

        if run["enable_outage"]:
            t = run["out_time"]
            idx_s = np.argmin(np.abs(t - outage_start))
            idx_e = np.argmin(np.abs(t - outage_end))
            ax2.scatter(t[idx_s], up[idx_s], marker="s", s=100, zorder=5)
            ax2.scatter(t[idx_e], up[idx_e], marker="D", s=100, zorder=5)
            ax2.axvline(outage_start, ls="--", alpha=0.4)
            ax2.axvline(outage_end,   ls="--", alpha=0.4)

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Up [m]")
    ax2.set_title("Vertical position")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    if save_plot:
        fig.savefig(plot_dir / f"Trajectory_{combined_label}.png",
                    dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # Figure 2: NIS
    # ------------------------------------------------------------------
    fig_nis, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig_nis.suptitle("KF Performance – NIS", fontsize=13, fontweight="bold")

    for run in runs:
        ax.plot(run["nis_times"], run["nis_vals"], lw=0.8,
                alpha=0.8, label=run["case_name"])
    ax.axhline(12.592, ls="--", color="C2", lw=1, label="95 % bound (dof=6)")
    ax.axhline(16.812, ls="--", color="C3", lw=1, label="99 % bound (dof=6)")
    ax.axhline(6.0,    ls=":",  color="gray", lw=1, label="Expected mean")
    if any(r["enable_outage"] for r in runs):
        ax.axvspan(outage_start, outage_end, alpha=0.12, color="gray",
                   label="GNSS outage")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("NIS")
    ax.set_ylim(0, 50)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, lw=0.4)

    fig_nis.tight_layout()
    if save_plot:
        fig_nis.savefig(plot_dir / f"KF_NIS_{combined_label}.png",
                        dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Figure 3: Position errors (3D error norm + cumulative RMSE)
    # ------------------------------------------------------------------
    fig_err, axes_err = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_err.suptitle("KF Performance – Position Errors", fontsize=13, fontweight="bold")

    # ── Panel 1: 3D position error norm ───────────────────────────────
    ax = axes_err[0]
    for run, (n, e, d) in zip(runs, ned_runs):
        N = min(len(n), len(gt_n), len(run["out_time"]))
        err = np.sqrt((n[:N] - gt_n[:N])**2
                    + (e[:N] - gt_e[:N])**2
                    + (d[:N] - gt_d[:N])**2)
        ax.plot(run["out_time"][:N], err, lw=0.8, alpha=0.85,
                label=run["case_name"])
    if any(r["enable_outage"] for r in runs):
        ax.axvspan(outage_start, outage_end, alpha=0.12, color="gray",
                   label="GNSS outage")
    ax.set_ylabel("3D position error [m]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # ── Panel 2: Cumulative RMSE ───────────────────────────────────────
    ax = axes_err[1]
    for run, (n, e, d) in zip(runs, ned_runs):
        N = min(len(n), len(gt_n), len(run["out_time"]))
        err = np.sqrt((n[:N] - gt_n[:N])**2
                    + (e[:N] - gt_e[:N])**2
                    + (d[:N] - gt_d[:N])**2)
        rmse_cum = np.sqrt(np.cumsum(err**2) / np.arange(1, N + 1))
        ax.plot(run["out_time"][:N], rmse_cum, lw=1.4,
                label=run["case_name"])
    if any(r["enable_outage"] for r in runs):
        ax.axvspan(outage_start, outage_end, alpha=0.12, color="gray",
                   label="GNSS outage")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative RMSE [m]")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    fig_err.tight_layout()
    if save_plot:
        fig_err.savefig(plot_dir / f"KF_Errors_{combined_label}.png",
                        dpi=300, bbox_inches="tight")

    plt.show()

    # ------------------------------------------------------------------
    # Figure 3: IMU bias estimates
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig3.suptitle("KF IMU Bias Estimates", fontsize=13, fontweight="bold")

    ax = axes3[0]
    for run in runs:
        for i, axis_label in enumerate(["X", "Y", "Z"]):
            ax.plot(run["out_time"], run["out_b_a"][:, i], lw=0.9,
                    label=f"{run['case_name']} — Accel {axis_label}")
    if any(r["enable_outage"] for r in runs):
        ax.axvspan(outage_start, outage_end, alpha=0.12, color="gray",
                   label="GNSS outage")
    ax.set_ylabel("Accel bias [m/s²]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, lw=0.4)

    ax = axes3[1]
    for run in runs:
        gyro_bias_deg = np.rad2deg(run["out_b_g"])
        for i, axis_label in enumerate(["X", "Y", "Z"]):
            ax.plot(run["out_time"], gyro_bias_deg[:, i], lw=0.9,
                    label=f"{run['case_name']} — Gyro {axis_label}")
    if any(r["enable_outage"] for r in runs):
        ax.axvspan(outage_start, outage_end, alpha=0.12, color="gray")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gyro bias [deg/s]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, lw=0.4)

    fig3.tight_layout()
    if save_plot:
        fig3.savefig(plot_dir / f"KF_Biases_{combined_label}.png",
                     dpi=300, bbox_inches="tight")
    plt.show()


def load_gnss(filepath, start_time):
    """Load and crop a GNSS solution CSV (SPP or RTK) to a common start time."""
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] > start_time].reset_index(drop=True)
    df = df.dropna(subset=[
        "X-ECEF", "Y-ECEF", "Z-ECEF",
        "VX-ECEF", "VY-ECEF", "VZ-ECEF"
    ]).reset_index(drop=True)
    return df


def main():
    gps_week = 2415
    dt = 0.01

    run_id = 2
    imu_file = f"data/run{run_id}_imu.txt"
    gt_file  = f"data/run{run_id}_groundtruth.txt"
    spp_file = f"data/run{run_id}_spp_solution.csv"
    rtk_file = f"data/run{run_id}_RTK.csv"

    plot_dir = Path("Plots") / f"Run_{run_id}" / "SPP_vs_RTK_KF"
    plot_dir.mkdir(parents=True, exist_ok=True)

    imu = read_imu_csv(imu_file, gps_week=gps_week)
    gt  = read_ground_truth_csv(gt_file, gps_week=gps_week)
    spp = pd.read_csv(spp_file)
    spp["datetime"] = pd.to_datetime(spp["datetime"])

    start_time = max(
        imu["datetime"].iloc[0],
        gt["datetime"].iloc[0],
        spp["datetime"].iloc[0],
    )

    imu = imu[imu["datetime"] > start_time].reset_index(drop=True)
    gt  = gt[gt["datetime"]  > start_time].reset_index(drop=True)

    # ── Bias correction ───────────────────────────────────────────────
    if run_id == 2:
        gyro_bias_deg = np.array([0.08489004, 0.06587116, 0.29253449])
        accel_bias    = np.array([-9.33629151e-05, -1.54420182e-04, 5.34426553e-03])
    elif run_id == 3:
        gyro_bias_deg = np.array([7.90044E-02, 8.41271E-02, 2.57198E-01])
        accel_bias    = np.array([2.53264E-03, 9.19847E-03, 7.02735E-03])
    elif run_id == 4:
        gyro_bias_deg = np.array([6.84378E-02, 7.71996E-02, 2.45559E-01])
        accel_bias    = np.array([-2.84238E-04, 2.65736E-02, 5.24837E-03])

    accel_raw = imu[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy()
    gyro_raw  = imu[["Gyro_X",  "Gyro_Y",  "Gyro_Z"]].to_numpy()
    accel = ENU_to_NED(accel_raw - accel_bias)
    gyro  = ENU_to_NED(np.deg2rad(gyro_raw - gyro_bias_deg))

    imu_time = (imu["datetime"] - start_time).dt.total_seconds().to_numpy()

    # ── Initial conditions ────────────────────────────────────────────
    r0_ecef = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].iloc[0].to_numpy()
    v0_ecef = gt[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].iloc[0].to_numpy()
    lat0, lon0, alt0 = pm.ecef2geodetic(*r0_ecef)
    g_e = make_gravity_ecef(lat0, lon0, alt0, r0_ecef)

    heading_deg = gt["Heading"].iloc[0] if "Heading" in gt.columns else 113.2533301520
    R_ned   = ned_rotation_matrix(lat0, lon0, alt0, r0_ecef)
    C_b_e0  = R_ned.T @ yaw_rotation_matrix(-np.deg2rad(heading_deg))

    # ── Noise matrices ────────────────────────────────────────────────
    P = np.zeros((15, 15))
    P += 1e-6 * np.eye(15)
    P[0:3,   0:3]   = (5.0**2)               * np.eye(3)
    P[3:6,   3:6]   = (0.5**2)               * np.eye(3)
    P[6:9,   6:9]   = (np.deg2rad(1.0)**2)   * np.eye(3)
    P[9:12,  9:12]  = (0.001**2)             * np.eye(3)
    P[12:15, 12:15] = (np.deg2rad(0.005)**2) * np.eye(3)

    Q = np.zeros((15, 15))
    Q[3:6,   3:6]   = 3.462133832010000e-07 * dt * np.eye(3)
    Q[6:9,   6:9]   = 3.046174197867087e-08 * dt * np.eye(3)
    Q[9:12,  9:12]  = 2.163833645006250e-08 * dt * np.eye(3)
    Q[12:15, 12:15] = 2.350443053909789e-09 * dt * np.eye(3)

    gt_pos = gt[["X-ECEF", "Y-ECEF", "Z-ECEF"]].to_numpy()

    # ── Load SPP and RTK ──────────────────────────────────────────────
    def make_R_and_arrays(df):
        pos = df[["X-ECEF", "Y-ECEF", "Z-ECEF"]].to_numpy()
        vel = df[["VX-ECEF", "VY-ECEF", "VZ-ECEF"]].to_numpy()
        t   = (df["datetime"] - start_time).dt.total_seconds().to_numpy()
        pos_var = np.maximum(df[["std_X",  "std_Y",  "std_Z"]].pow(2).mean().to_numpy(), 1.0**2)
        vel_var = np.maximum(df[["std_VX", "std_VY", "std_VZ"]].pow(2).mean().to_numpy(), 0.1**2)
        R = np.zeros((6, 6))
        R[0:3, 0:3] = np.diag(pos_var)
        R[3:6, 3:6] = np.diag(vel_var)
        return pos, vel, t, R

    spp_df = load_gnss(spp_file, start_time)
    rtk_df = load_gnss(rtk_file, start_time)

    spp_pos, spp_vel, spp_time, R_spp = make_R_and_arrays(spp_df)
    rtk_pos, rtk_vel, rtk_time, R_rtk = make_R_and_arrays(rtk_df)

    # ── Run both KF cases ─────────────────────────────────────────────
    run_spp = run_kf_case(
        f"Run {run_id} SPP", False, dt, accel, gyro, imu_time,
        spp_time, spp_pos, spp_vel,
        r0_ecef, v0_ecef, C_b_e0, P, Q, R_spp, g_e
    )

    run_rtk = run_kf_case(
        f"Run {run_id} RTK", False, dt, accel, gyro, imu_time,
        rtk_time, rtk_pos, rtk_vel,
        r0_ecef, v0_ecef, C_b_e0, P, Q, R_rtk, g_e
    )

    # ── Plot both on the same figures ─────────────────────────────────
    plot_kf_cases([run_spp, run_rtk], gt_pos, lat0, lon0, alt0,
                  plot_dir, save_plot=True)


if __name__ == "__main__":
    main()