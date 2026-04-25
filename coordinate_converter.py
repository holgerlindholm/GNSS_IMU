# Coordinate utils
import numpy as np
import pandas as pd
import pymap3d as pm
from datetime import datetime

from IMU_reader import read_imu_csv, read_ground_truth_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def skew_symmetric(vec):
    """Convert a 3D vector to its 3x3 skew-symmetric matrix."""
    return np.array([
        [ 0,       -vec[2],  vec[1]],
        [ vec[2],   0,      -vec[0]],
        [-vec[1],   vec[0],  0     ]
    ])

def ned_unit_in_ecef(n, e, d, lat0_deg, lon0_deg, alt0_m, r0_ecef):
    """
    Return the ECEF direction vector corresponding to a unit NED displacement.

    pm.ned2ecef returns an absolute ECEF coordinate, so subtracting the
    origin converts it to a direction vector.

    Parameters:
    n, e, d      : NED displacement components (one should be 1, rest 0)
    lat0_deg     : reference latitude  (degrees)
    lon0_deg     : reference longitude (degrees)
    alt0_m       : reference altitude  (metres)
    r0_ecef      : (3,) ECEF origin corresponding to (lat0, lon0, alt0)

    Returns:
    (3,) unit direction vector in ECEF
    """
    x, y, z = pm.ned2ecef(n, e, d, lat0_deg, lon0_deg, alt0_m)
    return np.array([x, y, z]) - r0_ecef

def ned_rotation_matrix(lat0_deg, lon0_deg, alt0_m, r0_ecef):
    """
    Build the 3x3 rotation matrix R_ned that maps ECEF vectors to NED vectors:
        v_ned = R_ned @ v_ecef

    Rows of R_ned are the N, E, D unit axes expressed in ECEF coordinates.
    Its transpose C_b_e = R_ned.T maps NED (body) vectors to ECEF vectors.

    Parameters:
    lat0_deg : reference latitude  (degrees)
    lon0_deg : reference longitude (degrees)
    alt0_m   : reference altitude  (metres)
    r0_ecef  : (3,) ECEF origin

    Returns:
    R_ned    : (3,3) rotation matrix, ECEF -> NED
    """
    north = ned_unit_in_ecef(1, 0, 0, lat0_deg, lon0_deg, alt0_m, r0_ecef)
    east  = ned_unit_in_ecef(0, 1, 0, lat0_deg, lon0_deg, alt0_m, r0_ecef)
    down  = ned_unit_in_ecef(0, 0, 1, lat0_deg, lon0_deg, alt0_m, r0_ecef)

    # Each row is one NED axis expressed in ECEF
    return np.vstack([north, east, down])

def yaw_rotation_matrix(heading_rad):
    """Rotation about Down axis (NED z-axis)"""
    c = np.cos(heading_rad)
    s = np.sin(heading_rad)
    return np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1]
    ])

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GRAVITY_MAGNITUDE = 9.81558596948718  # m/s²  (WGS-84 at 55.783°N, 12.517°E, 49 m)
tau      = 0.01                       # IMU sampling period (s) — 100 Hz
OMEGA_IE = 7.29211585494e-5           # Earth rotation rate (rad/s)
Omega_ie_e = skew_symmetric(np.array([0.0, 0.0, OMEGA_IE]))


# ---------------------------------------------------------------------------
# Mechanisation equations (ECEF frame)
# ---------------------------------------------------------------------------

def attitude_update(C_b_e_minus, omega_ib_b, tau):
    """
    Update the rotation matrix C_b^e using first-order integration.

    From Daniel's PhD thesis, eq. (4.6):
        C_b^{e+} = C_b^{e-} (I + Omega_{ib}^b * tau) - Omega_{ie}^e * C_b^{e-} * tau

    Parameters:
    C_b_e_minus : (3,3) previous body-to-ECEF rotation matrix
    omega_ib_b  : (3,)  gyroscope measurement in body frame (rad/s)
    tau         : scalar sampling interval (s)

    Returns:
    C_b_e_plus  : (3,3) updated body-to-ECEF rotation matrix
    """
    Omega_ib_b = skew_symmetric(omega_ib_b)
    C_b_e_plus = (C_b_e_minus @ (np.eye(3) + Omega_ib_b * tau)
                  - Omega_ie_e @ C_b_e_minus * tau)
    return C_b_e_plus


def velocity_update(v_eb_e_minus, C_b_e_minus, C_b_e_plus, f_ib_b, g, tau):
    """
    Update velocity in the ECEF frame.

    From Daniel's PhD thesis, eqs. (4.7)-(4.8):
        f_{ib}^e = 0.5 (C_b^{e-} + C_b^{e+}) f_{ib}^b
        v_{eb}^{e+} = v_{eb}^{e-} + (f_{ib}^e + g - 2 Omega_{ie}^e v_{eb}^{e-}) * tau

    Parameters:
    v_eb_e_minus : (3,) previous ECEF velocity (m/s)
    C_b_e_minus  : (3,3) previous body-to-ECEF rotation matrix
    C_b_e_plus   : (3,3) updated body-to-ECEF rotation matrix
    f_ib_b       : (3,)  accelerometer measurement in body frame (m/s²)
    g            : (3,)  gravity vector in ECEF frame (m/s²)
    tau          : scalar sampling interval (s)

    Returns:
    v_eb_e_plus  : (3,) updated ECEF velocity (m/s)
    f_ib_e       : (3,) specific force rotated to ECEF frame (m/s²)
    """
    f_ib_e      = 0.5 * (C_b_e_minus + C_b_e_plus) @ f_ib_b
    v_eb_e_plus = (v_eb_e_minus
                   + (f_ib_e + g - 2 * Omega_ie_e @ v_eb_e_minus) * tau)
    return v_eb_e_plus, f_ib_e


def position_update(r_eb_minus, f_ib_e, v_eb_e_plus, v_eb_e_minus, g, tau):
    """
    Update position in the ECEF frame.

    From Daniel's PhD thesis, eq. (4.9):
        r_{eb}^{e+} = r_{eb}^{e-}
                    + v_{eb}^{e+} * tau
                    + (f_{ib}^e + g - 2 Omega_{ie}^e v_{eb}^{e-}) * tau^2 / 2

    Parameters:
    r_eb_minus   : (3,) previous ECEF position (m)
    f_ib_e       : (3,) specific force in ECEF frame (m/s²)
    v_eb_e_plus  : (3,) updated ECEF velocity (m/s)
    v_eb_e_minus : (3,) previous ECEF velocity (m/s)
    g            : (3,)  gravity vector in ECEF frame (m/s²)
    tau          : scalar sampling interval (s)

    Returns:
    r_eb_plus    : (3,) updated ECEF position (m)
    """
    r_eb_plus = (r_eb_minus
                 + v_eb_e_plus * tau
                 + (f_ib_e + g - 2 * Omega_ie_e @ v_eb_e_minus) * tau**2 / 2)
    return r_eb_plus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    imu_file          = r"data\run2_imu.txt"
    ground_truth_file = r"data\run2_groundtruth.txt"

    df_imu   = read_imu_csv(imu_file, gps_week=2415)
    df_imu = df_imu[df_imu["datetime"]>datetime(2026,4,23,7,33)]

    df_truth = read_ground_truth_csv(ground_truth_file, gps_week=2415)

    print("IMU columns  :", df_imu.columns.tolist())
    print(f"IMU samples  : {len(df_imu)}")

    gyro  = df_imu[["Gyro_X",  "Gyro_Y",  "Gyro_Z" ]].to_numpy()  # (N,3) deg/s
    accel = df_imu[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy()  # (N,3) m/s²
    N     = len(df_imu)

    # Bias correction
    gyro_bias  = np.array([0.0855,  0.0780,  0.2700])   # deg/s

    accel_bias = np.array([-0.3237, 0.0300, 0])  # m/s²

    gyro  = gyro  - gyro_bias
    gyro = np.deg2rad(gyro) # rad/s
    accel = accel - accel_bias

    # ------------------------------------------------------------------
    # 2. Initial conditions from ground truth
    # ------------------------------------------------------------------
    r0_ecef = df_truth[["X-ECEF", "Y-ECEF", "Z-ECEF"]].iloc[0].to_numpy()
    print(f"r0 ECEF: {r0_ecef}")

    # Geodetic coordinates of the starting point (degrees, degrees, metres)
    lat0_deg, lon0_deg, alt0_m = pm.ecef2geodetic(*r0_ecef)
    print(f"r0 geodetic: lat={lat0_deg:.6f}  lon={lon0_deg:.6f}  alt={alt0_m:.2f} m")

    # ----------------------------------------
    # Set your known initial heading here
    # ----------------------------------------
    heading_deg = 113.2533301520   # <-- example (0° = North, 90° = East)
    heading_rad = np.deg2rad(heading_deg)

    # Gravity vector in ECEF:
    #   In NED, gravity points straight down: [0, 0, +g_magnitude].
    #   pm.ned2ecef converts that NED displacement to an absolute ECEF point;
    #   subtracting the ECEF origin turns it into a direction vector.
    gx, gy, gz = pm.ned2ecef(0.0, 0.0, GRAVITY_MAGNITUDE, lat0_deg, lon0_deg, alt0_m)
    g = np.array([gx, gy, gz]) - r0_ecef   # (3,) gravity in ECEF (m/s²)
    print(f"g ECEF: {g}")

    # Initial attitude: body frame is North-East-Up (NEU), *not* NED.
    # Evidence: Accel_Z ≈ +9.815 when stationary → body Z points UP.
    # A Z-down (NED) sensor would read −9.815 instead.
    #
    # C_b_e maps body vectors → ECEF:
    #   C_b_e = C_NED_to_ECEF  @  C_NEU_to_NED
    #         = R_ned.T         @  diag(1, 1, −1)
    #
    # diag(1,1,−1) flips the Z axis so body-Z (Up) → NED-D (Down) = negated.
    # This ensures f_ib_e = C_b_e @ [0, 0, +g] points radially outward and
    # exactly cancels g_e in the velocity update when the vehicle is stationary.
    R_ned       = ned_rotation_matrix(lat0_deg, lon0_deg, alt0_m, r0_ecef)
    C_b_e_minus = R_ned.T @ np.diag([1.0, 1.0, -1.0])   # body NEU → ECEF

    # Base alignment (NEU → NED)
    C_neu_to_ned = np.diag([1.0, 1.0, -1.0])

    # Apply heading rotation
    R_yaw = yaw_rotation_matrix(-heading_rad)

    # Final initialization
    C_b_e_minus = R_ned.T @ R_yaw @ C_neu_to_ned

    # Body forward vector in ECEF
    forward_ecef = C_b_e_minus @ np.array([1, 0, 0])

    # Convert to NED
    forward_ned = R_ned @ forward_ecef

    print("Initial heading vector (NED):", forward_ned)

    v_eb_e_minus = np.zeros(3)   # start at rest (m/s)
    r_eb_minus   = r0_ecef.copy()

    # ------------------------------------------------------------------
    # 3. Pre-allocate output arrays
    # ------------------------------------------------------------------
    positions  = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    positions[0]  = r_eb_minus
    velocities[0] = v_eb_e_minus

    # ------------------------------------------------------------------
    # 4. Integration loop
    # ------------------------------------------------------------------
    for k in range(1, N):
        omega_k = gyro[k]
        f_k     = accel[k]

        C_b_e_plus          = attitude_update(C_b_e_minus, omega_k, tau)
        v_eb_e_plus, f_ib_e = velocity_update(v_eb_e_minus, C_b_e_minus,
                                               C_b_e_plus, f_k, g, tau)
        r_eb_plus           = position_update(r_eb_minus, f_ib_e,
                                              v_eb_e_plus, v_eb_e_minus,
                                              g, tau)

        positions[k]  = r_eb_plus
        velocities[k] = v_eb_e_plus

        C_b_e_minus  = C_b_e_plus
        v_eb_e_minus = v_eb_e_plus
        r_eb_minus   = r_eb_plus

    # ------------------------------------------------------------------
    # 5. Convert ECEF -> NED using pymap3d  (fully vectorised)
    #    pm.ecef2ned returns displacements in metres relative to r0.
    # ------------------------------------------------------------------
    north_m, east_m, down_m = pm.ecef2ned(
        positions[:, 0], positions[:, 1], positions[:, 2],
        lat0_deg, lon0_deg, alt0_m
    )

    # Velocities are free vectors — rotate with R_ned (no translation needed)
    vel_ned = (R_ned @ velocities.T).T   # (N,3)  North / East / Down  m/s

    # ------------------------------------------------------------------
    # 6. Build results DataFrame and report
    # ------------------------------------------------------------------
    times = (df_imu["GPS_TOW"].to_numpy()
             if "GPS_TOW" in df_imu.columns
             else np.arange(N) * tau)

    results = pd.DataFrame({
        "time_s":   times,
        "North_m":  north_m,
        "East_m":   east_m,
        "Down_m":   down_m,
        "Vel_N_ms": vel_ned[:, 0],
        "Vel_E_ms": vel_ned[:, 1],
        "Vel_D_ms": vel_ned[:, 2],
        "ECEF_X_m": positions[:, 0],
        "ECEF_Y_m": positions[:, 1],
        "ECEF_Z_m": positions[:, 2],
    })

    print("\n--- First 5 rows ---")
    print(results.head().to_string(index=False))
    print("\n--- Last 5 rows ---")
    print(results.tail().to_string(index=False))
    print(f"\nFinal displacement — North: {north_m[-1]:.3f} m"
          f"  East: {east_m[-1]:.3f} m  Down: {down_m[-1]:.3f} m")
    print(f"Total distance (2D): {np.hypot(north_m[-1], east_m[-1]):.3f} m")

    out_csv = r"data\run2_trajectory.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nTrajectory saved to {out_csv}")

    return results


if __name__ == "__main__":
    main()