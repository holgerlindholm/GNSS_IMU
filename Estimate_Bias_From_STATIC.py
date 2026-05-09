import pandas as pd
import numpy as np
from coordinate_converter import GRAVITY_MAGNITUDE
from IMU_reader import read_imu_csv

# ---------------------------------------------------------------------------
# CONVENTION
# ---------------------------------------------------------------------------
# ENU frame:  East=X, North=Y, Up=Z
# Gravity in ENU: g_ENU = [0, 0, -G]  (pulls DOWN)
#
# Specific force (what the accelerometer measures, stationary, no bias):
#   f_body = R_nb.T @ (-g_ENU) = R_nb.T @ [0, 0, +G]
#
# Expected body-frame reading for LEVEL imu: [0, 0, G]
# If X or Y are non-zero → the imu is TILTED → roll/pitch ≠ 0
#
# Bias model:
#   a_measured = R_nb.T @ [0, 0, G] + ba
#   =>  ba = a_measured - R_nb.T @ [0, 0, G]
# ---------------------------------------------------------------------------

static_imu_file = r"data/static_imu.txt"
df = read_imu_csv(static_imu_file, gps_week=2415)

G   = GRAVITY_MAGNITUDE          # 9.815586 m/s²
YAW = 286                   # degrees — known from external source

accel_cols = ("Accel_X", "Accel_Y", "Accel_Z")
gyro_cols  = ("Gyro_X",  "Gyro_Y",  "Gyro_Z")


# ---------------------------------------------------------------------------
# STEP 1 — mean measurements
# ---------------------------------------------------------------------------
a_mean = df[list(accel_cols)].mean().values   # [m/s²]
w_mean = df[list(gyro_cols)].mean().values    # [rad/s]

print(f"Mean accel (body): {a_mean}")
print(f"Mean gyro  (body): {w_mean}")


# ---------------------------------------------------------------------------
# STEP 2 — estimate roll & pitch from static accelerometer
# ---------------------------------------------------------------------------
# For a stationary IMU (no bias yet):
#   fx ≈ -G·sin(pitch)
#   fy ≈  G·cos(pitch)·sin(roll)
#   fz ≈  G·cos(pitch)·cos(roll)
#
# → Use the mean accel directly to infer tilt angles.
#   (Yaw is unobservable from accelerometers alone — supplied externally.)

fx, fy, fz = a_mean

pitch_rad = np.arctan2(-fx, np.sqrt(fy**2 + fz**2))   # θ
roll_rad  = np.arctan2( fy, fz)                        # φ
yaw_rad   = np.radians(YAW)                            # ψ — known

pitch_deg = np.degrees(pitch_rad)
roll_deg  = np.degrees(roll_rad)

print(f"\nEstimated attitude:")
print(f"  Roll  φ = {roll_deg:+.4f} °")
print(f"  Pitch θ = {pitch_deg:+.4f} °")
print(f"  Yaw   ψ = {YAW:+.4f} °  (given)")


# ---------------------------------------------------------------------------
# STEP 3 — full rotation matrix  R_nb = Rz(ψ) @ Ry(θ) @ Rx(φ)
# ---------------------------------------------------------------------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

R_nb = Rz(yaw_rad) @ Ry(pitch_rad) @ Rx(roll_rad)   # body → ENU

# Gravity projected into body frame
g_nav  = np.array([0.0, 0.0, -G])         # gravity in ENU
g_body = R_nb.T @ (-g_nav)                # = R_nb.T @ [0, 0, G]
                                           # ≈ a_mean for a well-calibrated IMU

print(f"\nGravity in body frame (predicted): {g_body}")
print(f"Mean measured accel  (actual)    : {a_mean}")
print(f"Residual g-projection error      : {a_mean - g_body}")


# ---------------------------------------------------------------------------
# STEP 4 — accelerometer bias
# ---------------------------------------------------------------------------
ba_mean = a_mean - g_body

# Per-sample for std / noise floor
a_samples  = df[list(accel_cols)].values          # (N, 3)
ba_samples = a_samples - g_body[np.newaxis, :]
ba_std     = ba_samples.std(axis=0)


# ---------------------------------------------------------------------------
# GYROSCOPE BIAS — corrected
# ---------------------------------------------------------------------------

# UNIT CHECK — print raw values before any conversion
print(f"Raw gyro mean from CSV : {w_mean}")
# If these match expected (~0.086, 0.079, 0.250), the data is already in deg/s.
# If ~57× too large, data is in rad/s and needs np.degrees().
# Adjust GYRO_UNIT below accordingly.

GYRO_UNIT = "deg/s"   # ← set to "rad/s" if your reader returns rad/s

if GYRO_UNIT == "rad/s":
    to_deg = np.degrees          # conversion function
    to_rad = lambda x: x         # already rad/s for storage
else:
    to_deg = lambda x: x         # already deg/s — DO NOT multiply by 57.3 again
    to_rad = np.radians


# ---------------------------------------------------------------------------
# Filter to truly static samples before computing bias
# ---------------------------------------------------------------------------
# A "static" sample has near-zero angular rate on all axes.
# Large gyro readings during a nominally static window = vehicle vibration,
# micro-motion, or bad data — they bias the mean if included.

gyro_data   = df[list(gyro_cols)].values          # (N, 3), in GYRO_UNIT
gyro_norm   = np.linalg.norm(gyro_data, axis=1)   # scalar per sample

# Threshold: 5× expected bias magnitude is a generous but safe cutoff
expected_bg_magnitude = 0.30   # deg/s  (slightly above largest expected axis)
static_threshold      = 5.0 * expected_bg_magnitude   # 1.5 deg/s

if GYRO_UNIT == "rad/s":
    static_threshold = np.radians(static_threshold)

static_mask   = gyro_norm < static_threshold
n_total       = len(df)
n_static      = static_mask.sum()

print(f"\nStatic sample filter:")
print(f"  Threshold : {static_threshold:.4f} {GYRO_UNIT}")
print(f"  Kept      : {n_static} / {n_total}  ({100*n_static/n_total:.1f} %)")

if n_static < 100:
    print("  WARNING: very few static samples — check threshold or data window")

# Use only static samples
gyro_static = gyro_data[static_mask]
accel_static = df[list(accel_cols)].values[static_mask]

# Gyro bias: for a stationary IMU, mean(ω) = bg  (Earth rate << sensor noise)
bg_mean_raw = gyro_static.mean(axis=0)      # in GYRO_UNIT
bg_std_raw  = gyro_static.std(axis=0)

bg_mean_deg = to_deg(bg_mean_raw)           # always deg/s for display
bg_std_deg  = to_deg(bg_std_raw)

# Recompute accel bias using only static samples
a_mean_static = accel_static.mean(axis=0)

# (Re-estimate roll/pitch from static-filtered accel for cleanliness)
fx, fy, fz    = a_mean_static
pitch_rad     = np.arctan2(-fx, np.sqrt(fy**2 + fz**2))
roll_rad      = np.arctan2(fy, fz)
R_nb          = Rz(yaw_rad) @ Ry(pitch_rad) @ Rx(roll_rad)
g_body        = R_nb.T @ np.array([0.0, 0.0, G])

ba_mean       = a_mean_static - g_body
ba_std        = accel_static.std(axis=0)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("  IMU BIAS ESTIMATION — FINAL RESULTS")
print("="*55)

print(f"\n  Attitude:  roll={np.degrees(roll_rad):+.3f}°  "
      f"pitch={np.degrees(pitch_rad):+.3f}°  yaw={YAW:+.3f}°")

print(f"\n  ACCELEROMETER BIAS  [m/s²]")
print(f"    Estimated : {ba_mean}")
print(f"    Std       : {ba_std}")
print(f"    Expected  : [-0.0036,  0.0118,  0.0095]")

print(f"\n  GYROSCOPE BIAS  [{GYRO_UNIT}]")
print(f"    Estimated : {bg_mean_deg}")
print(f"    Std       : {bg_std_deg}")
print(f"    Expected  : [0.0859,  0.0787,  0.2498]")

# Store bias in consistent units for downstream use
bias = {
    "ba_mps2"    : ba_mean,                         # m/s²
    "bg_deg_s"   : bg_mean_deg,                     # deg/s
    "bg_rad_s"   : to_rad(bg_mean_raw),             # rad/s
    "roll_deg"   : np.degrees(roll_rad),
    "pitch_deg"  : np.degrees(pitch_rad),
    "yaw_deg"    : YAW,
    "n_static"   : int(n_static),
}
print("="*55)