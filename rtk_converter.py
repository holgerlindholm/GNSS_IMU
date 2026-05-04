import pandas as pd
import numpy as np
from datetime import datetime, timedelta
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def gpst_string_to_datetime(gpst_str: str) -> datetime:
    """
    Parse the GPST timestamp written by RTKLIB / RTKPost.
    Expected format: '2024/01/05 00:00:00.000'
    """
    return datetime.strptime(gpst_str.strip(), "%Y/%m/%d %H:%M:%S.%f")
 
 
def compute_pdop_from_stds(sdx: pd.Series, sdy: pd.Series, sdz: pd.Series) -> pd.Series:
    """
    Approximate PDOP from the diagonal position standard deviations.
    True PDOP requires the full DOP matrix, but this gives a reasonable
    scalar summary: PDOP ≈ sqrt(σx² + σy² + σz²) [metres, not dimensionless].
    If you need the true dimensionless DOP, provide it separately.
    """
    return np.sqrt(sdx**2 + sdy**2 + sdz**2)
 
 
# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------
 
def read_rtk_pos(file_path: str) -> pd.DataFrame | None:
    """
    Read an RTKLIB .pos file and return a DataFrame that matches the schema:
 
        datetime, X-ECEF, Y-ECEF, Z-ECEF, cdt,
        std_X, std_Y, std_Z, PDOP, nsats,
        VX-ECEF, VY-ECEF, VZ-ECEF, cdtdot,
        std_VX, std_VY, std_VZ, std_V3D
 
    Notes
    -----
    * cdt / cdtdot  – not present in .pos files; filled with NaN.
    * PDOP          – approximated as sqrt(sdx²+sdy²+sdz²).
    * Velocities    – central finite difference of ECEF positions w.r.t. time.
    * Velocity stds – error-propagation of position stds via finite differences:
                      std_V ≈ sqrt(2) * std_pos / Δt  (forward/backward: sqrt(2),
                      central: effectively the same order).
    * std_V3D       – sqrt(std_VX² + std_VY² + std_VZ²).
 
    Parameters
    ----------
    file_path : str
        Path to the RTKLIB .pos output file.
 
    Returns
    -------
    pd.DataFrame or None
    """
    try:
        # ------------------------------------------------------------------ #
        # 1. Read raw file, skipping comment / header lines (start with '%')  #
        # ------------------------------------------------------------------ #
        raw_cols = [
            'date', 'time',            # GPST split into two tokens
            'x_ecef', 'y_ecef', 'z_ecef',
            'Q', 'ns',
            'sdx', 'sdy', 'sdz',
            'sdxy', 'sdyz', 'sdzx',
            'age', 'ratio'
        ]
 
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            comment='%',
            header=None,
            names=raw_cols,
            engine='python',
        )
 
        # ------------------------------------------------------------------ #
        # 2. Parse timestamps                                                  #
        # ------------------------------------------------------------------ #
        gpst_strings = df['date'].str.strip() + ' ' + df['time'].str.strip()
        df['datetime'] = pd.to_datetime(gpst_strings, format='%Y/%m/%d %H:%M:%S.%f')
 
        # ------------------------------------------------------------------ #
        # 3. Rename / carry-over columns                                       #
        # ------------------------------------------------------------------ #
        df.rename(columns={
            'x_ecef': 'X-ECEF',
            'y_ecef': 'Y-ECEF',
            'z_ecef': 'Z-ECEF',
            'sdx':    'std_X',
            'sdy':    'std_Y',
            'sdz':    'std_Z',
            'ns':     'nsats',
        }, inplace=True)
 
        # Columns not available in .pos → NaN placeholders
        df['cdt']    = np.nan
        df['cdtdot'] = np.nan
 
        # ------------------------------------------------------------------ #
        # 4. PDOP approximation                                                #
        # ------------------------------------------------------------------ #
        df['PDOP'] = compute_pdop_from_stds(df['std_X'], df['std_Y'], df['std_Z'])
 
        # ------------------------------------------------------------------ #
        # 5. Velocity via central finite differences                           #
        #                                                                      #
        #   v[i] = (pos[i+1] - pos[i-1]) / (t[i+1] - t[i-1])               #
        #   Edges use one-sided (forward / backward) differences.             #
        # ------------------------------------------------------------------ #
        dt = 0.1
        for axis in ('X', 'Y', 'Z'):

            pos = df[f'{axis}-ECEF']
            
            # Central difference
            vel = (pos.shift(-1) - pos.shift(1)) / dt
            # Forward/backward for edges
            vel.iloc[0]  = (pos.iloc[1] - pos.iloc[0]) / dt
            vel.iloc[-1] = (pos.iloc[-1] - pos.iloc[-2]) / dt
            
            df[f'V{axis}-ECEF'] = vel

            # --- Std propagation ---
            std = df[f'std_{axis}']
            
            std_v = np.sqrt(std.shift(-1)**2 + std.shift(1)**2) / dt
            std_v.iloc[0]  = np.sqrt(std.iloc[0]**2 + std.iloc[1]**2) / dt
            std_v.iloc[-1] = np.sqrt(std.iloc[-2]**2 + std.iloc[-1]**2) / dt

            df[f'std_V{axis}'] = std_v
 
        # ------------------------------------------------------------------ #
        # 6. 3-D velocity standard deviation                                   #
        # ------------------------------------------------------------------ #
        df['std_V3D'] = np.sqrt(
            df['std_VX']**2 +
            df['std_VY']**2 +
            df['std_VZ']**2
        )
 
        # ------------------------------------------------------------------ #
        # 7. Select and order final columns                                    #
        # ------------------------------------------------------------------ #
        output_cols = [
            'datetime',
            'X-ECEF', 'Y-ECEF', 'Z-ECEF',
            'cdt',
            'std_X', 'std_Y', 'std_Z',
            'PDOP', 'nsats',
            'VX-ECEF', 'VY-ECEF', 'VZ-ECEF',
            'cdtdot',
            'std_VX', 'std_VY', 'std_VZ',
            'std_V3D',
        ]
 
        return df[output_cols].reset_index(drop=True)
 
    except Exception as e:
        print(f"Error reading RTK .pos file: {e}")
        return None
 
 
# ---------------------------------------------------------------------------
# Optional: write the result to CSV
# ---------------------------------------------------------------------------
 
def rtk_pos_to_csv(input_path: str, output_path: str) -> None:
    """Read a .pos file and write the converted DataFrame to CSV."""
    df = read_rtk_pos(input_path)
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
    else:
        print("Conversion failed.")


# ---------------------------------------------------------------------------
# Quick self-test with a synthetic .pos snippet
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    df = read_rtk_pos(r"C:\Users\holge\git\GNSS_IMU\data\run3_BUDD.pos")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top pane: velocities ---
    axs[0].plot(df["datetime"], df["VX-ECEF"], label="VX")
    axs[0].plot(df["datetime"], df["VY-ECEF"], label="VY")
    axs[0].plot(df["datetime"], df["VZ-ECEF"], label="VZ")
    axs[0].set_ylabel("Velocity (ECEF)")
    axs[0].legend()
    axs[0].grid()

    # --- Bottom pane: std dev ---
    axs[1].plot(df["datetime"], df["std_VX"], label="std VX")
    axs[1].plot(df["datetime"], df["std_VY"], label="std VY")
    axs[1].plot(df["datetime"], df["std_VZ"], label="std VZ")
    axs[1].set_ylabel("Std (ECEF)")
    axs[1].set_xlabel("Time")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()
    # rtk_pos_to_csv(r"C:\Users\holge\git\GNSS_IMU\data\run3_BUDD.pos",r"C:\Users\holge\git\GNSS_IMU\data\run3_RTK.csv")

    