import numpy as np
import pandas as pd
from tqdm import tqdm
import pymap3d as pm

import rinexReader as rr
import SatOrbits as so


class SPP:
    def __init__(self, obs_file_path, sp3_file_path,
                 consts=["G", "E", "C"], sig_types=["C1C", "L1C"],
                 hatch_filter=False, trop=True):
        self.clight = 299792458
        self.obs_file_path = obs_file_path
        self.sp3_file_path = sp3_file_path
        self.consts = consts
        self.sig_types = sig_types
        self.hatch_filter = hatch_filter
        self.trop = trop

        self.rinex = rr.rinexReader(obs_file_path)
        self.orbits = so.sp3Orbits(sp3_file_path)
        self.rinex.readFile(self.consts, self.sig_types)

    def _create_kernel(self, obs, satpos, x):
        """Build design matrix A and data vector L for one NLLS iteration."""
        rng = np.linalg.norm(satpos - x[:3], axis=1)
        A_xyz = (x[:3] - satpos) / rng[:, None]
        clk = pd.Series(np.ones(len(A_xyz)), index=A_xyz.index)
        A = pd.concat([A_xyz, clk], axis=1)
        L = obs - rng - x[3]   # obs is now a Series, no [:, None] needed
        return L, A

    def _spp(self, obs, satpos, x0):
        """Solve single-point position via NLLS. Returns (solution, Qx, s0)."""
        tol = 1e-3
        max_iter = 50
        x = x0.copy()
        h = np.full(4, np.inf)

        for _ in range(max_iter):
            if np.linalg.norm(h) <= tol:
                break
            L, A = self._create_kernel(obs, satpos, x)
            N = A.T @ A
            c = A.T @ L
            h = np.linalg.solve(N, c)
            x += h

        solution = pd.Series(x, index=["X", "Y", "Z", "cdt"], name="Solution")
        v = L.values - (A.values @ h)
        s0 = (v @ v) / max(len(L) - 4, 1)
        Qx = np.linalg.inv(A.T @ A)
        return solution, Qx, s0

    def _elevation_angle(self, satpos_xyz, receiver_xyz):
        """
        Compute satellite elevation angles in radians using ENU projection.

        Parameters
        ----------
        satpos_xyz : pd.DataFrame, shape (N, 3)  — satellite ECEF positions
        receiver_xyz : array-like, shape (3,)    — receiver ECEF position

        Returns
        -------
        pd.Series of elevation angles (radians), indexed like satpos_xyz
        """
        rx, ry, rz = receiver_xyz[:3]

        # Receiver geodetic coords for ENU rotation
        lat, lon, _ = pm.ecef2geodetic(rx, ry, rz)
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)

        # ENU rotation matrix rows (only need 'Up' row for elevation)
        #   East : [-sin_lon,          cos_lon,         0        ]
        #   North: [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat  ]
        #   Up   : [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat  ]
        up = np.array([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r)
        ])

        los = satpos_xyz.values - np.array([rx, ry, rz])   # (N, 3)
        rng = np.linalg.norm(los, axis=1)                  # (N,)
        sin_el = (los @ up) / rng                          # dot with Up unit vector

        # Clamp to valid range to avoid NaN from floating-point noise
        sin_el = np.clip(sin_el, -1.0, 1.0)
        return pd.Series(np.arcsin(sin_el), index=satpos_xyz.index)

    def _tropoCorr(self, satpos_xyz, receiver_xyz):
        """
        Zenith tropospheric delay via Saastamoinen (standard atmosphere),
        mapped to slant range using the correct elevation angle.
        """
        el = self._elevation_angle(satpos_xyz, receiver_xyz)  # radians

        # Mask out low-elevation satellites (< 5°) to avoid huge corrections
        min_el = np.radians(5.0)
        el = el.clip(lower=min_el)

        T = 288.15      # K
        P = 1013.25     # hPa
        e = 11.691      # hPa (partial water vapour pressure)

        lat_r = np.radians(pm.ecef2geodetic(*receiver_xyz[:3])[0])
        ZHD = 0.0022768 * P / (1 - 0.00266 * np.cos(2 * lat_r) - 0.00028 * 0)
        ZWD = 0.0022768 * (1255 / T + 0.05) * e

        # Simple 1/sin(el) mapping function
        tropo = (ZHD + ZWD) / np.sin(el)
        return tropo  # pd.Series, indexed like satpos_xyz

    def _hatch_filter(self, obs, prev_state, wavelength, N=100):
        """
        Apply Hatch filter with cycle slip detection and continuity checks.
        """
        smoothed = {}
        new_state = {}

        for sv in obs.index:
            P = obs.loc[sv, "C1C"]
            L = obs.loc[sv, "L1C"]
            
            # 1. Check if we have valid phase and code
            if np.isnan(P) or np.isnan(L):
                continue

            # 2. Check for Continuity / Cycle Slip
            # Logic: If SV was missing last epoch, or phase jumped > 0.5m (geometry-free)
            # For simplicity here, we check if SV was in prev_state.
            # In a production app, you'd also check the RINEX LLI flag.
            
            is_continuous = False
            if sv in prev_state:
                prev = prev_state[sv]
                # Simple Cycle Slip Detection: phase jump vs code jump
                # If the difference between Phase change and Code change is > 3m, reset.
                phase_increment = (L - prev["L_prev"]) * wavelength
                code_increment = P - prev["P_smooth"]
                
                if abs(phase_increment - code_increment) < 3.0: 
                    is_continuous = True

            if not is_continuous:
                # Reset the filter for this satellite
                smoothed[sv] = P
                new_state[sv] = {"P_smooth": P, "L_prev": L, "n": 1}
            else:
                prev = prev_state[sv]
                n = min(prev["n"] + 1, N)
                
                # The Hatch Equation
                # P_smooth = (1/n) * P_code + ((n-1)/n) * (P_smooth_prev + delta_Phase)
                P_smooth = (1/n) * P + ((n-1)/n) * (prev["P_smooth"] + wavelength * (L - prev["L_prev"]))

                smoothed[sv] = P_smooth
                new_state[sv] = {"P_smooth": P_smooth, "L_prev": L, "n": n}

        return pd.Series(smoothed), new_state

    def run(self, x0=None):
        """Run SPP for all epochs. Returns dict keyed by epoch."""
        lambda_L1 = self.clight / 1575.42e6

        if x0 is None:
            x0 = np.zeros(4)
        x0 = np.array(x0, dtype=float)

        solutions = {}
        hatch_state = {}

        for epoch in tqdm(self.rinex.timelist):
            obs = self.rinex.get_epoch_data(epoch, oTypes=self.sig_types).dropna()
            if len(obs) < 4:
                continue

            # Select pseudorange only
            if self.hatch_filter:
                obs_used, hatch_state = self._hatch_filter(obs, hatch_state, lambda_L1)
            else:
                obs_used = obs["C1C"]   # ← Series, not DataFrame

            # Satellite positions
            tau = obs_used / self.clight
            satpos = self.orbits.getSvPos(epoch, tau)
            cdts = satpos.iloc[:, 3] * self.clight
            satpos_xyz = satpos.iloc[:, :3]

            # Apply satellite clock correction to pseudorange Series
            obs_corr = obs_used + cdts.values

            # Apply tropospheric correction using CURRENT best position
            if self.trop and np.linalg.norm(x0[:3]) > 1e4:  # skip until we have a real position
                tropo = self._tropoCorr(satpos_xyz, x0)
                obs_corr = obs_corr - tropo

            x, Qx, s0 = self._spp(obs_corr, satpos_xyz, x0)
            solutions[epoch] = {"solution": x, "covariance": Qx, "variance": s0}

            # ← Update linearization point for next epoch
            x0 = x.values

        return solutions

    def _compute_velocity(self, solutions, dt=0.1):
        epochs = sorted(solutions.keys())
        rows = []
        for i in range(1, len(epochs)):
            t0, t1 = epochs[i - 1], epochs[i]
            actual_dt = (t1 - t0).total_seconds()
            if abs(actual_dt - dt) > dt * 0.5:
                continue
            p0 = solutions[t0]["solution"]
            p1 = solutions[t1]["solution"]
            rows.append({
                "datetime": t1,
                "VX-ECEF": (p1["X"] - p0["X"]) / actual_dt,
                "VY-ECEF": (p1["Y"] - p0["Y"]) / actual_dt,
                "VZ-ECEF": (p1["Z"] - p0["Z"]) / actual_dt,
            })
        return pd.DataFrame(rows)

    def export_csv(self, solutions, output_path="spp_results.csv"):
        pos_rows = [
            {
                "datetime": epoch,
                "X-ECEF": data["solution"]["X"],
                "Y-ECEF": data["solution"]["Y"],
                "Z-ECEF": data["solution"]["Z"],
                "cdt":    data["solution"]["cdt"],
                "variance": data["variance"],
            }
            for epoch, data in solutions.items()
        ]
        pos_df = pd.DataFrame(pos_rows)
        vel_df = self._compute_velocity(solutions)
        combined_df = pd.merge(pos_df, vel_df, on="datetime", how="outer")
        combined_df.sort_values("datetime").to_csv(output_path, index=False)
        print(f"Exported to {output_path}")


if __name__ == "__main__":
    # obs_file = r"data/run2.obs"
    # sp3_file = r"data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3"

    # spp_solver = SPP(obs_file, sp3_file, hatch_filter=True, trop=True)
    # results = spp_solver.run()
    # spp_solver.export_csv(results, output_path="data/run2_spp_solution_tropo_filter.csv")

    sp3_file = r"data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3"

    for i in [2,3,4]:
        obs_file = rf"data/run{i}.obs"
        spp_solver = SPP(obs_file, sp3_file, hatch_filter=False, trop=False)
        results = spp_solver.run()
        spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution.csv")

        spp_solver = SPP(obs_file, sp3_file, hatch_filter=False, trop=True)
        results = spp_solver.run()
        spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution_tropo.csv")

        spp_solver = SPP(obs_file, sp3_file, hatch_filter=True, trop=True)
        results = spp_solver.run()
        spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution_tropo_filter.csv")
