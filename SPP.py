import numpy as np
import pandas as pd
from tqdm import tqdm
import pymap3d as pm

import rinexReader as rr
import SatOrbits as so


class SPP:
    def __init__(self, obs_file_path, sp3_file_path,
                 consts=["G", "E", "C"], sig_types=["C1C", "L1C","D1C"],
                 hatch_filter=False, trop=True, elevation_mask_deg=15.0):
        self.clight = 299792458
        self.obs_file_path = obs_file_path
        self.sp3_file_path = sp3_file_path
        self.consts = consts
        self.sig_types = sig_types
        self.hatch_filter = hatch_filter
        self.trop = trop
        self.elevation_mask_deg = elevation_mask_deg

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

        # Avoid huge corrections at low elevations by bounding elevation angle.
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

    def _apply_elevation_mask(self, obs, satpos_xyz, receiver_xyz, cdts=None):
        """
        Filter satellites by elevation angle relative to the receiver position.
        """
        el = self._elevation_angle(satpos_xyz, receiver_xyz)
        min_el = np.radians(self.elevation_mask_deg)
        mask = el >= min_el

        if cdts is not None:
            cdts = cdts.loc[mask]
        return obs.loc[mask], satpos_xyz.loc[mask], cdts

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
  
    def _get_sat_velocities(self, epoch, sv_list, dt=1.0):
            """
            Estimate satellite velocities by central finite difference on SP3 positions.
            dt : half-step in seconds (1 s gives good accuracy for 5-min SP3)
            Returns DataFrame (N, 3) of [VX, VY, VZ] in m/s.
            """
            # Dummy tau (small — satellite moves little in 1 s relative to SP3 interval)
            tau_dummy = pd.Series(np.full(len(sv_list), 0.07), index=sv_list)

            # Epoch ± dt
            epoch_fwd = epoch + pd.Timedelta(seconds=dt)
            epoch_bwd = epoch - pd.Timedelta(seconds=dt)

            pos_fwd = self.orbits.getSvPos(epoch_fwd, tau_dummy).iloc[:, :3]
            pos_bwd = self.orbits.getSvPos(epoch_bwd, tau_dummy).iloc[:, :3]

            vel = (pos_fwd.values - pos_bwd.values) / (2 * dt)
            return pd.DataFrame(vel, index=sv_list, columns=["VX", "VY", "VZ"])

    def _doppler_velocity(self, doppler_obs, satpos_xyz, sat_vel, receiver_xyz):
        """
        Solve for receiver velocity from Doppler observations.
        RINEX convention: D > 0 for approaching satellite (ρ̇ < 0)
        → λD = ê·v_rec - ê·v_sat + c·δṫ_rec
        → L  = λD + ê·v_sat,   A = [ê | 1]
        """
        lambda_L1 = self.clight / 1575.42e6

        # ── Line-of-sight unit vectors (receiver → satellite) ────────────────────
        los = satpos_xyz.values - receiver_xyz[:3]            # (N, 3)
        rng = np.linalg.norm(los, axis=1, keepdims=True)
        e   = los / rng                                       # (N, 3)

        # ── Design matrix: [ê | 1] ──────────────────────────────────────────────
        A = np.hstack([e, np.ones((len(e), 1))])              # (N, 4)

        # ── Observation vector ───────────────────────────────────────────────────
        # L_i = λ·D_i + ê_i · v_sat_i    ← note the + sign
        range_rate_sat = np.sum(e * sat_vel.values, axis=1)   # ê · v_sat  (N,)
        L = lambda_L1 * doppler_obs.values + range_rate_sat   # ← WAS MINUS, NOW PLUS

        # ── Single-step LS (system is linear) ───────────────────────────────────
        N = A.T @ A
        c = A.T @ L
        h = np.linalg.solve(N, c)                             # [VX, VY, VZ, cdtdot]

        # ── Uncertainty ──────────────────────────────────────────────────────────
        v   = L - A @ h
        s0  = (v @ v) / max(len(L) - 4, 1)
        Qv  = np.linalg.inv(N)

        velocity = pd.Series(h, index=["VX", "VY", "VZ", "cdtdot"], name="Velocity")
        return velocity, Qv, s0
    
    def run(self, x0=None):
        """Run SPP + Doppler velocity for all epochs."""
        lambda_L1 = self.clight / 1575.42e6

        if x0 is None:
            x0 = np.zeros(4)
        x0 = np.array(x0, dtype=float)

        solutions  = {}
        hatch_state = {}

        for epoch in tqdm(self.rinex.timelist):
            obs = self.rinex.get_epoch_data(
                epoch, oTypes=["C1C", "L1C", "D1C"]   # ← add D1C
            ).dropna(subset=["C1C"])

            if len(obs) < 4:
                continue

            # ── Pseudorange (existing logic) ─────────────────────────────────
            if self.hatch_filter:
                obs_used, hatch_state = self._hatch_filter(
                    obs, hatch_state, lambda_L1
                )
            else:
                obs_used = obs["C1C"]

            tau       = obs_used / self.clight
            satpos    = self.orbits.getSvPos(epoch, tau)
            cdts      = satpos.iloc[:, 3] * self.clight
            satpos_xyz = satpos.iloc[:, :3]

            if self.elevation_mask_deg is not None and np.linalg.norm(x0[:3]) > 1e4:
                obs_used, satpos_xyz, cdts = self._apply_elevation_mask(
                    obs_used, satpos_xyz, x0, cdts=cdts
                )

            if len(obs_used) < 4:
                continue

            obs_corr = obs_used + cdts

            if self.trop and np.linalg.norm(x0[:3]) > 1e4:
                tropo    = self._tropoCorr(satpos_xyz, x0)
                obs_corr = obs_corr - tropo

            x, Qx, s0 = self._spp(obs_corr, satpos_xyz, x0)

            # ── Doppler velocity ─────────────────────────────────────────────
            vel_result = None
            if "D1C" in obs.columns:
                doppler = obs.loc[satpos_xyz.index, "D1C"].dropna()
                common_sv = satpos_xyz.index.intersection(doppler.index)

                if len(common_sv) >= 4:
                    sat_vel = self._get_sat_velocities(epoch, common_sv)

                    vel, Qv, s0v = self._doppler_velocity(
                        doppler.loc[common_sv],
                        satpos_xyz.loc[common_sv],
                        sat_vel,
                        x.values
                    )
                    vel_result = {
                        "velocity":            vel,
                        "velocity_covariance": Qv,
                        "velocity_variance":   s0v,
                    }

            solutions[epoch] = {
                "solution":   x,
                "covariance": Qx,
                "variance":   s0,
                "nsats":      len(obs_used),
                **(vel_result or {}),
            }

            x0 = x.values

        return solutions

    def export_csv(self, solutions, output_path="spp_results.csv"):
        pos_rows = []
        for epoch, data in solutions.items():
            x   = data["solution"]
            Qx  = data["covariance"]
            s0  = data["variance"]

            std_X = np.sqrt(s0 * Qx[0, 0])
            std_Y = np.sqrt(s0 * Qx[1, 1])
            std_Z = np.sqrt(s0 * Qx[2, 2])
            PDOP  = np.sqrt(Qx[0, 0] + Qx[1, 1] + Qx[2, 2])

            row = {
                "datetime": epoch,
                "X-ECEF":   x["X"],
                "Y-ECEF":   x["Y"],
                "Z-ECEF":   x["Z"],
                "cdt":      x["cdt"],
                "std_X":    std_X,
                "std_Y":    std_Y,
                "std_Z":    std_Z,
                "PDOP":     PDOP,
                "nsats":    data["nsats"],
            }

            # ── Doppler velocity columns (NaN if unavailable) ────────────────
            if "velocity" in data:
                vel = data["velocity"]
                Qv  = data["velocity_covariance"]
                s0v = data["velocity_variance"]
                row.update({
                    "VX-ECEF":  vel["VX"],
                    "VY-ECEF":  vel["VY"],
                    "VZ-ECEF":  vel["VZ"],
                    "cdtdot":   vel["cdtdot"],
                    "std_VX":   np.sqrt(s0v * Qv[0, 0]),
                    "std_VY":   np.sqrt(s0v * Qv[1, 1]),
                    "std_VZ":   np.sqrt(s0v * Qv[2, 2]),
                    "std_V3D":  np.sqrt(s0v * (Qv[0,0] + Qv[1,1] + Qv[2,2])),
                })
            else:
                row.update({
                    "VX-ECEF": np.nan, "VY-ECEF": np.nan, "VZ-ECEF": np.nan,
                    "cdtdot":  np.nan,
                    "std_VX":  np.nan, "std_VY":  np.nan, "std_VZ":  np.nan,
                    "std_V3D": np.nan,
                })

            pos_rows.append(row)

        pd.DataFrame(pos_rows).sort_values("datetime").to_csv(output_path, index=False)
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
        spp_solver = SPP(obs_file, sp3_file, hatch_filter=False, trop=True)
        results = spp_solver.run()
        spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution.csv")

        # spp_solver = SPP(obs_file, sp3_file, hatch_filter=False, trop=True)
        # results = spp_solver.run()
        # spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution_tropo.csv")

        # spp_solver = SPP(obs_file, sp3_file, hatch_filter=True, trop=True)
        # results = spp_solver.run()
        # spp_solver.export_csv(results, output_path=rf"data/run{i}_spp_solution_tropo_filter.csv")
