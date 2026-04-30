import numpy as np
import pandas as pd
from tqdm import tqdm

import rinexReader as rr
import SatOrbits as so


class SPP:
    def __init__(self, obs_file_path, sp3_file_path,
                 consts=["G", "E", "C"], sig_types=["C1C"]):
        self.clight = 299792458  # m/s
        self.obs_file_path = obs_file_path
        self.sp3_file_path = sp3_file_path
        self.consts = consts
        self.sig_types = sig_types

        self.rinex = rr.rinexReader(obs_file_path)
        self.orbits = so.sp3Orbits(sp3_file_path)
        self.rinex.readFile(self.consts, self.sig_types)

    def _create_kernel(self, obs, satpos, x):
        """Build design matrix A and data vector L for one NLLS iteration."""
        rng = np.linalg.norm(satpos - x[:3], axis=1)
        A_xyz = (x[:3] - satpos) / rng[:, None]
        clk = pd.Series(np.ones(len(A_xyz)), index=A_xyz.index)
        A = pd.concat([A_xyz, clk], axis=1)
        L = obs - rng[:, None] - x[3]  # x[3] is scalar clock bias
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
            h = (np.linalg.inv(N) @ c).iloc[:, 0].values
            x += h

        solution = pd.Series(x, index=["X", "Y", "Z", "cdt"], name="Solution")
        v = np.array(L).T - np.array(A) @ h
        s0 = (v @ v.T)[0][0] / (len(L) - len(h))
        Qx = np.linalg.inv(A.T @ A)
        return solution, Qx, s0

    def run(self, x0=None):
        """Run SPP for all epochs. Returns dict keyed by epoch."""
        if x0 is None:
            x0 = np.zeros(4)
        x0 = np.array(x0, dtype=float)

        solutions = {}
        # print(self.rinex.timelist)
        for epoch in tqdm(self.rinex.timelist):
            obs = self.rinex.get_epoch_data(epoch, oTypes=self.sig_types).dropna()
            if len(obs) < 4:
                continue

            tau = obs["C1C"] / self.clight
            satpos = self.orbits.getSvPos(epoch, tau)
            cdts = satpos.iloc[:, 3] * self.clight
            satpos_xyz = satpos.iloc[:, :3]
            obs_corr = obs + cdts.values[:, None]

            x, Qx, s0 = self._spp(obs_corr, satpos_xyz, x0)
            solutions[epoch] = {"solution": x, "covariance": Qx, "variance": s0}

        return solutions

    def export_csv(self, solutions, output_path="spp_results.csv"):
        """Export position solutions to CSV."""
        rows = [
            {
                "epoch": epoch,
                "X": data["solution"]["X"],
                "Y": data["solution"]["Y"],
                "Z": data["solution"]["Z"],
                "cdt": data["solution"]["cdt"],
                "variance": data["variance"],
            }
            for epoch, data in solutions.items()
        ]
        pd.DataFrame(rows).sort_values("epoch").to_csv(output_path, index=False)
        print(f"SPP results exported to {output_path}")


if __name__ == "__main__":
    obs_file = r"data/run2.obs"
    sp3_file = r"data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3"

    spp_solver = SPP(obs_file, sp3_file)
    results = spp_solver.run()

    print(results[list(results.keys())[0]]["solution"])
    spp_solver.export_csv(results, output_path="data/run2_spp_solution.csv")