import matplotlib.pyplot as plt
import pandas as pd
import pymap3d as pm
import contextily as ctx
import geopandas as gpd
import numpy as np

from IMU_reader import read_ground_truth_csv, read_imu_csv, read_spp_csv, read_pos_file


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ecef_to_gdf(X, Y, Z):
    """Convert ECEF arrays to a Web-Mercator GeoDataFrame, dropping NaNs."""
    mask = ~(pd.isna(X) | pd.isna(Y) | pd.isna(Z))
    lat, lon, _ = pm.ecef2geodetic(X[mask].values, Y[mask].values, Z[mask].values)
    return gpd.GeoDataFrame(
        {"lat": lat, "lon": lon},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)


def _add_north_arrow(ax, x=0.95, y=0.12, size=0.07):
    ax.annotate(
        "",
        xy=(x, y + size), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=15),
    )
    ax.text(x, y + size + 0.02, "N",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold")


def _add_distance_axes(ax):
    """Replace raw Web-Mercator ticks with relative distances (m) from centre."""
    ax.set_axis_on()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{(v - cx):+.0f}" for v in ax.get_xticks()], fontsize=8)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{(v - cy):+.0f}" for v in ax.get_yticks()], fontsize=8)

    ax.set_xlabel("East / West  (m)", fontsize=9)
    ax.set_ylabel("North / South  (m)", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="grey")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("grey")

def _add_latlon_axes(ax):
    """Label axes with geographic coordinates (lat/lon degrees)."""
    ax.set_axis_on()

    # X-axis → longitude
    x_ticks = ax.get_xticks()
    _, lon_labels = pm.ecef2geodetic(
        *pm.enu2ecef(x_ticks, np.zeros_like(x_ticks), np.zeros_like(x_ticks),
                     0, 0, 0)
    )[:2]  # not needed — use pyproj instead:
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    x_lons, _ = transformer.transform(x_ticks, np.full_like(x_ticks, ax.get_ylim()[0]))
    _, y_lats = transformer.transform(np.full_like(y_ticks, ax.get_xlim()[0]), y_ticks)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.5f}°" for v in x_lons], fontsize=8, rotation=25, ha="right")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.5f}°" for v in y_lats], fontsize=8)

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="grey")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("grey")


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_gnss_data(spp_df, gt_df, rtk_df=None, tol=0.1):
    """
    Merge SPP, ground truth, and (optionally) RTK on GPSTime.
    Each source's columns are renamed with a suffix (_spp / _gt / _rtk)
    BEFORE merging so the suffixes are always predictable.
    """
    def _add_suffix(df, suffix):
        return df.rename(columns={c: f"{c}_{suffix}" for c in df.columns if c != "GPSTime"})

    spp = _add_suffix(spp_df.sort_values("GPSTime"), "spp")
    gt  = _add_suffix(gt_df.sort_values("GPSTime"),  "gt")

    merged = pd.merge_asof(spp, gt, on="GPSTime", direction="nearest", tolerance=tol)

    if rtk_df is not None:
        rtk = _add_suffix(rtk_df.sort_values("GPSTime"), "rtk")
        merged = pd.merge_asof(merged, rtk, on="GPSTime", direction="nearest", tolerance=tol)

    required = ["X-ECEF_spp", "Y-ECEF_spp", "Z-ECEF_spp",
                "X-ECEF_gt",  "Y-ECEF_gt",  "Z-ECEF_gt"]
    merged = merged.dropna(subset=required)
    print(f"Rows after merge: {len(merged)}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Track plot  (SPP + GT + RTK on one map)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracks(merged, title="Track Comparison"):
    """
    Plot SPP, Ground Truth, and RTK (if available) tracks on a basemap.
    Expects column names produced by merge_gnss_data (…_spp, …_gt, …_rtk).
    """
    tracks = {
        "Ground Truth": ("X-ECEF_gt",  "Y-ECEF_gt",  "Z-ECEF_gt"),
        "SPP":          ("X-ECEF_spp", "Y-ECEF_spp", "Z-ECEF_spp"),
    }
    # Add RTK only if the columns are present
    if "X-ECEF_rtk" in merged.columns:
        tracks["RTK"] = ("X-ECEF_rtk", "Y-ECEF_rtk", "Z-ECEF_rtk")

    colors = {"Ground Truth": "black", "SPP": "tab:blue", "RTK": "tab:orange"}
    styles = {"Ground Truth": dict(linewidth=2.5, zorder=3),
              "SPP":          dict(linewidth=1.5, zorder=2),
              "RTK":          dict(linewidth=1.5, zorder=2)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for name, (xc, yc, zc) in tracks.items():
        gdf = _ecef_to_gdf(merged[xc], merged[yc], merged[zc])
        ax.plot(gdf.geometry.x, gdf.geometry.y,
                label=name, color=colors[name], **styles[name])

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    _add_distance_axes(ax)
    _add_north_arrow(ax)
    ax.legend(loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_gt_tracks(gt_dfs, labels=None, title="Ground Truth Track Comparison"):
    """
    Plot multiple ground truth tracks on the same basemap.

    Parameters
    ----------
    gt_dfs  : list of DataFrames, each with columns X-ECEF, Y-ECEF, Z-ECEF
    labels  : list of strings, one per track (defaults to "Run 1", "Run 2", …)
    title   : plot title
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(gt_dfs))]

    colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(11, 7))

    for df, label, color in zip(gt_dfs, labels, colors):
        gdf = _ecef_to_gdf(df["X-ECEF"], df["Y-ECEF"], df["Z-ECEF"])
        ax.plot(
            gdf.geometry.x, gdf.geometry.y,
            label=label, color=color, linewidth=2, zorder=3,
        )
        # Mark start and end of each track
        ax.scatter(gdf.geometry.x.iloc[0],  gdf.geometry.y.iloc[0],
                   color=color, marker="o", s=60, zorder=4)
        ax.scatter(gdf.geometry.x.iloc[-1], gdf.geometry.y.iloc[-1],
                   color=color, marker="s", s=60, zorder=4)

    # Expand view by 20 m on each side
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 40, x1 + 20)
    ax.set_ylim(y0 - 40, y1 + 40)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    _add_latlon_axes(ax)
    _add_north_arrow(ax)
    ax.legend(loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_static_ins_tracks(gt_dfs, labels=None, title="Static INS Position – Before and After Bias Correction"):
    """
    Plot 10 seconds of static INS-derived position tracks on a basemap.

    Parameters
    ----------
    gt_dfs  : list of DataFrames with columns: time_s, ECEF_X_m, ECEF_Y_m, ECEF_Z_m
    labels  : list of strings, one per track
    title   : plot title
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(gt_dfs))]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(11, 7))

    for df, label, color in zip(gt_dfs, labels, colors):
        # Trim to first 10 seconds
        t0 = df["time_s"].iloc[0]
        mask = df["time_s"] <= t0 + 10
        df_10s = df[mask]

        gdf = _ecef_to_gdf(df_10s["ECEF_X_m"], df_10s["ECEF_Y_m"], df_10s["ECEF_Z_m"])
        ax.plot(
            gdf.geometry.x, gdf.geometry.y,
            label=label, color=color, linewidth=2, zorder=3,
        )
        # Mark start and end
        ax.scatter(gdf.geometry.x.iloc[0],  gdf.geometry.y.iloc[0],
                   color=color, marker="o", s=60, zorder=4, label=f"{label} – start")
        ax.scatter(gdf.geometry.x.iloc[-1], gdf.geometry.y.iloc[-1],
                   color=color, marker="s", s=60, zorder=4, label=f"{label} – end")

    # Expand view by 20 m on each side
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 20, x1 + 20)
    ax.set_ylim(y0 - 20, y1 + 20)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    _add_latlon_axes(ax)
    _add_north_arrow(ax)
    ax.legend(loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Position error
# ─────────────────────────────────────────────────────────────────────────────

def plot_position_errors(merged, t, run_name="Run", compare_rtk=False):
    """
    3-panel plot:
      1. Absolute 3-D position error (SPP vs GT, optionally RTK vs GT)
      2. SPP position std (σX, σY, σZ)
      3. PDOP + number of satellites
    """
    # ── Compute errors ───────────────────────────────────────────────────────
    merged = merged.copy()

    for axis in ("X", "Y", "Z"):
        merged[f"d{axis}_spp"] = merged[f"{axis}-ECEF_spp"] - merged[f"{axis}-ECEF_gt"]
    merged["err_3D_spp"] = np.sqrt(
        merged["dX_spp"]**2 + merged["dY_spp"]**2 + merged["dZ_spp"]**2
    )

    has_rtk_vel = compare_rtk and all(
        f"{ax}-ECEF_rtk" in merged.columns for ax in ("X", "Y", "Z")
    )
    if has_rtk_vel:
        for axis in ("X", "Y", "Z"):
            merged[f"d{axis}_rtk"] = merged[f"{axis}-ECEF_rtk"] - merged[f"{axis}-ECEF_gt"]
        merged["err_3D_rtk"] = np.sqrt(
            merged["dX_rtk"]**2 + merged["dY_rtk"]**2 + merged["dZ_rtk"]**2
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Panel 1 – 3-D error
    ax = axes[0]
    ax.plot(t, merged["err_3D_spp"], label="SPP", linewidth=1, color="tab:blue")
    # if has_rtk_vel:
    #     ax.plot(t, merged["err_3D_rtk"], label="RTK", linewidth=1, color="tab:orange")
    ax.set_ylabel("3D Error [m]")
    ax.set_title(f"Absolute Position Error – {run_name}")
    ax.legend(); ax.grid(True)

    # Panel 2 – position std
    ax = axes[1]
    std_cols = {"std_X_spp": "σX", "std_Y_spp": "σY", "std_Z_spp": "σZ"}
    for col, lbl in std_cols.items():
        if col in merged.columns:
            ax.plot(t, merged[col], label=lbl)
    ax.set_ylabel("Std [m]")
    ax.set_title("Estimated Position Uncertainty (SPP)")
    ax.legend(); ax.grid(True)

    # Panel 3 – PDOP + nsats
    ax = axes[2]
    if "PDOP_spp" in merged.columns:
        ax.plot(t, merged["PDOP_spp"], label="PDOP", color="tab:purple")
    if "nsats_spp" in merged.columns:
        ax.step(t, merged["nsats_spp"], label="nsats", where="post", color="tab:green")
    ax.set_ylabel("PDOP / nsats")
    ax.set_xlabel("GPS Time [s]")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n=== Position Error Summary – {run_name} ===")
    print(f"  SPP  RMS 3D : {np.sqrt((merged['err_3D_spp']**2).mean()):.3f} m")
    if has_rtk_vel:
        print(f"  RTK  RMS 3D : {np.sqrt((merged['err_3D_rtk']**2).mean()):.3f} m")


# ─────────────────────────────────────────────────────────────────────────────
# Velocity error
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocity_errors(merged, t, run_name="Run", compare_rtk=False):
    """
    3-panel plot:
      1. Absolute 3-D velocity error (SPP Doppler vs GT)
      2. SPP 3-D velocity std envelope
      3. Per-axis velocity std (σVX, σVY, σVZ)

    RTK comparison is skipped silently when RTK velocity columns are absent
    (e.g. RTKLIB .pos files typically don't carry Doppler velocity).
    """
    merged = merged.copy()

    # ── SPP vs GT ────────────────────────────────────────────────────────────
    for axis in ("X", "Y", "Z"):
        merged[f"dV{axis}_spp"] = merged[f"V{axis}-ECEF_spp"] - merged[f"V{axis}-ECEF_gt"]
    merged["err_vel_3D_spp"] = np.sqrt(
        merged["dVX_spp"]**2 + merged["dVY_spp"]**2 + merged["dVZ_spp"]**2
    )

    # ── RTK vs GT (only if velocity columns exist) ───────────────────────────
    has_rtk_vel = compare_rtk and all(
        f"V{ax}-ECEF_rtk" in merged.columns for ax in ("X", "Y", "Z")
    )
    if compare_rtk and not has_rtk_vel:
        print("  [info] RTK velocity columns not found – skipping RTK velocity comparison.")

    if has_rtk_vel:
        for axis in ("X", "Y", "Z"):
            merged[f"dV{axis}_rtk"] = merged[f"V{axis}-ECEF_rtk"] - merged[f"V{axis}-ECEF_gt"]
        merged["err_vel_3D_rtk"] = np.sqrt(
            merged["dVX_rtk"]**2 + merged["dVY_rtk"]**2 + merged["dVZ_rtk"]**2
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # Panel 1 – 3-D velocity error
    ax = axes[0]
    ax.plot(t, merged["err_vel_3D_spp"], label="SPP Doppler",
            linewidth=1, color="tab:blue")
    if has_rtk_vel:
        ax.plot(t, merged["err_vel_3D_rtk"], label="RTK",
                linewidth=1, color="tab:orange")
    ax.set_ylabel("3D Velocity Error [m/s]")
    ax.set_title(f"Velocity Error – {run_name}")
    ax.legend(); ax.grid(True)

    # Panel 3 – per-axis std
    ax = axes[1]
    std_vel_cols = {"std_VX_spp": "σVX", "std_VY_spp": "σVY", "std_VZ_spp": "σVZ"}
    for col, lbl in std_vel_cols.items():
        if col in merged.columns:
            ax.plot(t, merged[col], label=lbl)
    ax.set_xlabel("GPS Time [s]")
    ax.set_ylabel("Std [m/s]")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n=== Velocity Error Summary – {run_name} ===")
    print(f"  SPP  RMS 3D : {np.sqrt((merged['err_vel_3D_spp']**2).mean()):.4f} m/s")
    if has_rtk_vel:
        print(f"  RTK  RMS 3D : {np.sqrt((merged['err_vel_3D_rtk']**2).mean()):.4f} m/s")

def plot_navigation_performance(
    merged,
    t,
    run_name="Run",
    compare_rtk=True,
):
    """
    5-panel GNSS performance plot.

    Panels:
      1. PDOP + number of satellites
      2. 3D position error (SPP + RTK)
      3. Position sigma components (XYZ)
      4. 3D velocity error (SPP + RTK)
      5. Velocity sigma components (VXYZ)

    Uses LaTeX formatting for labels/titles.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    merged = merged.copy()

    # ---------------------------------------------------------------------
    # Detect rover motion from GT velocity
    # ---------------------------------------------------------------------
    vel_gt = np.sqrt(
        merged["VX-ECEF_gt"]**2 +
        merged["VY-ECEF_gt"]**2 +
        merged["VZ-ECEF_gt"]**2
    )

    motion_threshold = 0.1  # [m/s]
    moving = vel_gt > motion_threshold

    # Find contiguous moving intervals
    motion_intervals = []

    start_idx = None

    for i, is_moving in enumerate(moving):

        if is_moving and start_idx is None:
            start_idx = i

        elif not is_moving and start_idx is not None:
            motion_intervals.append((t.iloc[start_idx], t.iloc[i - 1]))
            start_idx = None

    # Handle motion until end
    if start_idx is not None:
        motion_intervals.append((t.iloc[start_idx], t.iloc[-1]))

    # ---------------------------------------------------------------------
    # Compute position errors
    # ---------------------------------------------------------------------
    for axis in ("X", "Y", "Z"):
        merged[f"d{axis}_spp"] = (
            merged[f"{axis}-ECEF_spp"] - merged[f"{axis}-ECEF_gt"]
        )

    merged["err_3D_spp"] = np.sqrt(
        merged["dX_spp"]**2 +
        merged["dY_spp"]**2 +
        merged["dZ_spp"]**2
    )

    has_rtk_pos = compare_rtk and all(
        f"{ax}-ECEF_rtk" in merged.columns for ax in ("X", "Y", "Z")
    )

    if has_rtk_pos:
        for axis in ("X", "Y", "Z"):
            merged[f"d{axis}_rtk"] = (
                merged[f"{axis}-ECEF_rtk"] - merged[f"{axis}-ECEF_gt"]
            )

        merged["err_3D_rtk"] = np.sqrt(
            merged["dX_rtk"]**2 +
            merged["dY_rtk"]**2 +
            merged["dZ_rtk"]**2
        )

    # ---------------------------------------------------------------------
    # Compute velocity errors
    # ---------------------------------------------------------------------
    for axis in ("X", "Y", "Z"):
        merged[f"dV{axis}_spp"] = (
            merged[f"V{axis}-ECEF_spp"] - merged[f"V{axis}-ECEF_gt"]
        )

    merged["err_vel_3D_spp"] = np.sqrt(
        merged["dVX_spp"]**2 +
        merged["dVY_spp"]**2 +
        merged["dVZ_spp"]**2
    )

    has_rtk_vel = compare_rtk and all(
        f"V{ax}-ECEF_rtk" in merged.columns for ax in ("X", "Y", "Z")
    )

    if has_rtk_vel:
        for axis in ("X", "Y", "Z"):
            merged[f"dV{axis}_rtk"] = (
                merged[f"V{axis}-ECEF_rtk"] - merged[f"V{axis}-ECEF_gt"]
            )

        merged["err_vel_3D_rtk"] = np.sqrt(
            merged["dVX_rtk"]**2 +
            merged["dVY_rtk"]**2 +
            merged["dVZ_rtk"]**2
        )

    # ---------------------------------------------------------------------
    # RMS statistics
    # ---------------------------------------------------------------------
    rms_pos_spp = np.sqrt(np.nanmean(merged["err_3D_spp"]**2))
    rms_vel_spp = np.sqrt(np.nanmean(merged["err_vel_3D_spp"]**2))

    if has_rtk_pos:
        rms_pos_rtk = np.sqrt(np.nanmean(merged["err_3D_rtk"]**2))

    if has_rtk_vel:
        rms_vel_rtk = np.sqrt(np.nanmean(merged["err_vel_3D_rtk"]**2))

    # ---------------------------------------------------------------------
    # Matplotlib style
    # ---------------------------------------------------------------------
    plt.rcParams.update({
        "text.usetex": False,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 18),
        sharex=True,
        constrained_layout=True
    )

    # =====================================================================
    # Panel 1 — PDOP + nsats
    # =====================================================================
    ax = axes[0]

    if "PDOP_spp" in merged.columns:
        ax.plot(
            t,
            merged["PDOP_spp"],
            label=r"$\mathrm{PDOP}$",
            color="tab:purple",
            linewidth=1.5
        )

    if "nsats_spp" in merged.columns:
        ax.step(
            t,
            merged["nsats_spp"],
            where="post",
            label=r"$N_{\mathrm{sats}}$",
            color="tab:green"
        )

    ax.set_ylabel(r"$\mathrm{PDOP} \;/\; N_{\mathrm{sats}}$")
    ax.set_title(r"Satellite Geometry")
    ax.grid(True)
    ax.legend()
    # Shade rover motion intervals
    for t0, t1 in motion_intervals:
        ax.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            zorder=0
        )

    # =====================================================================
    # Panel 2 — 3D position error
    # =====================================================================
    ax = axes[1]

    ax.plot(
        t,
        merged["err_3D_spp"],
        label=rf"$\mathrm{{SPP}} \;\; (\mathrm{{RMS}}={rms_pos_spp:.2f}\,\mathrm{{m}})$",
        linewidth=1.2
    )

    if has_rtk_pos:
        ax.plot(
            t,
            merged["err_3D_rtk"],
            label=rf"$\mathrm{{RTK}} \;\; (\mathrm{{RMS}}={rms_pos_rtk:.2f}\,\mathrm{{m}})$",
            linewidth=1.2
        )

    ax.set_ylabel(r"$||\Delta \mathbf{r}||_{2}\;[\mathrm{m}]$")
    ax.set_title(r"3D Position Error")
    ax.grid(True)
    ax.legend()
    # Shade rover motion intervals
    for t0, t1 in motion_intervals:
        ax.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            zorder=0
        )

    # =====================================================================
    # Panel 3 — Position sigma components
    # =====================================================================
    ax = axes[2]

    pos_std_cols = {
        "std_X_spp": r"$\sigma_X^{\mathrm{SPP}}$",
        "std_Y_spp": r"$\sigma_Y^{\mathrm{SPP}}$",
        "std_Z_spp": r"$\sigma_Z^{\mathrm{SPP}}$",
    }

    for col, lbl in pos_std_cols.items():
        if col in merged.columns:
            ax.plot(t, merged[col], label=lbl+" "+f"({np.mean(merged[col]):.2f}) m")

    if has_rtk_pos:
        rtk_std_cols = {
            "std_X_rtk": r"$\sigma_X^{\mathrm{RTK}}$",
            "std_Y_rtk": r"$\sigma_Y^{\mathrm{RTK}}$",
            "std_Z_rtk": r"$\sigma_Z^{\mathrm{RTK}}$",
        }

        for col, lbl in rtk_std_cols.items():
            if col in merged.columns:
                ax.plot(t, merged[col], "--", label=lbl+" "+f"({np.mean(merged[col]):.2f})m")

    ax.set_ylabel(r"$\sigma_r\;[\mathrm{m}]$")
    ax.set_title(r"Position Uncertainty Components")
    ax.grid(True)
    # Shade rover motion intervals
    for t0, t1 in motion_intervals:
        ax.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            zorder=0
        )
        ax.legend(ncol=2)

    # =====================================================================
    # Panel 4 — 3D velocity error
    # =====================================================================
    ax = axes[3]

    ax.plot(
        t,
        merged["err_vel_3D_spp"],
        label=rf"$\mathrm{{SPP}} \;\; (\mathrm{{RMS}}={rms_vel_spp:.3f}\,\mathrm{{m/s}})$",
        linewidth=1.2
    )

    if has_rtk_vel:
        ax.plot(
            t,
            merged["err_vel_3D_rtk"],
            label=rf"$\mathrm{{RTK}} \;\; (\mathrm{{RMS}}={rms_vel_rtk:.3f}\,\mathrm{{m/s}})$",
            linewidth=1.2
        )

    ax.set_ylabel(r"$||\Delta \mathbf{v}||_{2}\;[\mathrm{m/s}]$")
    ax.set_title(r"3D Velocity Error")
    ax.grid(True)
    ax.legend()
    # Shade rover motion intervals
    for t0, t1 in motion_intervals:
        ax.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            zorder=0
        )

    # =====================================================================
    # Panel 5 — Velocity sigma components
    # =====================================================================
    ax = axes[4]

    vel_std_cols = {
        "std_VX_spp": r"$\sigma_{V_X}^{\mathrm{SPP}}$",
        "std_VY_spp": r"$\sigma_{V_Y}^{\mathrm{SPP}}$",
        "std_VZ_spp": r"$\sigma_{V_Z}^{\mathrm{SPP}}$",
    }

    for col, lbl in vel_std_cols.items():
        if col in merged.columns:
            ax.plot(t, merged[col], label=lbl+" "+f"({np.mean(merged[col]):.2f}) m/s")

    if has_rtk_vel:
        vel_std_cols_rtk = {
            "std_VX_rtk": r"$\sigma_{V_X}^{\mathrm{RTK}}$",
            "std_VY_rtk": r"$\sigma_{V_Y}^{\mathrm{RTK}}$",
            "std_VZ_rtk": r"$\sigma_{V_Z}^{\mathrm{RTK}}$",
        }

        for col, lbl in vel_std_cols_rtk.items():
            if col in merged.columns:
                ax.plot(t, merged[col], "--", label=lbl+" "+f"({np.mean(merged[col]):.2f}) m/s")

    ax.set_ylabel(r"$\sigma_v\;[\mathrm{m/s}]$")
    ax.set_xlabel(r"$t_{\mathrm{GPS}}\;[\mathrm{s}]$")
    ax.set_title("Velocity Uncertainty Components")
    ax.grid(True)
    ax.legend(ncol=2)
    # Shade rover motion intervals
    for t0, t1 in motion_intervals:
        ax.axvspan(
            t0,
            t1,
            color="gray",
            alpha=0.15,
            zorder=0
        )

    # ---------------------------------------------------------------------
    # Figure title
    # ---------------------------------------------------------------------
    fig.suptitle(
        f"{run_name}",
        fontsize=18
    )

    plt.show()

    # ---------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------
    print("\n===================================================")
    print(f" Navigation Performance Summary — {run_name}")
    print("===================================================")

    print(f"SPP Position RMS : {rms_pos_spp:.3f} m")

    if has_rtk_pos:
        print(f"RTK Position RMS : {rms_pos_rtk:.3f} m")

    print(f"SPP Velocity RMS : {rms_vel_spp:.4f} m/s")

    if has_rtk_vel:
        print(f"RTK Velocity RMS : {rms_vel_rtk:.4f} m/s")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spp_df = read_spp_csv(r"data/run2_spp_solution.csv")
    gt_df  = read_ground_truth_csv(r"data/run2_groundtruth.txt", gps_week=2415)
    rtk_df = read_spp_csv(r"data/run2_RTK.csv")

    merged = merge_gnss_data(spp_df, gt_df, rtk_df)
    t = merged["GPSTime"]

    # plot_tracks(merged, title="Run 2 – SPP / Ground Truth / RTK")
    # # plot_position_errors(merged, t, run_name="Run 2", compare_rtk=True)
    # # plot_velocity_errors(merged, t, run_name="Run 2", compare_rtk=True)
    # plot_navigation_performance(merged,t,run_name="Run 2", compare_rtk=True)



    # # Plot all ground truth tracks: 
    # run2_gt = read_ground_truth_csv(r"data/run2_groundtruth.txt", gps_week=2415)
    # run3_gt = read_ground_truth_csv(r"data/run3_groundtruth.txt", gps_week=2415)
    # run4_gt = read_ground_truth_csv(r"data/run4_groundtruth.txt", gps_week=2415)

    # plot_multiple_gt_tracks(
    # gt_dfs=[run2_gt, run3_gt, run4_gt],
    # labels=["Run 2", "Run 3", "Run 4"],
    # title="Ground Truth Track Comparison – Runs 2/3/4",
    # )

    # Plot static with and without corrected bias
    static_raw = pd.read_csv(r"data\static_trajectory.csv")
    static_corrected = pd.read_csv(r"data\static_trajectory_corrected.csv")
    static_raw = static_raw.iloc[:1000]
    plot_static_ins_tracks(
        gt_dfs=[static_raw, static_corrected],
        labels=["Raw", "Bias Corrected"],
        title="Static INS Position – Before and After Bias Correction (10 s)",
    )