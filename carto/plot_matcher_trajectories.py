import numpy as np
import matplotlib.pyplot as plt


def load_traj(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    th = data[:, 3]
    score = data[:, 4]
    return t, x, y, th, score


def load_debug(path):
    data = np.loadtxt(path, comments="#", skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    k = data[:, 0]
    t = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]
    th = data[:, 4]
    score = data[:, 5]
    inliers = data[:, 6]
    dx = data[:, 7]
    dy = data[:, 8]
    dth = data[:, 9]
    do_insert = data[:, 10]
    did_insert = data[:, 11]
    return k, t, x, y, th, score, inliers, dx, dy, dth, do_insert, did_insert


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def main():
    submap_traj_path = "carto_outputs/trajectory_scan_to_submap_4934.txt"
    map_traj_path = "carto_outputs/trajectory_scan_to_map_4934.txt"

    submap_dbg_path = "carto_outputs/trajectory_scan_to_submap_4934_debug.txt"
    map_dbg_path = "carto_outputs/trajectory_scan_to_map_4934_debug.txt"

    t1, x1, y1, th1, s1 = load_traj(submap_traj_path)
    t2, x2, y2, th2, s2 = load_traj(map_traj_path)

    _, _, _, _, _, _, inl1, dx1, dy1, dth1, _, _ = load_debug(submap_dbg_path)
    _, _, _, _, _, _, inl2, dx2, dy2, dth2, _, _ = load_debug(map_dbg_path)

    dth1 = np.rad2deg(wrap_angle(dth1))
    dth2 = np.rad2deg(wrap_angle(dth2))

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="scan_to_submap")
    plt.plot(x2, y2, label="scan_to_map", color="tab:orange")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Carto Orchestration: Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_xy.png", dpi=200)
    plt.show()

    # ------------------------------------------------
    # XY trajectory
    # ------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="scan_to_submap")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Carto Orchestration: scan_to_submap Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_xy_submap.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x2, y2, label="scan_to_map", color="tab:orange")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Carto Orchestration: scan_to_map Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_xy_map.png", dpi=200)
    plt.show()

    # ------------------------------------------------
    # Score comparison
    # ------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(s1, label="scan_to_submap score")
    plt.plot(s2, label="scan_to_map score")
    plt.xlabel("scan index")
    plt.ylabel("score")
    plt.title("Carto Orchestration: Matcher Score Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_score.png", dpi=200)
    plt.show()

    # ------------------------------------------------
    # Inlier comparison
    # ------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(inl1, label="scan_to_submap inliers")
    plt.plot(inl2, label="scan_to_map inliers")
    plt.xlabel("scan index")
    plt.ylabel("inliers")
    plt.title("Carto Orchestration: Matcher Inlier Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_inliers.png", dpi=200)
    plt.show()

    # ------------------------------------------------
    # Delta comparison
    # ------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(dx1, label="submap dx")
    plt.plot(dx2, label="map dx")
    plt.xlabel("scan index")
    plt.ylabel("dx [m]")
    plt.title("Carto Orchestration: Delta X Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_dx.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dy1, label="submap dy")
    plt.plot(dy2, label="map dy")
    plt.xlabel("scan index")
    plt.ylabel("dy [m]")
    plt.title("Carto Orchestration: Delta Y Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_dy.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dth1, label="submap dtheta")
    plt.plot(dth2, label="map dtheta")
    plt.xlabel("scan index")
    plt.ylabel("dtheta [deg]")
    plt.title("Carto Orchestration: Delta Theta Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_dtheta.png", dpi=200)
    plt.show()

    # ------------------------------------------------
    # Heading comparison
    # ------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(np.rad2deg(th1), label="scan_to_submap theta")
    plt.plot(np.rad2deg(th2), label="scan_to_map theta")
    plt.xlabel("scan index")
    plt.ylabel("heading [deg]")
    plt.title("Carto Orchestration: Heading Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("carto_outputs/trajectory_compare_theta.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()