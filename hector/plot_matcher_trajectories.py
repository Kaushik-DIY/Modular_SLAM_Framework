import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import hector.config as cfg


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
    data = np.loadtxt(path, comments="#", skiprows=5)
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


def _dataset_tag() -> str:
    if cfg.DATASET_NAME == "lab_run_2":
        return f"{cfg.DATASET_NAME}_{cfg.DATASET_SCAN_VARIANT}"
    return cfg.DATASET_NAME


def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return matches[-1]


def main():
    dataset_tag = _dataset_tag()

    submap_traj_path = _latest(f"hector_outputs/trajectory_{dataset_tag}_scan_to_submap_*.txt")
    map_traj_path = _latest(f"hector_outputs/trajectory_{dataset_tag}_scan_to_map_*.txt")

    submap_dbg_path = _latest(f"hector_outputs/trajectory_{dataset_tag}_scan_to_submap_*_debug.txt")
    map_dbg_path = _latest(f"hector_outputs/trajectory_{dataset_tag}_scan_to_map_*_debug.txt")

    _, x1, y1, th1, s1 = load_traj(submap_traj_path)
    _, x2, y2, th2, s2 = load_traj(map_traj_path)

    _, _, _, _, _, _, inl1, dx1, dy1, dth1, _, _ = load_debug(submap_dbg_path)
    _, _, _, _, _, _, inl2, dx2, dy2, dth2, _, _ = load_debug(map_dbg_path)

    dth1 = np.rad2deg(wrap_angle(dth1))
    dth2 = np.rad2deg(wrap_angle(dth2))
    th1_deg = np.rad2deg(th1)
    th2_deg = np.rad2deg(th2)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="scan_to_submap")
    plt.plot(x2, y2, label="scan_to_map", color="tab:orange")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Hector Orchestration: Trajectory Comparison ({dataset_tag})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_xy_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="scan_to_submap")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Hector Orchestration: scan_to_submap Trajectory ({dataset_tag})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_xy_submap_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(x2, y2, label="scan_to_map", color="tab:orange")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Hector Orchestration: scan_to_map Trajectory ({dataset_tag})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_xy_map_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(s1, label="scan_to_submap score")
    plt.plot(s2, label="scan_to_map score")
    plt.xlabel("scan index")
    plt.ylabel("score")
    plt.title(f"Hector Orchestration: Matcher Score Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_score_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(inl1, label="scan_to_submap inliers")
    plt.plot(inl2, label="scan_to_map inliers")
    plt.xlabel("scan index")
    plt.ylabel("inliers")
    plt.title(f"Hector Orchestration: Matcher Inlier Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_inliers_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(th1_deg, label="scan_to_submap theta")
    plt.plot(th2_deg, label="scan_to_map theta")
    plt.xlabel("scan index")
    plt.ylabel("heading [deg]")
    plt.title(f"Hector Orchestration: Heading Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_theta_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dx1, label="submap dx")
    plt.plot(dx2, label="map dx")
    plt.xlabel("scan index")
    plt.ylabel("dx [m]")
    plt.title(f"Hector Orchestration: Delta X Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_dx_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dy1, label="submap dy")
    plt.plot(dy2, label="map dy")
    plt.xlabel("scan index")
    plt.ylabel("dy [m]")
    plt.title(f"Hector Orchestration: Delta Y Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_dy_{dataset_tag}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dth1, label="submap dtheta")
    plt.plot(dth2, label="map dtheta")
    plt.xlabel("scan index")
    plt.ylabel("dtheta [deg]")
    plt.title(f"Hector Orchestration: Delta Theta Comparison ({dataset_tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_compare_dtheta_{dataset_tag}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()