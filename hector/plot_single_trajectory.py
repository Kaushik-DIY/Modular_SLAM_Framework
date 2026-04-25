import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import hector.config as cfg


def latest_file(pattern, exclude_debug=True):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if exclude_debug:
        files = [f for f in files if "_debug.txt" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return files[-1]


def dataset_tag():
    if cfg.DATASET_NAME == "lab_run_2":
        return f"{cfg.DATASET_NAME}_{cfg.DATASET_SCAN_VARIANT}"
    return cfg.DATASET_NAME


def load_traj(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]

    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    theta = data[:, 3]
    score = data[:, 4]
    return t, x, y, theta, score


def main():
    tag = dataset_tag()
    matcher = cfg.MATCHER_TYPE

    traj_path = latest_file(
        f"hector_outputs/trajectory_{tag}_{matcher}_*.txt",
        exclude_debug=True,
    )
    print("Using trajectory file:", traj_path)

    t, x, y, theta, score = load_traj(traj_path)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectory ({tag}, {matcher})")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_xy_{tag}_{matcher}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(score)
    plt.xlabel("scan index")
    plt.ylabel("score")
    plt.title(f"Matcher Score ({tag}, {matcher})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_score_{tag}_{matcher}.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(np.rad2deg(theta))
    plt.xlabel("scan index")
    plt.ylabel("theta [deg]")
    plt.title(f"Heading ({tag}, {matcher})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hector_outputs/trajectory_theta_{tag}_{matcher}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()