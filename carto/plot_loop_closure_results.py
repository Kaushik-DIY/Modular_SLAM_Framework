import numpy as np
import matplotlib.pyplot as plt


def load_local_traj(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def load_debug(path):
    data = np.loadtxt(path, comments="#", skiprows=3)
    if data.ndim == 1:
        data = data[None, :]
    return {
        "k": data[:, 0],
        "t": data[:, 1],
        "x": data[:, 2],
        "y": data[:, 3],
        "theta": data[:, 4],
        "score": data[:, 5],
        "inliers": data[:, 6],
        "dx": data[:, 7],
        "dy": data[:, 8],
        "dtheta": data[:, 9],
        "do_insert": data[:, 10],
        "did_insert": data[:, 11],
        "constraints": data[:, 12],
        "nodes": data[:, 13],
        "submaps": data[:, 14],
    }


def load_optimized_nodes(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def main():
    prefix = "carto_outputs/trajectory_scan_to_submap_loop_2000"

    local_traj_path = f"{prefix}.txt"
    debug_path = f"{prefix}_debug.txt"
    opt_nodes_path = f"{prefix}_optimized_nodes.txt"

    t, x, y, th, score = load_local_traj(local_traj_path)
    dbg = load_debug(debug_path)
    node_id_opt, x_opt, y_opt, th_opt = load_optimized_nodes(opt_nodes_path)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="local trajectory")
    plt.plot(x_opt, y_opt, label="optimized trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Loop-Closure SLAM: Local vs Optimized Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_compare_xy.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(dbg["constraints"], label="constraints")
    plt.plot(dbg["nodes"], label="nodes")
    plt.plot(dbg["submaps"], label="submaps")
    plt.xlabel("scan index")
    plt.ylabel("count")
    plt.title("Graph Growth During Loop-Closure SLAM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_graph_growth.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(score, label="matcher score")
    plt.xlabel("scan index")
    plt.ylabel("score")
    plt.title("Loop-Closure Run: Matcher Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_score.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()