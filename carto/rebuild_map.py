from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from slam_core.dataio.carmen import read_carmen_log
from slam_core.common.types import Pose2
from slam_core.common.se2 import transform_points_pose
from carto.local_slam.range_to_points import ranges_to_points
from carto.config import (
    ANGLE_MIN,
    ANGLE_INC,
    RANGE_MIN,
    RANGE_MAX,
    BEAM_STRIDE,
    SUBMAP_RESOLUTION,
    L0,
    L_FREE,
    L_OCC,
    L_MIN,
    L_MAX,
)


class OccupancyGrid2D:
    def __init__(self, min_x, min_y, max_x, max_y, resolution):
        self.res = float(resolution)
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)

        self.w = int(np.ceil((self.max_x - self.min_x) / self.res)) + 1
        self.h = int(np.ceil((self.max_y - self.min_y) / self.res)) + 1

        self.L = np.full((self.h, self.w), float(L0), dtype=np.float32)

    def world_to_grid(self, x: float, y: float):
        gx = int(np.floor((x - self.min_x) / self.res))
        gy = int(np.floor((y - self.min_y) / self.res))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.w and 0 <= gy < self.h

    def update_cell(self, gx: int, gy: int, delta: float):
        if not self.in_bounds(gx, gy):
            return
        self.L[gy, gx] = np.clip(self.L[gy, gx] + delta, L_MIN, L_MAX)

    @staticmethod
    def bresenham(gx0: int, gy0: int, gx1: int, gy1: int):
        points = []
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy
        x, y = gx0, gy0

        while True:
            points.append((x, y))
            if x == gx1 and y == gy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def integrate_scan(self, pose: Pose2, scan_points_local: np.ndarray):
        pts_world = transform_points_pose(pose, scan_points_local)
        gx0, gy0 = self.world_to_grid(pose.x, pose.y)

        if not self.in_bounds(gx0, gy0):
            return

        for ex, ey in pts_world:
            gx1, gy1 = self.world_to_grid(float(ex), float(ey))
            if not self.in_bounds(gx1, gy1):
                continue

            cells = self.bresenham(gx0, gy0, gx1, gy1)

            for gx, gy in cells[:-1]:
                self.update_cell(gx, gy, L_FREE)

            self.update_cell(gx1, gy1, L_OCC)

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.L))


def load_debug_inserted(debug_path: str):
    data = np.loadtxt(debug_path, comments="#", skiprows=3)
    if data.ndim == 1:
        data = data[None, :]

    k = data[:, 0].astype(int)
    x = data[:, 2]
    y = data[:, 3]
    theta = data[:, 4]
    did_insert = data[:, 11].astype(int)
    node_count = data[:, 19].astype(int)

    mask = did_insert == 1

    inserted = []
    for kk, xx, yy, th, nn in zip(k[mask], x[mask], y[mask], theta[mask], node_count[mask]):
        inserted.append(
            {
                "scan_idx": int(kk),
                "node_id": int(nn - 1),
                "local_pose": Pose2(float(xx), float(yy), float(th)),
            }
        )
    return inserted


def load_optimized_nodes(path: str):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    out = {}
    for row in data:
        node_id = int(row[0])
        out[node_id] = Pose2(float(row[1]), float(row[2]), float(row[3]))
    return out


def collect_common_samples(log_path: str, debug_path: str, optimized_nodes_path: str):
    scans = read_carmen_log(log_path)
    inserted = load_debug_inserted(debug_path)
    opt_nodes = load_optimized_nodes(optimized_nodes_path)

    samples = []
    for item in inserted:
        node_id = item["node_id"]
        scan_idx = item["scan_idx"]

        if node_id not in opt_nodes:
            continue
        if scan_idx < 0 or scan_idx >= len(scans):
            continue

        ranges = scans[scan_idx]["ranges"]
        pts = ranges_to_points(
            ranges,
            ANGLE_MIN,
            ANGLE_INC,
            RANGE_MIN,
            RANGE_MAX,
            stride=BEAM_STRIDE,
        )

        if pts.shape[0] == 0:
            continue

        samples.append(
            {
                "node_id": node_id,
                "scan_idx": scan_idx,
                "scan_points": pts,
                "pose_local": item["local_pose"],
                "pose_optimized": opt_nodes[node_id],
            }
        )

    return samples


def compute_bounds(samples, margin: float = 1.0):
    xs = []
    ys = []

    for s in samples:
        for pose_key in ["pose_local", "pose_optimized"]:
            pose = s[pose_key]
            pts_world = transform_points_pose(pose, s["scan_points"])

            xs.append(pose.x)
            ys.append(pose.y)
            xs.extend(pts_world[:, 0].tolist())
            ys.extend(pts_world[:, 1].tolist())

    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin
    return min_x, min_y, max_x, max_y


def rebuild_maps(samples, resolution=SUBMAP_RESOLUTION):
    min_x, min_y, max_x, max_y = compute_bounds(samples)

    grid_local = OccupancyGrid2D(min_x, min_y, max_x, max_y, resolution)
    grid_opt = OccupancyGrid2D(min_x, min_y, max_x, max_y, resolution)

    traj_local = []
    traj_opt = []

    for s in samples:
        pose_local = s["pose_local"]
        pose_opt = s["pose_optimized"]
        scan_points = s["scan_points"]

        grid_local.integrate_scan(pose_local, scan_points)
        grid_opt.integrate_scan(pose_opt, scan_points)

        traj_local.append([pose_local.x, pose_local.y, pose_local.theta])
        traj_opt.append([pose_opt.x, pose_opt.y, pose_opt.theta])

    traj_local = np.asarray(traj_local, dtype=float)
    traj_opt = np.asarray(traj_opt, dtype=float)

    return grid_local, grid_opt, traj_local, traj_opt


def plot_before_after(grid_local, grid_opt, traj_local, traj_opt, out_prefix: str):
    prob_local = grid_local.probability()
    prob_opt = grid_opt.probability()

    extent = [grid_local.min_x, grid_local.max_x, grid_local.min_y, grid_local.max_y]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(prob_local, origin="lower", extent=extent, cmap="gray")
    axes[0].plot(traj_local[:, 0], traj_local[:, 1], "r-", linewidth=1.0)
    axes[0].set_title("Map Before Loop Closure")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True)

    axes[1].imshow(prob_opt, origin="lower", extent=extent, cmap="gray")
    axes[1].plot(traj_opt[:, 0], traj_opt[:, 1], "r-", linewidth=1.0)
    axes[1].set_title("Map After Loop Closure")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].axis("equal")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_before_after.png", dpi=200)
    plt.show()

    np.save(f"{out_prefix}_before_prob.npy", prob_local)
    np.save(f"{out_prefix}_after_prob.npy", prob_opt)
    np.savetxt(
        f"{out_prefix}_inserted_local_nodes.txt",
        traj_local,
        fmt="%.6f",
        header="x y theta",
    )
    np.savetxt(
        f"{out_prefix}_optimized_nodes_aligned.txt",
        traj_opt,
        fmt="%.6f",
        header="x y theta",
    )

    print("Wrote:", f"{out_prefix}_before_after.png")
    print("Wrote:", f"{out_prefix}_before_prob.npy")
    print("Wrote:", f"{out_prefix}_after_prob.npy")
    print("Wrote:", f"{out_prefix}_inserted_local_nodes.txt")
    print("Wrote:", f"{out_prefix}_optimized_nodes_aligned.txt")


def main():
    log_path = "datasets/fr079/fr079.clf"
    prefix = "carto_outputs/trajectory_scan_to_submap_loop_1000"

    debug_path = f"{prefix}_debug.txt"
    optimized_nodes_path = f"{prefix}_optimized_nodes.txt"
    out_prefix = f"{prefix}_map_rebuild"

    samples = collect_common_samples(
        log_path=log_path,
        debug_path=debug_path,
        optimized_nodes_path=optimized_nodes_path,
    )

    print("Inserted node samples used for rebuild:", len(samples))

    grid_local, grid_opt, traj_local, traj_opt = rebuild_maps(samples)
    plot_before_after(grid_local, grid_opt, traj_local, traj_opt, out_prefix)


if __name__ == "__main__":
    main()