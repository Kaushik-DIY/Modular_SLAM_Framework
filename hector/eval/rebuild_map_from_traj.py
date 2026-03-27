# eval/rebuild_map_from_traj.py
from __future__ import annotations

import argparse
import os
import numpy as np

import config as cfg
from slam_core.dataio.carmen import read_carmen_log
from slam.lidar import scan_to_points
from slam.pyramid import MapPyramid
from slam.se2 import transform_points, wrap_angle
from viz.plot_final import plot_map_and_traj


def load_traj(path: str) -> np.ndarray:
    traj = np.loadtxt(path, comments="#")
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    traj = traj[:, :3].astype(float)
    traj[:, 2] = np.array([wrap_angle(t) for t in traj[:, 2]])
    return traj


def rebuild_map(
    log_path: str,
    traj: np.ndarray,
    angle_min: float,
    angle_inc: float,
    map_every: int = 1,
    beam_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = read_carmen_log(log_path)
    if not data:
        raise RuntimeError("No flaser entries found in log file")

    # Ensure same length (traj is usually len(scans))
    N = min(len(traj), len(data))
    data = data[:N]
    traj = traj[:N]

    stride = cfg.BEAM_STRIDE if beam_stride is None else beam_stride

    pyr = MapPyramid.create(
        base_res=cfg.MAP_RESOLUTION,
        size_m=cfg.MAP_SIZE_METERS,
        num_levels=cfg.PYRAMID_LEVELS,
        l0=cfg.L0,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )

    for k, e in enumerate(data):
        if map_every > 1 and (k % map_every) != 0:
            continue

        pts = scan_to_points(
            e["ranges"],
            angle_min=angle_min,
            angle_inc=angle_inc,
            rmin=cfg.LIDAR_MIN_RANGE,
            rmax=cfg.LIDAR_MAX_RANGE,
            stride=stride,
        )

        pose = traj[k].copy()
        pose[2] = wrap_angle(pose[2])

        pts_world = transform_points(pose, pts)

        for grid in pyr.levels:
            grid.integrate_scan_simple(
                pose=pose,
                pts_world=pts_world,
                l_free=cfg.L_FREE,
                l_occ=cfg.L_OCC,
                ray_steps=cfg.RAY_STEPS,
            )

    prob = pyr.finest().prob().astype(np.float32)
    return prob, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Freiburg .clf log")
    ap.add_argument("--traj", required=True, help="trajectory file (x y theta)")
    ap.add_argument("--out_dir", default="outputs/rebuilt")
    ap.add_argument("--tag", default="rebuilt", help="tag used in output filenames")

    ap.add_argument("--angle_min", type=float, default=-np.pi / 2)
    ap.add_argument("--angle_inc_deg", type=float, default=0.5)

    ap.add_argument("--map_every", type=int, default=1, help="integrate every k-th scan")
    ap.add_argument("--beam_stride", type=int, default=None, help="override cfg.BEAM_STRIDE for rebuild")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    angle_inc = np.deg2rad(args.angle_inc_deg)
    traj = load_traj(args.traj)

    prob, traj_used = rebuild_map(
        log_path=args.log,
        traj=traj,
        angle_min=args.angle_min,
        angle_inc=angle_inc,
        map_every=args.map_every,
        beam_stride=args.beam_stride,
    )

    out_map = os.path.join(args.out_dir, f"map_{args.tag}.npy")
    out_png = os.path.join(args.out_dir, f"map_traj_{args.tag}.png")

    np.save(out_map, prob)

    # plot map + trajectory
    # plot_map_and_traj expects a GridMap object; we’ll fake it by rebuilding once more as pyramid
    # easiest: call plot_map_and_traj on a small wrapper:
    # Instead: just re-use plot_map_and_traj by reconstructing pyramid and using its finest grid again.
    # To avoid rewriting, do a tiny rebuild with pyramid again:

    # Rebuild pyramid again to pass grid to plot (cheap compared to run time)
    pyr = MapPyramid.create(
        base_res=cfg.MAP_RESOLUTION,
        size_m=cfg.MAP_SIZE_METERS,
        num_levels=cfg.PYRAMID_LEVELS,
        l0=cfg.L0,
        l_min=cfg.L_MIN,
        l_max=cfg.L_MAX,
    )
    # load saved prob into finest log-odds is not trivial; so simply plot from prob directly:
    # But your existing plot_map_and_traj already works with GridMap, so easiest is:
    # - just rebuild once more and use pyr.finest()
    # We'll do that rebuild quickly (still OK for offline once)

    # Quick second pass for plotting with your existing helper
    data = read_carmen_log(args.log)[:len(traj_used)]
    for k, e in enumerate(data):
        if args.map_every > 1 and (k % args.map_every) != 0:
            continue
        pts = scan_to_points(
            e["ranges"],
            angle_min=args.angle_min,
            angle_inc=angle_inc,
            rmin=cfg.LIDAR_MIN_RANGE,
            rmax=cfg.LIDAR_MAX_RANGE,
            stride=cfg.BEAM_STRIDE if args.beam_stride is None else args.beam_stride,
        )
        pose = traj_used[k].copy()
        pose[2] = wrap_angle(pose[2])
        pts_world = transform_points(pose, pts)
        for grid in pyr.levels:
            grid.integrate_scan_simple(
                pose=pose,
                pts_world=pts_world,
                l_free=cfg.L_FREE,
                l_occ=cfg.L_OCC,
                ray_steps=cfg.RAY_STEPS,
            )
    if args.tag == "with_lc":
        plot_map_and_traj(pyr.finest(), traj_used, out_path=out_png, title="Hector SLAM with Loop Closure")
    elif args.tag == "no_lc":
        plot_map_and_traj(pyr.finest(), traj_used, out_path=out_png, title="Hector SLAM without Loop Closure")
    else:
        plot_map_and_traj(pyr.finest(), traj_used, out_path=out_png, title="Hector SLAM: final map")

    print("Saved:", out_map)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
