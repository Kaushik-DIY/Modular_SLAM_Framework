#!/usr/bin/env python3
"""
Build a TSDF-fused dense reconstruction from a completed RGB-D SLAM run.

Uses the SLAM pipeline's loop-corrected keyframe poses as a pose oracle and
fuses every per-keyframe depth observation into a signed-distance volume via
Open3D's ScalableTSDFVolume. Extracts a watertight triangle mesh via marching
cubes, plus a colored point cloud.

This is "Path A" — dense mapping on top of sparse SLAM. The core SLAM logic
(tracking, mapping, loop closure) is unchanged; only the post-processing /
visualisation stage is upgraded from per-pixel projection + voxel-binning to
proper TSDF fusion.

Outputs
-------
  tsdf_mesh.ply               Watertight triangle mesh (vertex-colored)
  tsdf_pointcloud.ply         Mesh vertices as a clean point cloud
  tsdf_mesh_topdown.png       Top-down render (shaded)
  tsdf_mesh_3d.png            Perspective render (shaded)
  tsdf_mesh_height.png        Top-down with vertex height as color

Usage
-----
python -m tools.build_tsdf_dense_map \\
    --run visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened \\
    --dataset datasets/lab_rgbd_run_2 \\
    --output visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened/tsdf_mesh \\
    --voxel-size 0.015 --sdf-trunc 0.06 --max-depth 4.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def read_camera_yaml(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%YAML"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def read_associations(path: Path) -> list[tuple[float, str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            out.append((float(parts[0]), parts[1], parts[3]))
    return out


def build_tsdf_volume(
    keyframes_json: Path,
    dataset_path: Path,
    *,
    voxel_size: float = 0.015,
    sdf_trunc: float = 0.06,
    max_depth_m: float = 4.0,
    min_depth_m: float = 0.2,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """Fuse all keyframes' RGB-D images into a TSDF volume."""
    cam = read_camera_yaml(dataset_path / "camera.yaml")
    fx = float(cam["Camera.fx"])
    fy = float(cam["Camera.fy"])
    cx = float(cam["Camera.cx"])
    cy = float(cam["Camera.cy"])
    depth_scale = float(cam["DepthMapFactor"])

    width, height = 640, 480
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    kf_data = json.loads(keyframes_json.read_text())
    assoc = read_associations(dataset_path / "associations.txt")
    assoc_ts = np.asarray([a[0] for a in assoc], dtype=np.float64)

    print(f"  TSDF volume: voxel_size={voxel_size*1000:.0f} mm, "
          f"sdf_trunc={sdf_trunc*1000:.0f} mm, max_depth={max_depth_m} m")
    print(f"  Fusing {len(kf_data)} keyframes ...")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_integrated = 0
    for i, kf in enumerate(kf_data):
        try:
            Twc = np.asarray(kf["Twc"], dtype=np.float64).reshape(4, 4)
            ts = float(kf.get("timestamp") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue

        idx = int(np.argmin(np.abs(assoc_ts - ts)))
        _, rgb_rel, dep_rel = assoc[idx]
        rgb_path = dataset_path / rgb_rel
        dep_path = dataset_path / dep_rel
        if not rgb_path.exists() or not dep_path.exists():
            continue

        rgb_bgr = cv2.imread(str(rgb_path))
        depth_raw = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_raw is None:
            continue

        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_raw))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=max_depth_m,
            convert_rgb_to_intensity=False,
        )

        # Open3D expects extrinsic = world->camera (Tcw). Twc is camera->world.
        Tcw = np.linalg.inv(Twc)
        volume.integrate(rgbd, intrinsic, Tcw)
        n_integrated += 1

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(kf_data)} keyframes fused")

    print(f"  Done: {n_integrated} keyframes integrated into TSDF")
    return volume


def render_mesh_topdown_shaded(mesh: o3d.geometry.TriangleMesh, traj, out_path, title):
    """Top-down render with simple Lambert-style shading via vertex normals."""
    verts = np.asarray(mesh.vertices)
    cols = np.asarray(mesh.vertex_colors)
    norms = np.asarray(mesh.vertex_normals)
    # Y is the world "down" axis. Render top-down by plotting X vs Z.
    # Simple shading: dot(normal, (0, -1, 0)) emphasizes horizontal surfaces.
    light = np.array([0.3, -0.8, 0.4])
    light /= np.linalg.norm(light)
    shading = np.clip(norms @ light, 0.2, 1.0)  # 0.2 ambient floor
    shaded = cols * shading[:, None]
    shaded = np.clip(shaded, 0, 1)

    fig, ax = plt.subplots(figsize=(13, 16))
    # Draw points sorted by Y so floor renders first, walls/ceiling on top
    order = np.argsort(verts[:, 1])[::-1]
    ax.scatter(verts[order, 0], verts[order, 2], c=shaded[order], s=0.5,
               marker=".", linewidths=0)
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 2], "-", color="#27ae60", lw=1.5, alpha=0.9, label="Trajectory")
        ax.scatter([traj[0,0]], [traj[0,2]], c="#2ecc71", s=160, zorder=6, label="Start",
                   edgecolors="black", linewidths=1.5)
        ax.scatter([traj[-1,0]], [traj[-1,2]], c="#e74c3c", s=160, marker="X", zorder=6,
                   label="End", edgecolors="black", linewidths=1.5)
    ax.set_xlabel("X (m, right)", fontsize=12)
    ax.set_ylabel("Z (m, forward)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
    ax.tick_params(colors="#888")
    ax.xaxis.label.set_color("#444")
    ax.yaxis.label.set_color("#444")
    ax.legend(loc="upper right", fontsize=10, facecolor="#222",
              labelcolor="white", edgecolor="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def render_mesh_height_topdown(mesh, traj, out_path, title):
    verts = np.asarray(mesh.vertices)
    height = -verts[:, 1]
    fig, ax = plt.subplots(figsize=(13, 16))
    order = np.argsort(verts[:, 1])[::-1]
    sc = ax.scatter(verts[order, 0], verts[order, 2], c=height[order], s=0.5,
                    marker=".", linewidths=0, cmap="viridis",
                    vmin=np.percentile(height, 2), vmax=np.percentile(height, 98))
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 2], "-", color="#ff5252", lw=1.5, alpha=0.9, label="Trajectory")
        ax.scatter([traj[0,0]], [traj[0,2]], c="#2ecc71", s=160, zorder=6, label="Start",
                   edgecolors="black", linewidths=1.5)
        ax.scatter([traj[-1,0]], [traj[-1,2]], c="#e74c3c", s=160, marker="X", zorder=6,
                   label="End", edgecolors="black", linewidths=1.5)
    ax.set_xlabel("X (m, right)", fontsize=12)
    ax.set_ylabel("Z (m, forward)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55)
    cbar.set_label("Height above floor (m)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def render_mesh_3d(mesh, traj, out_path, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    verts = np.asarray(mesh.vertices)
    cols = np.asarray(mesh.vertex_colors)
    norms = np.asarray(mesh.vertex_normals)
    n = len(verts)
    if n > 250_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 250_000, replace=False)
        verts = verts[idx]; cols = cols[idx]; norms = norms[idx]
    light = np.array([0.3, -0.8, 0.4])
    light /= np.linalg.norm(light)
    shading = np.clip(norms @ light, 0.2, 1.0)
    shaded = np.clip(cols * shading[:, None], 0, 1)

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 2], -verts[:, 1], c=shaded, s=0.5, marker=".", linewidths=0)
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 2], -traj[:, 1], "-", color="#ff5252", lw=2.0, alpha=0.95)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("-Y (m, up)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1, 1.4, 0.4))
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def load_trajectory(run_dir: Path) -> np.ndarray:
    cands = sorted(run_dir.glob("trajectory_*.txt"))
    cands = [c for c in cands if "completed_" not in c.name] or cands
    if not cands:
        return np.empty((0, 3))
    poses = []
    with cands[0].open() as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.015,
                        help="TSDF voxel size in metres (default 1.5 cm)")
    parser.add_argument("--sdf-trunc", type=float, default=0.06,
                        help="Truncation distance for SDF (default 6 cm)")
    parser.add_argument("--max-depth", type=float, default=4.0,
                        help="Discard depths beyond this (m)")
    parser.add_argument("--label", default=None, help="Title label for plots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    label = args.label or args.run.name

    print(f"[1/4] Fusing TSDF from {args.run.name} ...")
    volume = build_tsdf_volume(
        args.run / "keyframes.json", args.dataset,
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        max_depth_m=args.max_depth,
    )

    print(f"[2/4] Extracting mesh via marching cubes ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"  Mesh: {n_verts:,} vertices, {n_tris:,} triangles")

    # Crop to scene bounding box (removes far stray voxels)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([-4.0, -2.0, -3.0]),
        max_bound=np.array([4.5, 2.5, 12.0]),
    )
    mesh = mesh.crop(bbox)
    mesh.compute_vertex_normals()
    print(f"  After scene crop: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    print(f"[3/4] Saving mesh artifacts ...")
    o3d.io.write_triangle_mesh(str(args.output / "tsdf_mesh.ply"), mesh)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(str(args.output / "tsdf_pointcloud.ply"), pcd)
    print(f"  saved: tsdf_mesh.ply ({len(mesh.triangles):,} triangles)")
    print(f"  saved: tsdf_pointcloud.ply ({len(mesh.vertices):,} vertices)")

    print(f"[4/4] Rendering presentation views ...")
    traj = load_trajectory(args.run)
    print(f"  trajectory: {len(traj)} poses")

    render_mesh_topdown_shaded(
        mesh, traj,
        args.output / "tsdf_mesh_topdown.png",
        f"TSDF-fused dense map (top-down, shaded) — {label}\n"
        f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles, voxel={args.voxel_size*1000:.0f}mm")
    render_mesh_height_topdown(
        mesh, traj,
        args.output / "tsdf_mesh_height.png",
        f"TSDF-fused dense map (height-colored) — {label}\n"
        f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    render_mesh_3d(
        mesh, traj,
        args.output / "tsdf_mesh_3d.png",
        f"TSDF-fused dense map (3D perspective, shaded) — {label}\n"
        f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    print(f"\nDone. Outputs in {args.output}")
    print("Mesh can be opened in MeshLab, CloudCompare, Blender, etc.")


if __name__ == "__main__":
    main()
