#!/usr/bin/env python3
"""
Render presentation-quality views of a TSDF-fused mesh using Open3D's
offscreen renderer (proper triangle shading, not vertex scatter).

Usage
-----
python -m tools.render_tsdf_mesh_views \\
    --mesh visual_slam_outputs/.../tsdf_mesh/tsdf_mesh.ply \\
    --output visual_slam_outputs/.../tsdf_mesh \\
    --label "Phase 3 (locked-in baseline)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def setup_renderer(width=2400, height=1800) -> rendering.OffscreenRenderer:
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.05, 0.05, 0.06, 1.0])
    return renderer


def add_mesh_to_scene(renderer, mesh, name="mesh"):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.base_roughness = 0.85
    mat.base_metallic = 0.0
    renderer.scene.add_geometry(name, mesh, mat)


def add_trajectory_to_scene(renderer, traj_points, name="trajectory",
                            color=(1.0, 0.30, 0.30)):
    if traj_points is None or len(traj_points) < 2:
        return
    points = o3d.utility.Vector3dVector(traj_points)
    lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
    line_set = o3d.geometry.LineSet(points=points, lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 4.0
    renderer.scene.add_geometry(name, line_set, mat)


def add_marker_to_scene(renderer, position, name, color, radius=0.07):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    sphere.translate(np.asarray(position))
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry(name, sphere, mat)


def setup_lighting(renderer, intensity=70000):
    renderer.scene.scene.set_sun_light([0.3, -1.0, 0.3], [1.0, 1.0, 0.95], intensity)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.enable_indirect_light(True)
    renderer.scene.scene.set_indirect_light_intensity(35000)


def render_view(renderer, eye, look_at, up, fov_deg, out_path):
    renderer.setup_camera(fov_deg, look_at, eye, up)
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
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


def add_title_overlay(img_path: Path, title: str, subtitle: str | None = None):
    """Composite a title bar onto the rendered PNG using matplotlib."""
    img = plt.imread(str(img_path))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=12)
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color="#ccc")
    ax.axis("off")
    fig.patch.set_facecolor("#0d0d0d")
    plt.savefig(str(img_path), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path,
                        help="Path to tsdf_mesh.ply")
    parser.add_argument("--run", required=True, type=Path,
                        help="Run directory (for trajectory)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for renders")
    parser.add_argument("--label", default="Lab Phase 3", help="Title label")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1800)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"  vertices: {len(mesh.vertices):,}, triangles: {len(mesh.triangles):,}")

    traj = load_trajectory(args.run)
    print(f"  trajectory: {len(traj)} poses")

    # Scene center for camera positioning
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    print(f"  scene center: {center}, extent: {extent}")

    print(f"[2/3] Setting up Open3D offscreen renderer ({args.width}x{args.height})")
    renderer = setup_renderer(args.width, args.height)
    add_mesh_to_scene(renderer, mesh, "mesh")
    add_trajectory_to_scene(renderer, traj, "trajectory")
    if len(traj):
        add_marker_to_scene(renderer, traj[0], "start", [0.18, 0.80, 0.44])
        add_marker_to_scene(renderer, traj[-1], "end", [0.91, 0.30, 0.24])
    setup_lighting(renderer)

    print(f"[3/3] Rendering views ...")
    # Camera UP: world frame has Y down, so use -Y for "up"
    up = np.array([0.0, -1.0, 0.0])

    # View 1: True top-down (looking down the -Y axis from above)
    top_eye = center + np.array([0.0, -extent[1] * 4.0, extent[2] * 0.0])
    top_eye[1] = center[1] - max(extent[2], extent[0]) * 1.3  # above scene
    # Open3D top-down: eye above (in -Y dir), look_at center, up = +Z
    eye_top = center + np.array([0.0, -8.0, 0.0])
    out_top = args.output / "tsdf_render_topdown.png"
    render_view(renderer, eye_top, center, up=np.array([0.0, 0.0, 1.0]),
                fov_deg=45.0, out_path=out_top)
    add_title_overlay(out_top,
                      f"TSDF Dense Mesh — Top-down view ({args.label})",
                      f"{len(mesh.vertices):,} vertices · {len(mesh.triangles):,} triangles · 1.5 cm voxel · marching cubes")

    # View 2: 3/4 perspective from start side
    eye_persp = center + np.array([3.0, -3.0, -6.0])
    out_persp = args.output / "tsdf_render_perspective.png"
    render_view(renderer, eye_persp, center, up=up, fov_deg=55.0, out_path=out_persp)
    add_title_overlay(out_persp,
                      f"TSDF Dense Mesh — 3D Perspective ({args.label})",
                      "Photo-realistic shaded mesh, viewed from start-side")

    # View 3: Looking up the corridor toward top-loop region
    eye_corridor = np.array([0.0, -1.5, -3.0])
    look_corridor = np.array([0.0, -0.5, 6.0])
    out_corr = args.output / "tsdf_render_corridor.png"
    render_view(renderer, eye_corridor, look_corridor, up=up, fov_deg=70.0, out_path=out_corr)
    add_title_overlay(out_corr,
                      f"TSDF Dense Mesh — Corridor view ({args.label})",
                      "Looking from the start position toward the top-loop room")

    # View 4: Inside the top-loop region looking outward
    eye_inside = np.array([0.0, -1.0, 6.0])
    look_inside = np.array([0.0, -0.5, 0.0])
    out_inside = args.output / "tsdf_render_top_loop_inside.png"
    render_view(renderer, eye_inside, look_inside, up=up, fov_deg=80.0, out_path=out_inside)
    add_title_overlay(out_inside,
                      f"TSDF Dense Mesh — Inside the top-loop region ({args.label})",
                      "Looking back toward the start from the loop area")

    print(f"\nDone. Mesh renders in {args.output}")


if __name__ == "__main__":
    main()
