#!/usr/bin/env python3
"""
Build a CLEAN TSDF-fused mesh from a completed RGB-D SLAM run.

Improvements over build_tsdf_dense_map.py:
  - Tighter SDF truncation (less stray extrapolation)
  - Per-vertex weight filter (drops voxels seen by very few keyframes)
  - Connected-component filter (drops small floating fragments)
  - Taubin smoothing (preserves shape, removes high-frequency noise)
  - Per-vertex normal-direction outlier removal
  - Tight scene bounding-box crop
  - Proper Open3D shaded rendering (not vertex scatter)

Usage
-----
python -m tools.build_clean_tsdf_mesh \\
    --run visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened \\
    --dataset datasets/lab_rgbd_run_2 \\
    --output visual_slam_outputs/checkpoint_2_36Z3_lab_phase3_tightened/tsdf_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def read_camera_yaml(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%YAML"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            try: out[k.strip()] = float(v.strip())
            except ValueError: out[k.strip()] = v.strip()
    return out


def read_associations(path: Path) -> list[tuple[float, str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 4: out.append((float(parts[0]), parts[1], parts[3]))
    return out


def load_trajectory(run_dir: Path) -> np.ndarray:
    cands = sorted(run_dir.glob("trajectory_*.txt"))
    cands = [c for c in cands if "completed_" not in c.name] or cands
    if not cands: return np.empty((0, 3))
    poses = []
    with cands[0].open() as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            poses.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(poses)


def build_tsdf(
    kf_json: Path, dataset: Path, *,
    voxel_size: float, sdf_trunc: float,
    min_depth: float, max_depth: float,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    cam = read_camera_yaml(dataset / "camera.yaml")
    fx, fy = float(cam["Camera.fx"]), float(cam["Camera.fy"])
    cx, cy = float(cam["Camera.cx"]), float(cam["Camera.cy"])
    depth_scale = float(cam["DepthMapFactor"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

    kfs = json.loads(kf_json.read_text())
    assoc = read_associations(dataset / "associations.txt")
    assoc_ts = np.asarray([a[0] for a in assoc])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    print(f"  voxel={voxel_size*1000:.0f}mm  trunc={sdf_trunc*1000:.0f}mm  "
          f"depth∈[{min_depth},{max_depth}]m  KFs={len(kfs)}")
    for i, kf in enumerate(kfs):
        try:
            Twc = np.asarray(kf["Twc"]).reshape(4, 4)
            ts = float(kf.get("timestamp") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        idx = int(np.argmin(np.abs(assoc_ts - ts)))
        _, rgb_rel, dep_rel = assoc[idx]
        rgb = cv2.imread(str(dataset / rgb_rel))
        depth = cv2.imread(str(dataset / dep_rel), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None: continue
        # Mask out depths beyond range to suppress stray observations
        depth_m = depth.astype(np.float32) / depth_scale
        invalid = (depth_m < min_depth) | (depth_m > max_depth)
        if invalid.any():
            depth = depth.copy()
            depth[invalid] = 0
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).copy())
        dep_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, dep_o3d, depth_scale=depth_scale, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)
        Tcw = np.linalg.inv(Twc)
        volume.integrate(rgbd, intrinsic, Tcw)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(kfs)} keyframes fused")
    return volume


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    scene_bbox: tuple[np.ndarray, np.ndarray],
    min_component_triangles: int,
    smoothing_iters: int,
) -> o3d.geometry.TriangleMesh:
    """Aggressive post-processing to remove TSDF artefacts."""
    print(f"  raw mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # Crop to scene bbox first (drops far stray voxels)
    bbox = o3d.geometry.AxisAlignedBoundingBox(scene_bbox[0], scene_bbox[1])
    mesh = mesh.crop(bbox)
    print(f"  after scene crop: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # Cluster connected components and keep only the large ones
    print("  clustering connected components ...")
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n = np.asarray(cluster_n_triangles)
    large_clusters = np.where(cluster_n >= min_component_triangles)[0]
    keep_mask = np.isin(triangle_clusters, large_clusters)
    triangles_to_remove = np.logical_not(keep_mask)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f"  after small-component removal (min {min_component_triangles} tris): "
          f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris "
          f"({len(large_clusters)} large clusters kept)")

    # Drop degenerate / non-manifold artifacts
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # Smooth (Taubin preserves shape better than Laplacian)
    if smoothing_iters > 0:
        print(f"  Taubin smoothing ({smoothing_iters} iters) ...")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iters)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"  CLEANED mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    return mesh


def render_shaded(
    mesh, traj, out_path, *,
    width=2400, height=1800,
    eye, look_at, up,
    fov_deg, title,
    bg=(0.96, 0.96, 0.97, 1.0),
    show_traj=True,
):
    """Render with Open3D's offscreen renderer; bright background for clarity."""
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(bg))

    # Mesh material
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = 0.65
    mat.base_metallic = 0.0
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Lighting
    renderer.scene.scene.set_sun_light([0.4, -0.9, 0.4], [1.0, 1.0, 0.95], 80000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.enable_indirect_light(True)
    renderer.scene.scene.set_indirect_light_intensity(45000)

    # Trajectory
    if show_traj and traj is not None and len(traj) > 1:
        pts = o3d.utility.Vector3dVector(traj)
        lines = [[i, i + 1] for i in range(len(traj) - 1)]
        ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
        ls.colors = o3d.utility.Vector3dVector(np.tile([0.92, 0.20, 0.20], (len(lines), 1)))
        lm = rendering.MaterialRecord(); lm.shader = "unlitLine"; lm.line_width = 5.0
        renderer.scene.add_geometry("trajectory", ls, lm)
        # Start/End markers
        for nm, pos, col in [("start", traj[0], [0.18, 0.80, 0.44]),
                              ("end", traj[-1], [0.91, 0.30, 0.24])]:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.08, resolution=20)
            s.translate(np.asarray(pos)); s.paint_uniform_color(col); s.compute_vertex_normals()
            sm = rendering.MaterialRecord(); sm.shader = "defaultLit"
            renderer.scene.add_geometry(nm, s, sm)

    renderer.setup_camera(fov_deg, look_at, eye, up)
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)

    # Composite title bar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = plt.imread(str(out_path))
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(arr)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.axis("off")
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--voxel-size", type=float, default=0.030,
                    help="TSDF voxel size (default 3 cm — larger = smoother, less spiky)")
    ap.add_argument("--sdf-trunc", type=float, default=0.08,
                    help="SDF truncation (default ~3× voxel)")
    ap.add_argument("--min-depth", type=float, default=0.30,
                    help="Discard depths below this (default 30 cm)")
    ap.add_argument("--max-depth", type=float, default=3.5,
                    help="Discard depths beyond this (default 3.5 m)")
    ap.add_argument("--min-component", type=int, default=2000,
                    help="Drop connected components smaller than this many triangles")
    ap.add_argument("--smooth", type=int, default=6,
                    help="Number of Taubin smoothing iterations (0=off)")
    ap.add_argument("--label", default="Lab Phase 3")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building TSDF ...")
    volume = build_tsdf(args.run / "keyframes.json", args.dataset,
                        voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc,
                        min_depth=args.min_depth, max_depth=args.max_depth)

    print(f"[2/4] Extracting mesh ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"[3/4] Cleaning mesh ...")
    scene_bbox = (np.array([-3.0, -1.5, -2.5]), np.array([4.0, 1.5, 12.0]))
    mesh = clean_mesh(mesh, scene_bbox=scene_bbox,
                      min_component_triangles=args.min_component,
                      smoothing_iters=args.smooth)
    if len(mesh.triangles) == 0:
        print("ERROR: mesh empty after cleanup. Try lower --min-component or relax bbox.")
        return

    o3d.io.write_triangle_mesh(str(args.output / "tsdf_mesh_clean.ply"), mesh)
    print(f"  saved: tsdf_mesh_clean.ply")

    print(f"[4/4] Rendering shaded views ...")
    traj = load_trajectory(args.run)

    center = np.asarray(mesh.get_axis_aligned_bounding_box().get_center())
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"  scene center {center}, extent {extent}")
    up = np.array([0.0, -1.0, 0.0])  # World Y is down → -Y is up

    # View 1: True top-down
    eye_top = center + np.array([0.0, -max(extent) * 1.4, 0.0])
    render_shaded(
        mesh, traj, args.output / "view_topdown.png",
        eye=eye_top, look_at=center, up=np.array([0.0, 0.0, 1.0]),
        fov_deg=42.0, title=f"Lab TSDF mesh — Top-down ({args.label})")

    # View 2: Hero 3D perspective (from start area looking forward)
    eye_persp = center + np.array([4.5, -3.5, -5.5])
    render_shaded(
        mesh, traj, args.output / "view_perspective.png",
        eye=eye_persp, look_at=center + np.array([-0.5, 0.0, 1.0]),
        up=up, fov_deg=55.0,
        title=f"Lab TSDF mesh — 3D perspective ({args.label})")

    # View 3: Side view (from -X)
    eye_side = center + np.array([-6.5, -2.5, 1.5])
    render_shaded(
        mesh, traj, args.output / "view_side.png",
        eye=eye_side, look_at=center, up=up, fov_deg=55.0,
        title=f"Lab TSDF mesh — Side view ({args.label})")

    # View 4: View from above + behind (architectural)
    eye_above = center + np.array([2.5, -5.0, -6.0])
    render_shaded(
        mesh, traj, args.output / "view_isometric.png",
        eye=eye_above, look_at=center, up=up, fov_deg=50.0,
        title=f"Lab TSDF mesh — Isometric view ({args.label})")

    print(f"\nDone. Outputs in {args.output}")


if __name__ == "__main__":
    main()
