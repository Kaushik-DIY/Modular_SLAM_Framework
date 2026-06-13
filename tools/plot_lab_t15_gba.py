#!/usr/bin/env python3
"""Visualize the lab_rgbd_run_2 loops+GBA (theta=15) run: top-down trajectory,
side view (planarity), and sparse map. Floor plane is X-Z, Y is vertical (gravity)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("visual_slam_outputs/lab_rgbd_run_2_t15_gba")
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def read_tum(p):
    xs = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        v = ln.split()
        xs.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(xs)


def read_ply_xyz(p):
    data = p.read_bytes()
    end = data.find(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace")
    n = 0
    is_bin = "binary" in header
    for ln in header.splitlines():
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
    if is_bin:
        # x,y,z float32 first; stride from property count
        props = [l for l in header.splitlines() if l.startswith("property")]
        stride = 0
        for pr in props:
            t = pr.split()[1]
            stride += {"float": 4, "float32": 4, "double": 8, "uchar": 1,
                       "uint8": 1, "int": 4, "float64": 8}.get(t, 4)
        buf = np.frombuffer(data[end:end + n * stride], dtype=np.uint8).reshape(n, stride)
        xyz = buf[:, :12].copy().view(np.float32).reshape(n, 3).astype(np.float64)
        return xyz
    else:
        arr = np.loadtxt(p, skiprows=len(header.splitlines()))
        return arr[:, :3] if arr.ndim == 2 else arr[:3].reshape(1, 3)


def load_keyframes(p):
    """Return (kf_xyz ordered by kid, kid->xyz map)."""
    d = json.loads(p.read_text())
    rows = []
    for kf in d:
        if kf.get("is_bad"):
            continue
        pos = kf.get("position")
        if pos is None and kf.get("Twc") is not None:
            pos = np.array(kf["Twc"]).reshape(4, 4)[:3, 3].tolist()
        rows.append((kf["kid"], np.array(pos[:3], float)))
    rows.sort(key=lambda r: r[0])
    kid_to_xyz = {k: v for k, v in rows}
    kf_xyz = np.array([v for _, v in rows]) if rows else np.zeros((0, 3))
    return kf_xyz, kid_to_xyz


def load_graph(p):
    d = json.loads(p.read_text())
    loops = [(e["source"], e["target"]) for e in d.get("loop_edges", [])]
    covis = [(e["source"], e["target"]) for e in d.get("covisibility_edges", [])]
    return covis, loops


def clip(pts, traj, margin=(4.0, 1.5, 4.0)):
    """Keep map points within the trajectory bounding box + per-axis margin.
    Far points are stray triangulation outliers that crush the plot scale."""
    if len(pts) == 0:
        return pts
    lo = traj.min(0) - np.array(margin)
    hi = traj.max(0) + np.array(margin)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[keep]


traj = read_tum(ROOT / "trajectory_lab_rgbd_run_2.txt")
kf, kid_xyz = load_keyframes(ROOT / "keyframes.json")
covis, loops = load_graph(ROOT / "keyframe_graph.json")
mp = read_ply_xyz(ROOT / "map_points.ply")

X, Y, Z = 0, 1, 2  # floor = X-Z, vertical = Y
print(f"poses={len(traj)} kf={len(kf)} map_pts={len(mp)} loops={len(loops)} covis={len(covis)}")

# ---- 1. Top-down trajectory (X-Z) + keyframes + loop edges ----
fig, ax = plt.subplots(figsize=(7, 11))
ax.plot(traj[:, X], traj[:, Z], "-", color="steelblue", lw=1.0, alpha=0.9, label=f"Trajectory ({len(traj)} poses)")
if len(kf):
    ax.scatter(kf[:, X], kf[:, Z], s=10, c="navy", zorder=4, label=f"Keyframes ({len(kf)})")
for a, b in loops:
    if a in kid_xyz and b in kid_xyz:
        pa, pb = kid_xyz[a], kid_xyz[b]
        ax.plot([pa[X], pb[X]], [pa[Z], pb[Z]], "r-", lw=2.2, alpha=0.9, zorder=6)
        ax.scatter([pa[X], pb[X]], [pa[Z], pb[Z]], s=45, facecolors="none",
                   edgecolors="red", linewidths=1.6, zorder=7)
ax.plot(traj[0, X], traj[0, Z], "go", ms=11, label="Start", zorder=8)
ax.plot(traj[-1, X], traj[-1, Z], "rs", ms=11, label="End", zorder=8)
if loops:
    ax.plot([], [], "r-", lw=2.2, label=f"Loop closures ({len(loops)})")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title("Lab run — top-down trajectory (loops + GBA, θ=15)\nATE-vs-reference 68 mm")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "trajectory_topdown_xz.png", dpi=150)
plt.close(fig)

# ---- 2. Side view (Z-Y) showing planarity ----
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(traj[:, Z], traj[:, Y], "-", color="steelblue", lw=1.0)
if len(kf):
    ax.scatter(kf[:, Z], kf[:, Y], s=8, c="navy", zorder=4)
ax.set_xlabel("z [m]")
ax.set_ylabel("y [m] (vertical)")
ax.set_title(f"Side view — planarity check (Y extent = {traj[:, Y].ptp()*1000:.0f} mm over {traj[:, Z].ptp():.1f} m)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "trajectory_sideview_zy.png", dpi=150)
plt.close(fig)

# ---- 3. Sparse map top-down (X-Z) + KF trajectory ----
mc = clip(mp, traj)
fig, ax = plt.subplots(figsize=(8, 11))
if len(mc) > 40000:
    mc = mc[np.random.choice(len(mc), 40000, replace=False)]
ax.scatter(mc[:, X], mc[:, Z], s=0.4, c="dimgray", alpha=0.35, rasterized=True)
if len(kf):
    ax.plot(kf[:, X], kf[:, Z], "-", color="royalblue", lw=0.8, alpha=0.8)
    ax.scatter(kf[:, X], kf[:, Z], s=8, c="navy", zorder=4)
for a, b in loops:
    if a in kid_xyz and b in kid_xyz:
        pa, pb = kid_xyz[a], kid_xyz[b]
        ax.plot([pa[X], pb[X]], [pa[Z], pb[Z]], "r-", lw=2.0, alpha=0.9, zorder=6)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title(f"Sparse map — top-down (X-Z), {len(mp):,} points")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "sparse_map_topdown_xz.png", dpi=150)
plt.close(fig)

# ---- 4. Sparse map 3D ----
mc3 = clip(mp, traj)
if len(mc3) > 25000:
    mc3 = mc3[np.random.choice(len(mc3), 25000, replace=False)]
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(mc3[:, X], mc3[:, Z], mc3[:, Y], s=0.3, c="dimgray", alpha=0.25, rasterized=True)
if len(kf):
    ax.plot(kf[:, X], kf[:, Z], kf[:, Y], "-", color="royalblue", lw=0.9)
    ax.scatter(kf[:, X], kf[:, Z], kf[:, Y], s=6, c="navy", zorder=4)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("y [m]")
ax.set_title(f"Sparse map 3D — {len(mp):,} points, {len(kf)} keyframes")
fig.tight_layout()
fig.savefig(OUT / "sparse_map_3d.png", dpi=150)
plt.close(fig)

print("wrote:")
for f in sorted(OUT.glob("*.png")):
    print(" ", f)
