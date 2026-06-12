"""Build the V4.5 final-matrix report (loop table + runtime table + map and
trajectory montages) from a final_matrix folder.

Usage: build_matrix_report.py [<final_matrix_dir>]  (default: newest)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

run = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    sorted(Path("fusion2_outputs").glob("final_matrix_*"))[-1]
res = json.load(open(run / "all_results.json"))

PIPE = {
    "lidar_s2s_bnb": ("lidar", "scan_to_submap", "B&B"),
    "lidar_s2s_icp": ("lidar", "scan_to_submap", "ICP"),
    "lidar_s2m_bnb": ("lidar", "scan_to_map", "B&B"),
    "lidar_s2m_icp": ("lidar", "scan_to_map", "ICP"),
    "lidar_orb_s2s": ("lidar_orb", "scan_to_submap", "ORB+PnP"),
    "lidar_orb_s2m": ("lidar_orb", "scan_to_map", "ORB+PnP"),
    "orb_lidar_bnb": ("orb_lidar", "native VO", "B&B"),
    "orb_lidar_icp": ("orb_lidar", "native VO", "ICP"),
    "orb_pnp":       ("orb", "native VO", "ORB+PnP"),
}
ORDER = list(PIPE.keys())


def g(s, k, d="-"):
    v = s.get(k, d)
    return v if v is not None else d


lines = [
    "# Fusion V4 — final 9-combination validation matrix (lab_hybrid BIG)\n",
    f"Run folder: `{run}`\n",
    "## Loop-closure accounting\n",
    "| # | Combo | Mode | Front-end | Verifier | KFs | Proposed | Verified | "
    "**Accepted** | Rate |",
    "|---|---|---|---|---|---:|---:|---:|---:|---:|",
]
for i, name in enumerate(ORDER, 1):
    s = res.get(name)
    if not s:
        continue
    if "error" in s:
        lines.append(f"| {i} | `{name}` | - | - | - | FAILED: {s['error']} | | | | |")
        continue
    mode, fe, ver = PIPE[name]
    pr, ac = int(g(s, "proposals", 0)), int(g(s, "loops_accepted", 0))
    rate = f"{100 * ac / pr:.0f}%" if pr else "-"
    lines.append(f"| {i} | `{name}` | {mode} | {fe} | {ver} | {g(s,'keyframes')} | "
                 f"{pr} | {g(s,'verified')} | **{ac}** | {rate} |")

lines += [
    "\n## Runtime / memory / IMU\n",
    "| Combo | Wall (min) | fps / scan-ms | Verify ms | Peak RSS | "
    "Fallbacks / reinits | IMU | Tiers S/W/L |",
    "|---|---:|---:|---:|---:|---:|---|---:|",
]
for name in ORDER:
    s = res.get(name)
    if not s or "error" in s:
        continue
    speed = g(s, "fps", "-")
    speed = f"{speed} fps" if speed != "-" else "-"
    fb = g(s, "fallbacks", g(s, "reinits", "-"))
    tiers = f"{g(s,'stm')}/{g(s,'wm')}/{g(s,'ltm')}"
    lines.append(f"| `{name}` | {g(s,'wall_min')} | {speed} | "
                 f"{g(s,'verify_ms_mean')} | {g(s,'peak_rss_gb')} GB | {fb} | "
                 f"{'on' if g(s,'imu_active', False) else 'OFF'} | {tiers} |")

(run / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# montages
ok = [n for n in ORDER if n in res and "error" not in res[n]]
rows = math.ceil(len(ok) / 3)
fig, ax = plt.subplots(rows, 3, figsize=(24, 5.5 * rows))
for a, name in zip(np.ravel(ax), ok + [None] * (rows * 3 - len(ok))):
    if name is None:
        a.axis("off")
        continue
    p = Path(res[name]["run_dir"]) / "occupancy.png"
    if p.exists():
        a.imshow(mpimg.imread(p))
        a.set_title(f"{name}: {res[name].get('loops_accepted')} loops", fontsize=13)
    a.axis("off")
fig.suptitle("V4.5 final matrix — fused occupancy maps", fontsize=16)
fig.tight_layout()
fig.savefig(run / "maps_montage.png", dpi=80)
plt.close(fig)

fig, ax = plt.subplots(rows, 3, figsize=(20, 6 * rows))
for a, name in zip(np.ravel(ax), ok + [None] * (rows * 3 - len(ok))):
    if name is None:
        a.axis("off")
        continue
    p = Path(res[name]["run_dir"]) / "trajectory.tum"
    if p.exists():
        t = np.loadtxt(p)
        a.plot(t[:, 1], t[:, 2], "-", lw=0.9, c="navy")
        a.scatter(t[0, 1], t[0, 2], c="g", s=40, zorder=5)
        a.scatter(t[-1, 1], t[-1, 2], c="r", s=40, zorder=5)
        end_err = math.hypot(t[-1, 1] - t[0, 1], t[-1, 2] - t[0, 2])
        a.set_title(f"{name} (end-start {end_err:.2f} m)", fontsize=12)
        a.set_aspect("equal")
        a.grid(alpha=.3)
fig.suptitle("V4.5 final matrix — optimized trajectories", fontsize=16)
fig.tight_layout()
fig.savefig(run / "trajectories_montage.png", dpi=85)
plt.close(fig)
print(f"\nwrote {run}/REPORT.md + maps_montage.png + trajectories_montage.png")
