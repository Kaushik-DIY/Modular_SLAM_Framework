"""Build the comparison report (table + map montage + trajectory montage) from a
final_4mode run folder. Usage: build_4mode_report.py <final_4mode_dir>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

run = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    sorted(Path("fusion2_outputs").glob("final_4mode_*"))[-1]
res = json.load(open(run / "all_results.json"))

ORDER = ["lidar", "orb", "orb_lidar", "lidar_orb"]
PIPELINE = {
    "lidar":     ("LiDAR scan_to_submap", "proximity (WM)", "candidate-local B&B (scan)"),
    "orb":       ("native windowed VO",   "DBoW appearance", "ORB match + PnP"),
    "orb_lidar": ("native windowed VO",   "DBoW appearance", "candidate-local B&B (LiDAR)"),
    "lidar_orb": ("LiDAR scan_to_submap", "proximity (WM)", "ORB match + PnP"),
}

def g(s, k, d="-"):
    v = s.get(k, d)
    return v if v is not None else d

# ---- loop-accounting table -------------------------------------------------
lines = []
lines.append("# Fusion v2 — final four-mode validation (lab_hybrid BIG)\n")
lines.append(f"Run folder: `{run}`\n")
lines.append("## Loop-closure accounting\n")
lines.append("| Mode | Front-end | Proposer | Verifier | Keyframes | "
             "Loops proposed | Verified | **Accepted** | Accept-rate |")
lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
for m in ORDER:
    s = res.get(m)
    if not s:
        continue
    fe, prop, ver = PIPELINE[m]
    pr, vf, ac = int(g(s, "proposals", 0)), int(g(s, "verified", 0)), int(g(s, "loops_accepted", 0))
    rate = f"{100 * ac / pr:.0f}%" if pr else "-"
    lines.append(f"| `{m}` | {fe} | {prop} | {ver} | {g(s,'keyframes')} | "
                 f"{pr} | {vf} | **{ac}** | {rate} |")

lines.append("\n## Runtime / resources\n")
lines.append("| Mode | Keyframes | Reinits (VO) / blind | Verify ms (mean) | fps | Peak RSS | Wall (min) | "
             "Tiers S/W/L |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in ORDER:
    s = res.get(m)
    if not s:
        continue
    blind = g(s, "reinits", g(s, "blind_kfs", "-"))
    tiers = f"{g(s,'stm')}/{g(s,'wm')}/{g(s,'ltm')}"
    lines.append(f"| `{m}` | {g(s,'keyframes')} | {blind} | {g(s,'verify_ms_mean')} | "
                 f"{g(s,'fps')} | {g(s,'peak_rss_gb')} GB | {g(s,'wall_min')} | {tiers} |")

# IMU calibration learned online (orb modes)
for m in ("orb", "orb_lidar"):
    s = res.get(m)
    if s and s.get("imu_calibration"):
        lines.append(f"\n- `{m}` online IMU self-calibration: {s['imu_calibration']}")

(run / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- map + trajectory montages --------------------------------------------
def occ_path(m):
    return Path(res[m]["run_dir"]) / "occupancy.png"

def traj_path(m):
    return Path(res[m]["run_dir"]) / "trajectory.tum"

fig, ax = plt.subplots(2, 2, figsize=(22, 13))
for a, m in zip(ax.ravel(), ORDER):
    if m in res and occ_path(m).exists():
        a.imshow(mpimg.imread(occ_path(m)))
        s = res[m]
        a.set_title(f"{m}: {s.get('loops_accepted')} loops accepted "
                    f"/ {s.get('proposals')} proposed", fontsize=14)
    a.axis("off")
fig.suptitle("fusion2 final — occupancy maps (lab_hybrid BIG)", fontsize=16)
fig.tight_layout()
fig.savefig(run / "maps_montage.png", dpi=85)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(20, 14))
for a, m in zip(ax.ravel(), ORDER):
    if m in res and traj_path(m).exists():
        t = np.loadtxt(traj_path(m))
        a.plot(t[:, 1], t[:, 2], "-", lw=1.0, c="navy")
        a.scatter(t[0, 1], t[0, 2], c="g", s=45, zorder=5, label="start")
        a.scatter(t[-1, 1], t[-1, 2], c="r", s=45, zorder=5, label="end")
        a.set_title(f"{m}: {res[m].get('keyframes')} keyframes", fontsize=14)
        a.set_aspect("equal"); a.grid(alpha=.3); a.legend()
fig.suptitle("fusion2 final — optimized trajectories (lab_hybrid BIG)", fontsize=16)
fig.tight_layout()
fig.savefig(run / "trajectories_montage.png", dpi=90)
plt.close(fig)
print(f"\nwrote: {run}/REPORT.md, maps_montage.png, trajectories_montage.png")
