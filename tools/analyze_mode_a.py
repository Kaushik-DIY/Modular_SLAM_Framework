"""
Mode A analysis: pass-through equality + SE(2) planarity preview.

1) Pass-through: compare the fusion `--mode orb` trajectory against the
   standalone run_rgbd_slam.py trajectory (should match within ORB-SLAM's own
   nondeterminism) -> confirms the runner is a faithful pass-through.
2) SE(2) preview: characterize how planar ORB-SLAM's SE(3) trajectory is on the
   wheeled-robot data (out-of-plane RMS via PCA), and how lossy an SE(2)
   projection would be -> previews whether the planar assumption is valid for
   Modes C/D on this robot (unlike handheld TUM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tum(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        rows.append([float(x) for x in p[:8]])  # t tx ty tz qx qy qz qw
    return np.array(rows)


def associate(a, b, tol=1e-3):
    bi = {round(t, 6): i for i, t in enumerate(b[:, 0])}
    ia, ib = [], []
    bt = b[:, 0]
    for i, t in enumerate(a[:, 0]):
        j = int(np.argmin(np.abs(bt - t)))
        if abs(bt[j] - t) <= tol:
            ia.append(i); ib.append(j)
    return np.array(ia), np.array(ib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone", required=True)
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--out", default="fusion_outputs/mode_a_analysis")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    A = load_tum(args.standalone)   # standalone ORB
    B = load_tum(args.fusion)       # fusion pass-through

    # --- 1) pass-through equality ---
    ia, ib = associate(A, B)
    dpos = np.linalg.norm(A[ia, 1:4] - B[ib, 1:4], axis=1)
    print("=" * 60)
    print("  MODE A — PASS-THROUGH EQUALITY (fusion vs standalone)")
    print("=" * 60)
    print(f"  poses standalone / fusion : {len(A)} / {len(B)}")
    print(f"  associated poses          : {len(ia)}")
    print(f"  position diff  max        : {dpos.max()*1000:.3f} mm")
    print(f"  position diff  mean       : {dpos.mean()*1000:.3f} mm")
    verdict = "IDENTICAL (byte/float-noise)" if dpos.max() < 1e-4 else (
        "MATCH within ORB-SLAM nondeterminism" if dpos.max() < 0.02 else "DIFFERS — investigate")
    print(f"  verdict                   : {verdict}")

    # --- 2) SE(2) planarity preview (on the standalone SE(3) trajectory) ---
    P = A[:, 1:4]
    c = P - P.mean(0)
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    # smallest singular direction = plane normal; coords along it = out-of-plane
    in_plane = c @ Vt[:2].T
    out_plane = c @ Vt[2]
    extent = np.linalg.norm(in_plane, axis=1).max() * 2
    print("\n" + "=" * 60)
    print("  SE(2) PLANARITY PREVIEW (wheeled robot, ORB SE(3) traj)")
    print("=" * 60)
    print(f"  trajectory in-plane extent : {extent:.3f} m")
    print(f"  out-of-plane RMS           : {out_plane.std()*100:.2f} cm")
    print(f"  out-of-plane max           : {np.abs(out_plane).max()*100:.2f} cm")
    print(f"  out-of-plane / extent      : {100*out_plane.std()/max(extent,1e-9):.2f} %")
    print(f"  singular values (m)        : {np.round(S/np.sqrt(len(P)),3)}")
    print(f"  -> SE(2) validity          : "
          f"{'GOOD (near-planar)' if out_plane.std() < 0.05 else 'LOSSY (significant 3D motion)'}")

    # plot: top-down in-plane trajectory + out-of-plane profile
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(in_plane[:, 0], in_plane[:, 1], "b-", lw=1)
    ax[0].scatter(in_plane[0, 0], in_plane[0, 1], c="g", s=40, label="start")
    ax[0].set_title("ORB trajectory projected to dominant plane (top-down)")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(out_plane * 100, "r-", lw=1)
    ax[1].set_title("Out-of-plane deviation (cm) vs keyframe")
    ax[1].grid(alpha=0.3); ax[1].set_ylabel("cm")
    fig.tight_layout(); fig.savefig(out / "mode_a_se2_preview.png", dpi=110)
    plt.close(fig)
    print(f"\n  plot: {out}/mode_a_se2_preview.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
