"""Thesis Results matrix — 9 method combos x 2 maps through the fusion2 REALTIME
runner at real-time pace (--speed 1.0), so the reported lag/late% are literal.

Each run lands in thesis_outputs/<UTC>/<map>/<combo>/ with run_summary.json (incl.
realtime_lag_s, late_pct, per-stage ms, RSS, tiers), verifications.csv (per loop
candidate), trajectory.tum, map.npy, map_meta.json. Headless (MPLBACKEND=Agg).

    .venv/bin/python tools/run_thesis_matrix.py                 # all 18, real-time (~90 min)
    .venv/bin/python tools/run_thesis_matrix.py --speed 0       # fast (lag/late% become ~0; for a dry run)
    .venv/bin/python tools/run_thesis_matrix.py --maps lab_hybrid_small --combos orb
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# combo id -> run_realtime CLI args (mode/frontend/verifier)
MATRIX = {
    "lidar_s2s_bnb": ["--mode", "lidar", "--lidar-frontend", "native_s2s", "--verifier", "bnb"],
    "lidar_s2s_icp": ["--mode", "lidar", "--lidar-frontend", "native_s2s", "--verifier", "icp"],
    "lidar_s2m_bnb": ["--mode", "lidar", "--lidar-frontend", "native_s2m", "--verifier", "bnb"],
    "lidar_s2m_icp": ["--mode", "lidar", "--lidar-frontend", "native_s2m", "--verifier", "icp"],
    "lidar_orb_s2s": ["--mode", "lidar_orb", "--lidar-frontend", "native_s2s"],
    "lidar_orb_s2m": ["--mode", "lidar_orb", "--lidar-frontend", "native_s2m"],
    "orb_lidar_bnb": ["--mode", "orb_lidar", "--verifier", "bnb"],
    "orb_lidar_icp": ["--mode", "orb_lidar", "--verifier", "icp"],
    "orb":           ["--mode", "orb"],
    # front-end-only (loop closure OFF): pure local mapping baselines
    "s2s_frontend":  ["--mode", "lidar", "--lidar-frontend", "native_s2s", "--no-loops"],
    "s2m_frontend":  ["--mode", "lidar", "--lidar-frontend", "native_s2m", "--no-loops"],
    "vo_frontend":   ["--mode", "orb", "--no-loops"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["lab_hybrid_small", "lab_hybrid"])
    ap.add_argument("--combos", nargs="+", default=list(MATRIX))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=Path("thesis_outputs"))
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = a.output / f"thesis_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    # truly headless: strip DISPLAY so run_realtime's LiveView never opens an
    # interactive window (its final show(block=True) would hang a batch run).
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["MPLBACKEND"] = "Agg"
    print(f"thesis matrix -> {root}  (speed={a.speed})", flush=True)

    for mp in a.maps:
        for combo in a.combos:
            run_dir = root / mp / combo
            tmp_out = root / mp / f"_tmp_{combo}"
            print(f"\n{'='*64}\n  [{mp}] {combo}  (speed {a.speed})\n{'='*64}", flush=True)
            t0 = time.perf_counter()
            argv = ["-m", "slam_core.fusion2.run_realtime",
                    "--source", "dataset", "--dataset", f"datasets/{mp}",
                    "--speed", str(a.speed), "--print-every", "500",
                    "--output", str(tmp_out)] + MATRIX[combo]
            # stdin=DEVNULL so run_realtime's stdin switch-reader gets immediate EOF
            # and the process exits cleanly (an inherited TTY stdin would block it).
            r = subprocess.run([sys.executable] + argv, env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True)
            # the runner writes to tmp_out/realtime_<mode>_<ts>/ ; move -> combo dir
            written = sorted((tmp_out).glob("realtime_*"))
            if r.returncode != 0 or not written:
                print(f"  FAILED rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir)
            shutil.move(str(written[-1]), str(run_dir))
            shutil.rmtree(tmp_out, ignore_errors=True)
            # quick line from the run summary
            import json
            s = json.load(open(run_dir / "run_summary.json"))
            print(f"  -> kf={s.get('keyframes')} prop={s.get('proposals')} "
                  f"acc={s.get('loops_accepted')} lag={s.get('realtime_lag_s')}s "
                  f"late={s.get('late_pct')}% rss={s.get('peak_rss_gb')}GB "
                  f"wall={time.perf_counter()-t0:.0f}s", flush=True)

    print(f"\nDONE -> {root}", flush=True)


if __name__ == "__main__":
    main()
