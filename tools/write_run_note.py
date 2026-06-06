#!/usr/bin/env python3
"""Write a RUN_NOTE.md for a finished SLAM run and rebuild visual_slam_outputs/RUNS_INDEX.md.

Convention (2026-06-06): every run keeps its own folder; each gets a short note saying
WHEN it finished, the CONFIG, the GIT COMMIT (code state), and the headline RESULTS — so
runs can be compared and we never re-run a config that hasn't changed.

Usage:
  .venv/bin/python tools/write_run_note.py <run_dir> --purpose "why this run was done"
  .venv/bin/python tools/write_run_note.py --reindex-only   # just rebuild the index
"""
import argparse
import json
import subprocess
from pathlib import Path

RUNS_ROOT = Path("visual_slam_outputs")


def git_short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def ply_count(p: Path) -> int | None:
    if not p.exists():
        return None
    with open(p, "rb") as f:
        for ln in f:
            if ln.startswith(b"element vertex"):
                return int(ln.split()[-1])
            if ln.startswith(b"end_header"):
                break
    return None


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def summarize(run_dir: Path) -> dict:
    summ = load_json(run_dir / "run_summary.json")
    cfg = load_json(run_dir / "effective_run_config.json")
    mp = ply_count(run_dir / "map_points.ply")
    return {
        "dir": run_dir.name,
        "completed": summ.get("completed_timestamp", "?"),
        "cpp": "?",  # USE_CPP_CORE not in summary; filled via --purpose/launcher knowledge
        "threaded": cfg.get("start_local_mapping_thread"),
        "loops": cfg.get("enable_loop_closing"),
        "gba": cfg.get("global_ba_after_loop"),
        "frames": summ.get("frames_attempted"),
        "avg_fps": summ.get("avg_fps"),
        "kf": summ.get("final_keyframes"),
        "map_pts": mp,
        "lost": summ.get("tracking_lost_count"),
        "accepted_loops": summ.get("accepted_loops"),
        "elapsed_sec": summ.get("elapsed_sec"),
    }


def write_note(run_dir: Path, purpose: str, commit: str) -> None:
    s = summarize(run_dir)
    fps = f"{s['avg_fps']:.2f}" if isinstance(s["avg_fps"], (int, float)) else s["avg_fps"]
    note = f"""# Run note — {s['dir']}

- **Completed:** {s['completed']}
- **Git commit:** {commit}
- **Purpose:** {purpose}

## Config
- threaded (local-mapping thread): {s['threaded']}
- loop closing: {s['loops']}    |    GBA-after-loop: {s['gba']}
- frames attempted: {s['frames']}

## Results
- avg fps: {fps}    (elapsed {s['elapsed_sec']} s)
- keyframes: {s['kf']}    |    map points: {s['map_pts']}
- tracking lost: {s['lost']}    |    accepted loops: {s['accepted_loops']}

_See run_summary.json / runtime_profile.csv / frame_timing.csv in this folder for full detail._
"""
    (run_dir / "RUN_NOTE.md").write_text(note)
    # stash machine-readable fields for the index
    s["purpose"] = purpose
    s["commit"] = commit
    (run_dir / "RUN_NOTE.json").write_text(json.dumps(s, indent=2))
    print(f"wrote {run_dir/'RUN_NOTE.md'}")


def rebuild_index() -> None:
    rows = []
    for nj in sorted(RUNS_ROOT.glob("*/RUN_NOTE.json")):
        rows.append(load_json(nj))
    rows.sort(key=lambda r: str(r.get("completed", "")))
    lines = [
        "# Runs index",
        "",
        "Every row is a finished run with its own folder + RUN_NOTE.md. "
        "Compare here before re-running a config (check the commit — if code is unchanged, reuse).",
        "",
        "| Completed | Run folder | Commit | threaded | loops | GBA | frames | fps | KF | map pts | lost | loops✓ | Purpose |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fps = f"{r['avg_fps']:.2f}" if isinstance(r.get("avg_fps"), (int, float)) else r.get("avg_fps")
        lines.append(
            f"| {r.get('completed','?')} | `{r.get('dir','?')}` | {r.get('commit','?')} | "
            f"{r.get('threaded')} | {r.get('loops')} | {r.get('gba')} | {r.get('frames')} | "
            f"{fps} | {r.get('kf')} | {r.get('map_pts')} | {r.get('lost')} | "
            f"{r.get('accepted_loops')} | {r.get('purpose','')} |"
        )
    (RUNS_ROOT / "RUNS_INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"rebuilt {RUNS_ROOT/'RUNS_INDEX.md'} ({len(rows)} runs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--purpose", default="(unspecified)")
    ap.add_argument("--commit", default=None, help="override git commit (default: current HEAD)")
    ap.add_argument("--reindex-only", action="store_true")
    a = ap.parse_args()
    if not a.reindex_only:
        if a.run_dir is None:
            ap.error("run_dir required unless --reindex-only")
        write_note(a.run_dir, a.purpose, a.commit or git_short_head())
    rebuild_index()


if __name__ == "__main__":
    main()
