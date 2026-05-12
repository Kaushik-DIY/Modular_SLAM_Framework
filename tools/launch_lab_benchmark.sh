#!/usr/bin/env bash
# =============================================================================
# tools/launch_lab_benchmark.sh
#
# Launches the lab SLAM benchmark with sleep/suspend prevention.
#
# Tries three escalating sleep-prevention strategies:
#   1. systemd-inhibit (prevents idle suspend + screen blank)
#   2. xset s off + xset -dpms  (X11 screensaver disable)
#   3. Background caffeinate loop (simulates activity every 50s)
#
# Usage:
#   cd /home/kaushik/slam_ws
#   bash tools/launch_lab_benchmark.sh
# =============================================================================

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MASTER="${REPO}/tools/master_lab_run.sh"
MASTER_LOG="${REPO}/visual_slam_outputs/lab_benchmark_master.log"

mkdir -p "${REPO}/visual_slam_outputs"

# ── sleep prevention ──────────────────────────────────────────────────────────

# Strategy 2: X11 screensaver/DPMS off (non-fatal if X not running)
xset s off 2>/dev/null && echo "[sleep-guard] xset screensaver disabled" || true
xset -dpms 2>/dev/null && echo "[sleep-guard] xset DPMS power saving disabled" || true
xset s noblank 2>/dev/null || true

# Strategy 3: background keep-alive loop (touch a file every 50s)
KEEP_ALIVE_PIDFILE="/tmp/slam_keepalive.pid"
(
  while true; do
    # Moving mouse 1px and back via xdotool prevents X idle timer
    xdotool mousemove_relative -- 1 0 2>/dev/null || true
    xdotool mousemove_relative -- -1 0 2>/dev/null || true
    touch "${REPO}/visual_slam_outputs/.keepalive_$(date +%s)"
    sleep 50
  done
) &
KEEP_ALIVE_PID=$!
echo "${KEEP_ALIVE_PID}" > "${KEEP_ALIVE_PIDFILE}"
echo "[sleep-guard] keep-alive loop started (PID ${KEEP_ALIVE_PID})"

# Cleanup on exit
cleanup() {
  echo "[sleep-guard] cleaning up keep-alive loop (PID ${KEEP_ALIVE_PID})"
  kill "${KEEP_ALIVE_PID}" 2>/dev/null || true
  xset s on 2>/dev/null || true
  xset +dpms 2>/dev/null || true
  rm -f "${KEEP_ALIVE_PIDFILE}"
}
trap cleanup EXIT INT TERM

# ── launch ────────────────────────────────────────────────────────────────────

echo "Starting benchmark at $(date)"
echo "Log: ${MASTER_LOG}"

# Strategy 1: use systemd-inhibit if available; otherwise run directly
if command -v systemd-inhibit &>/dev/null; then
  echo "[sleep-guard] using systemd-inhibit"
  systemd-inhibit \
    --what=idle:sleep:handle-lid-switch \
    --who="SLAM Benchmark" \
    --why="Running full lab RGB-D SLAM benchmark (Run A + Run B)" \
    --mode=block \
    bash "${MASTER}" 2>&1 | tee "${MASTER_LOG}"
else
  echo "[sleep-guard] systemd-inhibit not available — using keep-alive loop only"
  bash "${MASTER}" 2>&1 | tee "${MASTER_LOG}"
fi

echo "Benchmark finished at $(date)"
