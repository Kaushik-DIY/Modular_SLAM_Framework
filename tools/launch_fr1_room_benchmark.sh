#!/usr/bin/env bash
# Launch fr1_room benchmark: loop closing + Global BA + LM threading.
# Prevents sleep, runs evaluation and all plots after SLAM completes.
# Usage: bash tools/launch_fr1_room_benchmark.sh

set -e
REPO=/home/kaushik/slam_ws
DATASET="${REPO}/datasets/tum/rgbd_dataset_freiburg1_room"
RUN_NAME="fr1_room_loop_gba_threaded"
OUTPUT="${REPO}/visual_slam_outputs/${RUN_NAME}"
GT="${DATASET}/groundtruth.txt"
EVAL_OUT="${OUTPUT}/trajectory_eval"
PLOTS_OUT="${OUTPUT}/plots"
LOG="${OUTPUT}/run.log"

mkdir -p "${OUTPUT}"

# ---- sleep prevention ----
echo "[launch] Disabling display sleep..."
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0 2>/dev/null || true
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s off 2>/dev/null || true

# background keep-alive mouse jiggle (moves 0 pixels every 50 s)
(while true; do
  xdotool mousemove --sync 0 0 2>/dev/null || true
  sleep 50
done) &
JIGGLE_PID=$!
trap "kill ${JIGGLE_PID} 2>/dev/null; true" EXIT

# ---- activate venv ----
source "${REPO}/.venv/bin/activate"
echo "[venv] $(python -c 'import sys; print(sys.executable)')"

echo "========================================================================"
echo "Phase 1: SLAM run — fr1_room, loop + GBA + LM threading"
echo "Output:  ${OUTPUT}"
echo "Started: $(date)"
echo "========================================================================"

systemd-inhibit \
  --what=idle:sleep:handle-lid-switch \
  --why="fr1_room SLAM benchmark" \
  --mode=block \
  python -m visual_slam.orbslam.run_tum_rgbd_smoke \
    "${DATASET}" \
    --output "${OUTPUT}" \
    --max-frames 0 \
    --print-every 10 \
    --feature-backend pyslam_orb2 \
    --enable-loop-closing \
    --enable-global-ba \
    --global-ba-after-loop \
    --global-ba-iterations 10 \
    --start-local-mapping-thread \
    2>&1 | tee "${LOG}"

echo ""
echo "========================================================================"
echo "Phase 2: Trajectory evaluation against ground truth"
echo "========================================================================"

python tools/evaluate_tum_trajectory.py \
  --groundtruth "${GT}" \
  --trajectory "${OUTPUT}/$(ls ${OUTPUT}/trajectory_*.txt | xargs -n1 basename | head -1)" \
  --output "${EVAL_OUT}" \
  2>&1 | tee -a "${LOG}"

echo ""
echo "========================================================================"
echo "Phase 3: Plot generation"
echo "========================================================================"

python tools/plot_tum_evaluation.py \
  --run-dir "${OUTPUT}" \
  --groundtruth "${GT}" \
  --output "${PLOTS_OUT}" \
  2>&1 | tee -a "${LOG}"

echo ""
echo "========================================================================"
echo "All phases complete: $(date)"
echo "Run output:  ${OUTPUT}"
echo "Eval:        ${EVAL_OUT}/trajectory_metrics.json"
echo "Plots:       ${PLOTS_OUT}"
echo "========================================================================"
