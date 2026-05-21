#!/usr/bin/env bash
# =============================================================================
# tools/master_lab_run.sh
#
# Master orchestration script for lab RGB-D SLAM two-run benchmark.
#
# Runs sequentially:
#   Phase 1: Run A — baseline (no loop closing, no global BA)
#   Phase 2: Plot generation for Run A
#   Phase 3: Run B — full pipeline (loop closing + global BA)
#   Phase 4: Plot/map generation for Run B (incl. semi-dense map)
#   Phase 5: Comparison stats extraction
#
# Sleep prevention: this script must be launched from within a
#   systemd-inhibit wrapper (see launch_lab_benchmark.sh).
#
# Usage (do not call directly; use launch_lab_benchmark.sh):
#   bash tools/master_lab_run.sh 2>&1 | tee visual_slam_outputs/lab_benchmark_master.log
# =============================================================================

set -e
set -o pipefail

REPO=/home/kaushik/slam_ws
DATASET="${REPO}/datasets/lab_rgbd_run_2"

OUT_A="${REPO}/visual_slam_outputs/lab_rgbd_run_2_A_baseline"
OUT_B="${REPO}/visual_slam_outputs/lab_rgbd_run_2_B_loop_gba"
OUT_CMP="${REPO}/visual_slam_outputs/lab_comparison"

LOG_A="${REPO}/visual_slam_outputs/lab_run_A_baseline.log"
LOG_B="${REPO}/visual_slam_outputs/lab_run_B_loop_gba.log"

PLOTS_A="${OUT_A}/plots"
MAPS_B="${OUT_B}/map_figures"

VENV="${REPO}/.venv/bin/activate"

# ─── preamble ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Lab RGB-D Benchmark  —  $(date)"
echo "============================================================"

# Activate venv
# shellcheck source=/dev/null
source "${VENV}"
python -c "import sys; assert 'slam_ws/.venv' in sys.executable, sys.executable"
echo "[OK] venv: $(python -c 'import sys; print(sys.executable)')"

# ─── PHASE 1: Run A — BASELINE ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 1: Run A — baseline (no loop closing, no GBA)"
echo "  Output: ${OUT_A}"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "${OUT_A}"
python3 -u -m visual_slam.orbslam.run_tum_rgbd_smoke \
    "${DATASET}" \
    --output "${OUT_A}" \
    --max-frames 0 \
    --feature-backend pyslam_orb2 \
    --disable-loop-closing \
    --disable-global-ba \
    --print-every 100 \
    2>&1 | tee "${LOG_A}"

echo "[DONE] Phase 1 completed: $(date)"

# ─── PHASE 2: Plots for Run A ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 2: Trajectory plots for Run A"
echo "  Started: $(date)"
echo "============================================================"

# trajectory + tracking stats plots (no map export artifacts needed)
python3 -u -m tools.plot_rgbd_run \
    --run "${OUT_A}" \
    --output "${PLOTS_A}" \
    2>&1 | tee -a "${LOG_A}" || echo "[WARN] plot_rgbd_run failed for Run A — continuing"

# generate_lab_map for Run A (sparse map only — no semi-dense without PLY)
python3 -u -m tools.generate_lab_map \
    --run  "${OUT_A}" \
    --output "${OUT_A}/map_figures" \
    2>&1 | tee -a "${LOG_A}" || echo "[WARN] generate_lab_map failed for Run A (expected if no PLY) — continuing"

echo "[DONE] Phase 2 completed: $(date)"

# ─── PHASE 3: Run B — LOOP CLOSING + GLOBAL BA ───────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 3: Run B — loop closing + global BA"
echo "  Output: ${OUT_B}"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "${OUT_B}"
python3 -u -m visual_slam.orbslam.run_tum_rgbd_smoke \
    "${DATASET}" \
    --output "${OUT_B}" \
    --max-frames 0 \
    --feature-backend pyslam_orb2 \
    --enable-loop-closing \
    --enable-global-ba \
    --loop-debug \
    --dump-loop-candidate-reports \
    --print-every 100 \
    2>&1 | tee "${LOG_B}"

echo "[DONE] Phase 3 completed: $(date)"

# ─── PHASE 4: Plots + Maps for Run B ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 4: Trajectory + Map figures for Run B"
echo "  Started: $(date)"
echo "============================================================"

python3 -u -m tools.plot_rgbd_run \
    --run "${OUT_B}" \
    --output "${OUT_B}/plots" \
    2>&1 | tee -a "${LOG_B}" || echo "[WARN] plot_rgbd_run failed for Run B — continuing"

python3 -u -m tools.generate_lab_map \
    --run     "${OUT_B}" \
    --dataset "${DATASET}" \
    --output  "${MAPS_B}" \
    2>&1 | tee -a "${LOG_B}" || echo "[WARN] generate_lab_map failed for Run B — continuing"

echo "[DONE] Phase 4 completed: $(date)"

# ─── PHASE 5: Comparison stats ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PHASE 5: Comparison stats extraction"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "${OUT_CMP}"
python3 -u -m tools.compare_lab_runs \
    --run-a "${OUT_A}" \
    --run-b "${OUT_B}" \
    --output "${OUT_CMP}/comparison_summary.json" \
    2>&1 | tee "${OUT_CMP}/comparison.log"

echo "[DONE] Phase 5 completed: $(date)"

# ─── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL PHASES COMPLETE"
echo "  Run A output:     ${OUT_A}"
echo "  Run A plots:      ${PLOTS_A}"
echo "  Run B output:     ${OUT_B}"
echo "  Run B maps:       ${MAPS_B}"
echo "  Comparison JSON:  ${OUT_CMP}/comparison_summary.json"
echo "  Finished: $(date)"
echo "============================================================"
