# Checkpoint 2.17 - ORB Extractor Parity Decision

## Current backend architecture

```text
FeatureTracker
  -> FeatureManager
     -> FeatureExtractorBackend
        -> OpenCVORBBackend
        -> PySLAMORB2Backend optional
```

The stable feature contract is `FeatureExtractionResult`, carrying OpenCV
keypoints, `uint8` descriptors, octave metadata, angle metadata, keypoint size
metadata, backend name, and success/message fields.

## OpenCV ORB baseline results

Generated with:

```bash
.venv/bin/python tools/compare_orb_extractors.py \
    --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
    --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_17_extractor_comparison" \
    --max-frames 30
```

Output files:

- `visual_slam_outputs/checkpoint_2_17_extractor_comparison/extractor_comparison_summary.md`
- `visual_slam_outputs/checkpoint_2_17_extractor_comparison/extractor_frame_metrics.csv`
- `visual_slam_outputs/checkpoint_2_17_extractor_comparison/opencv_orb_smoke/`

Summary:

| Backend | Available | Avg features | Descriptor | Grid coverage | Match count | Extractor FPS | Smoke |
|---|---:|---:|---|---:|---:|---:|---|
| `opencv_orb` | yes | `1864.2` | `uint8`, `(N, 32)` | `0.898` | `827.1` | `78.15` | 3/10/30 all OK |

Smoke details:

- 3 frames: `3/3 OK`, lost `0`, final keyframes `2`, map points `2234`
- 10 frames: `10/10 OK`, lost `0`, final keyframes `3`, map points `2484`
- 30 frames: `30/30 OK`, lost `0`, final keyframes `6`, map points `3473`

## pySLAM ORB2 availability and results

`orbslam2_features` is not importable inside `/home/kaushik/slam_ws/.venv`.
Therefore pySLAM ORB2 was not benchmarked. The optional backend reports
unavailable cleanly and explicit selection fails with `BackendUnavailableError`.

No C++ build was attempted and no global installation was attempted.

## Full validation

Command:

```bash
.venv/bin/python tools/validate_orbslam_pyslam_port.py \
    --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
    --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_17_validation"
```

Result:

- validation passed
- `90 passed, 1 skipped`
- local BA after-mapping keyframe stats: p90 `2.178`, z min `0.570648`, z max `1.834`
- TUM 3-frame smoke: `3/3 OK`, lost `0`
- TUM 10-frame smoke: `10/10 OK`, lost `0`
- TUM 30-frame smoke: `30/30 OK`, lost `0`, final keyframes `6`, map points `3473`

## Default backend decision

Keep `opencv_orb` as the default backend.

Reason:

- It preserves current validated tracking behavior.
- It passes backend contract tests.
- It passes TUM 3/10/30-frame smoke.
- pySLAM ORB2 cannot be evaluated until `orbslam2_features` is built/importable
  inside the project venv.

## Risks and next actions

- The current comparison cannot report pySLAM ORB2 parity until the C++ module is
  locally built in `.venv`.
- Projection-match and pose-optimization inlier counts are not currently emitted
  by the extractor-only comparison tool; the smoke summaries record downstream
  tracking OK/lost counts plus final keyframe and map-point counts.
- Next action: if the user wants pySLAM ORB2 benchmarking, build
  `third_party/pyslam_reference/thirdparty/orbslam2_features` only into
  `/home/kaushik/slam_ws/third_party/build/` and `/home/kaushik/slam_ws/.venv/`,
  then rerun the same tests and comparison.

