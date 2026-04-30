# ORB-SLAM pySLAM-aligned implementation strategy

Goal:
- Implement only the ORB/RGB-D SLAM path needed for benchmarkable ORB-SLAM behavior.
- Use pySLAM as the closest structural and algorithmic reference.
- Keep the final implementation self-contained under visual_slam/orbslam.
- Do not install or execute pySLAM inside this workspace.

Reference source:
- third_party/pyslam_reference
- pySLAM commit recorded in visual_slam/reference_audit/PYSLAM_REFERENCE_VERSION.md

Runtime source of truth:
- Current slam_ws virtual environment
- Installed g2o binding under .venv
- g2o projection API adapted through visual_slam/g2o_compat.py

Porting rule:
- Preserve pySLAM module boundaries as much as possible.
- Remove non-required branches:
  - deep feature extractors
  - semantic mapping
  - dense/volumetric mapping
  - GTSAM backend
  - VO-only path
  - active stereo runtime unless required by shared RGB-D APIs
- Keep RGB-D and ORB2 path.
- Keep DBoW3 loop closure path for later loop/relocalization parity.

Implementation checkpoints:
1. Package skeleton
2. Minimal config/common enums
3. Camera and pose classes
4. ORB2 feature manager/tracker/shared feature state
5. Frame, KeyFrame, MapPoint, Map
6. Optimizer g2o port with g2o_compat
7. Motion model, rotation histogram, geometry matchers
8. Tracking core and tracking
9. Local mapping core and local mapping
10. Relocalization and DBoW3 loop closing
11. TUM RGB-D benchmark runner
