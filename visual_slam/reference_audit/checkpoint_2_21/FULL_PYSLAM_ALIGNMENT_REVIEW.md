# Full pySLAM Alignment Review — Checkpoints 2.19-2.21

## Scope

This review covers the RGB-D sparse ORB path through relocalization, BoW/DBoW keyframe database, loop detection, and RGB-D loop correction. It does not claim monocular Sim3 parity.

## Alignment table

| Area | pySLAM file(s) | Our file(s) | Alignment level | Missing pieces | Reason | Next action |
|---|---|---|---|---|---|---|
| Parameters/config | `config_parameters.py` | `config_parameters.py` | Medium | Full pySLAM parameter surface | Local path only exposes needed sparse RGB-D parameters | Add loop/optimizer knobs as more parity lands |
| Camera | `slam/camera.py` | `slam/camera.py` | High | Full sensor variants | RGB-D TUM target only | Keep RGB-D stable |
| Feature extractor | `local_features/*` | `local_features/*` | Medium | Full pySLAM extractor matrix | Scope is OpenCV ORB and optional ORB2 | Preserve backend contract |
| Feature matcher | `local_features/feature_matcher.py` | `local_features/feature_matcher.py`, `slam/bow_matcher.py` | Medium | Non-ORB matchers | Sparse ORB target | Add only if benchmark requires |
| ORB2 C++ extractor | `thirdparty/orbslam2_features` style | `feature_orbslam2.py`, local build tooling | High | Default switch not approved | Baseline keeps `opencv_orb` default | Keep optional backend |
| DBoW vocabulary install | `loop_detector_vocabulary.py`, pydbow3 assets | `tools/install_pyslam_vocabulary_local.sh` | High | Network fallback confirmation handling | Local bundled vocab exists | Keep local script reproducible |
| pydbow3 C++ binding | `thirdparty/pydbow3/src/py_dbow3.cpp` | `tools/build_pyslam_pydbow3_local.sh`, local copy under `third_party/build` | Medium | Upstream binding is patched locally | Needed FeatureVector exposure safely | Keep patch scripted and local |
| BoW wrapper | `loop_detector_dbow3.py` | `slam/bow.py` | High | Full database persistence | In-process wrapper is enough for checkpoints | Add persistence later |
| BoW-guided matching | ORB-SLAM SearchByBoW pattern | `slam/bow_matcher.py` | High | Exact C++ matcher parity | Python implementation mirrors control flow | Tune only with benchmark evidence |
| Frame | `slam/frame.py` | `slam/frame.py` | Medium | Serialization, extra modalities | RGB-D sparse subset | Fill only when needed |
| KeyFrame | `slam/keyframe.py` | `slam/keyframe.py` | Medium | Full reload/serialization and full bad-KF complexity | Local subset benchmark-focused | Extend with map persistence |
| MapPoint | `slam/map_point.py` | `slam/map_point.py` | Medium | Semantic/dense fields | Excluded scope | Keep sparse behavior stable |
| Map | `slam/map.py` | `slam/map.py` | Medium | Full multi-map/session logic | Excluded unless requested | Add only for later checkpoints |
| KeyFrameDatabase | `loop_closing/keyframe_database.py` | `slam/keyframe_database.py` | High | C++ DBOW2 database persistence | Python inverted file follows scoring flow | Add save/load if needed |
| Tracking | `slam/tracking.py` | `slam/tracking.py` | Medium | Full pySLAM state-machine breadth | RGB-D sparse subset | Continue structural parity |
| Relocalization | `slam/relocalizer.py` | `slam/relocalizer.py` | Medium-High | MLPnP wrapper, async detector path | Local wrapper unavailable | Replace OpenCV PnP only if wrapper is safely added |
| Local mapping | `slam/local_mapping.py`, `local_mapping_core.py` | `slam/local_mapping.py`, `local_mapping_core.py` | Medium | Threaded pySLAM behavior | Sequential for stability | Revisit threading after loop parity |
| Pose-only optimizer | `slam/optimizer_g2o.py` | `slam/optimizer_g2o.py` | Medium | Exact pySLAM edge API parity | Python g2o binding differences | Keep synthetic optimizer tests |
| Local BA | `slam/optimizer_g2o.py` | `slam/optimizer_g2o.py` | Medium | Full robust edge lifecycle parity | Binding differences | Continue audit before tuning |
| Loop detector | `loop_detector_dbow3.py`, `loop_detector_base.py` | `slam/loop_detector.py` | Medium | Multiprocess detector, image debug products | In-process local checkpoint target | Add process/persistence later |
| Loop consistency | `loop_closing.py` | `slam/loop_closing.py` | High | Debug image support | Not needed for headless validation | Keep tests |
| Loop geometry verification | `loop_closing.py`, `sim3solver` | `slam/loop_closing.py`, `slam/sim3_solver.py` | Medium | pySLAM RANSAC Sim3 solver and optimize_sim3 | Wrapper unavailable; RGB-D scale known | Add Sim3 optimizer if binding path is stable |
| Loop correction | `loop_closing.py` | `slam/loop_closing.py`, `slam/essential_graph.py` | Medium | Full Sim3 correction propagation | RGB-D SE3 fallback only | Improve pose graph parity |
| Essential graph / pose graph | `optimizer_g2o.optimize_essential_graph` | `slam/essential_graph.py` | Low-Medium | Full Sim3 essential graph optimization | Python fallback is conservative | Future checkpoint |
| Map-point fusion | `geometry_matchers.py`, `loop_closing.py` | `slam/loop_closing.py`, `geometry_matchers.py` | Medium | Full projection search/fuse breadth | Minimal validated fusion now | Port wider projection search |
| Global BA | `global_bundle_adjustment.py` | `optimizer_g2o.global_bundle_adjustment` exists, loop GBA not wired | Deferred | Loop-triggered GBA | Explicitly outside this task | Separate checkpoint |
| Runner/evaluation tools | pySLAM runners/evaluation | `run_tum_rgbd_smoke.py`, `tools/validate_orbslam_pyslam_port.py`, `tools/run_orb_backend_durability.py` | Medium | Full benchmark harness | Smoke/durability gates exist | Add loop benchmark tool |

## Validation summary

- Checkpoint 2.19/2.20/BoW-guided matching focused tests: `23 passed`.
- Checkpoint 2.21 loop tests: `11 passed`.
- Final full ORB-SLAM suite: `134 passed, 1 skipped`.
- Default TUM RGB-D validation on `rgbd_dataset_freiburg1_desk`: `VALIDATION PASSED`.
- Backend durability on `fr1_desk`:
  - `opencv_orb`: `100/100`, `300/300`, `596/596` OK, no lost frames, eval `ok`.
  - `pyslam_orb2`: `100/100`, `300/300`, `596/596` OK, no lost frames, eval `ok`.
- Loop-enabled smoke on `fr1_desk`, 30 frames: `30/30 OK`, final state `OK`.
- Real loop-capable local dataset smoke on `fr1_room`, 60 frames: `60/60 OK`, final state `OK`.

## Important deviations

- OpenCV EPnP/RANSAC replaces pySLAM `MLPnPsolver`.
- pySLAM `sim3solver` is unavailable; RGB-D uses scale-fixed SE3 correction.
- Local `g2o` exposes Sim3 classes, but full pySLAM Sim3 essential graph optimization is not ported.
- Loop detection/correction is in-process, not pySLAM's multiprocessing loop-detection process.
- Global BA after loop correction is deferred.

## Remaining gaps

- Full monocular Sim3 parity.
- Full pySLAM essential graph optimizer parity.
- Full projection search and fusion breadth during loop correction.
- Loop-triggered global BA.
- Full real loop-dataset benchmark over a complete loop sequence.

No blocking gaps remain for the requested RGB-D 2.19-2.21 checkpoint scope, with the caveat that loop correction is the documented RGB-D SE3 fallback rather than monocular Sim3 parity.
