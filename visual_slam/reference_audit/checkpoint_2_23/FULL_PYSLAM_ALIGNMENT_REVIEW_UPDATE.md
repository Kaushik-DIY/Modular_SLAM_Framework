# Full pySLAM Alignment Review Update — Checkpoints 2.22-2.23

| Area | pySLAM file(s) | Our file(s) | Alignment before | Alignment after | Remaining gap | Next action |
|---|---|---|---|---|---|---|
| Loop projection search | `pyslam/slam/geometry_matchers.py`, `pyslam/loop_closing/loop_closing.py` | `visual_slam/orbslam/slam/geometry_matchers.py`, `visual_slam/orbslam/slam/loop_closing.py` | Direct loop matches only; loop projection fusion unimplemented. | Added loop-correction projection fusion into corrected current-side keyframes using loop keyframe plus covisible loop points. | Python implementation; no Sim3 projection path claimed. | Tune thresholds after longer loop-heavy sequences. |
| Map-point fusion | `pyslam/slam/geometry_matchers.py` | `visual_slam/orbslam/slam/geometry_matchers.py`, `visual_slam/orbslam/slam/loop_closing.py` | Narrow one-keyframe fusion. | Adds missing observations and identifies duplicates across current/loop covisible groups. | Orientation consistency is limited for map-point projection candidates. | Add per-point orientation reference if needed. |
| MapPoint replacement | `pyslam/slam/map_point.py` | `visual_slam/orbslam/slam/map_point.py` | Replacement could call bad-point cleanup after transfer and clear transferred slots. | Replacement now preserves observations, avoids duplicate keyframe observations, marks old point bad/replaced, updates descriptor/normal/depth, and removes old map point. | Frame-view transfer remains conservative for keyframes. | Revisit during broader frame-view persistence work. |
| Covisibility update | `pyslam/slam/keyframe.py`, `pyslam/loop_closing/loop_closing.py` | `visual_slam/orbslam/slam/keyframe.py`, `visual_slam/orbslam/slam/loop_closing.py` | Covisibility updated after direct fusion only. | Affected keyframes update connections after projection fusion and pose graph write-back. | No full bad-keyframe child reparenting expansion in this stage. | Keep for later map-maintenance checkpoint. |
| Loop pose correction | `pyslam/loop_closing/loop_closing.py` | `visual_slam/orbslam/slam/loop_closing.py`, `visual_slam/orbslam/slam/sim3_solver.py` | Scale-fixed RGB-D fallback with simple correction. | Correction now feeds wider fusion and explicit SE3 essential graph optimization. | Sim3 geometry solver parity is not claimed. | Continue RGB-D validation; defer monocular scope. |
| Essential graph vertices | `pyslam/slam/optimizer_g2o.py` | `visual_slam/orbslam/slam/essential_graph.py` | No explicit graph vertices. | Adds SE3 keyframe vertices from map/relevant graph neighborhood with fixed root/origin gauge. | Uses SE3 vertices instead of Sim3 vertices. | Keep SE3 for RGB-D; document Sim3 gap. |
| Spanning tree edges | `pyslam/slam/optimizer_g2o.py`, `pyslam/slam/keyframe.py` | `visual_slam/orbslam/slam/essential_graph.py`, `visual_slam/orbslam/slam/keyframe.py` | Not represented in loop optimizer. | Adds parent-child SE3 edges. | Reduced spanning-tree maintenance compared with full pySLAM bad-keyframe logic. | Expand if keyframe culling becomes loop-sensitive. |
| Covisibility edges | `pyslam/slam/optimizer_g2o.py` | `visual_slam/orbslam/slam/essential_graph.py` | Not represented in loop optimizer. | Adds strong covisibility SE3 edges with duplicate-edge filtering. | Information matrix is identity. | Tune edge weights after benchmark analysis. |
| Loop edges | `pyslam/slam/optimizer_g2o.py`, `pyslam/slam/keyframe.py` | `visual_slam/orbslam/slam/essential_graph.py`, `visual_slam/orbslam/slam/keyframe.py` | Only keyframe loop-edge bookkeeping after correction. | Adds new loop-connection edges and persistent loop edges into the optimizer graph. | Loop-edge weighting remains simple. | Tune robust weighting later. |
| Pose graph optimization | `pyslam/slam/optimizer_g2o.py` | `visual_slam/orbslam/slam/essential_graph.py` | Simple SE3 write-back, no graph optimization. | Runs g2o `VertexSE3Expmap` / `EdgeSE3Expmap` optimization with finite-pose validation and atomic write-back. | SE3 adaptation, not Sim3. | Keep Global BA separate for 2.24. |
| Map-point correction after pose graph | `pyslam/slam/optimizer_g2o.py` | `visual_slam/orbslam/slam/essential_graph.py` | Map points corrected before simple pose update. | Map points write back only after optimizer success and pose validation. | Uses corrected keyframe references from the current-side group only. | Broaden if future loop scenarios require all-map propagation. |
| Global BA | `pyslam/slam/global_bundle_adjustment.py`, `pyslam/loop_closing/loop_closing.py` | Not implemented for this stage | Deferred. | Deferred to Checkpoint 2.24. | Global BA absent by design. | Implement in Checkpoint 2.24. |

## Validation Summary

- Baseline checkpoint revalidation: 34 passed for Checkpoints 2.19-2.21.
- Final full ORB-SLAM suite: 154 passed, 1 skipped.
- Final RGB-D validation: `VALIDATION PASSED`.
- Backend durability:
  - `opencv_orb` 100/300/full: all OK; full 596/596, ATE 0.059235317.
  - `pyslam_orb2` 100/300/full: all OK; full 596/596, ATE 0.035477545.
- Loop-enabled smokes:
  - `fr1_desk` 100 and 300 frames: all OK.
  - `fr1_room` 100 and 300 frames: all OK.

Global BA is deferred to Checkpoint 2.24.
