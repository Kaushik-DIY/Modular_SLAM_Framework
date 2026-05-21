# Full pySLAM Alignment Review Update 2.24-2.25

| Area | pySLAM file(s) | Our file(s) | Alignment before 2.24 | Alignment after 2.25 | Remaining gap | Next action |
|---|---|---|---|---|---|---|
| Loop-triggered Global BA | `loop_closing.py`, `global_bundle_adjustment.py` | `loop_closing.py`, `global_ba.py` | Missing | Explicit loop-triggered synchronous GBA | No background process/thread | Checkpoint 2.26 benchmark |
| Global BA graph construction | `optimizer_g2o.py` | `optimizer_g2o.py`, `global_ba.py` | Thin wrapper only | All valid KFs/points with mono/RGB-D edges | No GTSAM backend | Checkpoint 2.26 |
| Global BA abort/safety | `global_bundle_adjustment.py` | `global_ba.py`, `optimizer_g2o.py` | Missing coordinator | Abort-compatible wrapper and validation | No multiprocessing flag sync | Checkpoint 2.26 |
| Global BA write-back | `global_bundle_adjustment.py` | `global_ba.py` | Immediate low-level write-back | Deferred atomic write-back | No new-KF propagation during background GBA | Checkpoint 2.26 |
| Pose-only optimizer | `optimizer_g2o.py` | `optimizer_g2o.py` | Basic parity | Adds non-finite/depth pre-filtering | Binding-specific projection API | Checkpoint 2.26 |
| Local BA | `optimizer_g2o.py`, `map.py` | `optimizer_g2o.py`, `map.py` | Basic parity | Keeps local-only outlier pruning | No parallel LBA | Checkpoint 2.26 |
| Global BA | `optimizer_g2o.py`, `global_bundle_adjustment.py` | `optimizer_g2o.py`, `global_ba.py` | Not loop-integrated | Loop-available, safe, RGB-D | Synchronous only | Checkpoint 2.26 |
| Essential graph optimizer | `optimizer_g2o.py` | `essential_graph.py` | SE3 parity with identity info | Adds weighted SE3 info matrices | No Sim3 | Checkpoint 2.26 |
| Edge types | `optimizer_g2o.py` | `optimizer_g2o.py`, `essential_graph.py` | Mono/stereo BA, SE3 graph | Same plus safer filtering | g2o compatibility wrapper | Checkpoint 2.26 |
| Information matrices | `optimizer_g2o.py` | `optimizer_g2o.py`, `essential_graph.py` | BA octave weights; graph identity | BA octave weights; weighted graph edges | SE3 weighting is conservative adaptation | Checkpoint 2.26 |
| Robust kernels | `optimizer_g2o.py` | `optimizer_g2o.py` | Pose/LBA robust kernels | Global BA robust schedule supported | No GTSAM switchable model | Checkpoint 2.26 |
| Outlier lifecycle | `optimizer_g2o.py` | `optimizer_g2o.py` | Shared pruning behavior | Local BA prunes, Global BA preserves | No semantic weights | Checkpoint 2.26 |
| Positive-depth checks | `optimizer_g2o.py` | `optimizer_g2o.py`, `global_ba.py` | Mostly post-edge | Pre-edge and validation checks | Binding-specific manual depth checks | Checkpoint 2.26 |
| Map-point normal/depth update | `map_point.py`, `global_bundle_adjustment.py` | `map_point.py`, `global_ba.py` | Present for direct BA | Recomputed after GBA write-back | No semantic descriptor update | Checkpoint 2.26 |
| Loop diagnostics | `loop_closing.py` | `loop_closing.py`, `run_tum_rgbd_smoke.py` | No GBA fields | Required GBA fields in diagnostics/log | Only populated when accepted loop occurs | Checkpoint 2.26 |
| Runtime cost | `global_bundle_adjustment.py` | `global_ba.py`, runner | Background in pySLAM | Explicit opt-in synchronous GBA | Slow on `pyslam_orb2` | Checkpoint 2.26 |
| Full benchmark readiness | pySLAM benchmark scripts | smoke/evaluation tools | GBA deferred | GBA available and auditable | Multi-dataset quantitative benchmark pending | Checkpoint 2.26 — final full dataset benchmark and thesis-ready quantitative evaluation |
