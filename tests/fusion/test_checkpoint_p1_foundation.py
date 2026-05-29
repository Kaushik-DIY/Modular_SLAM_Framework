"""
Phase 1 checkpoint — Foundation.

Covers the "Done when" criteria from CLAUDE.md §4 Phase 1:
  - SoftSync window logic (match / nearest / out-of-window / streaming hold).
  - Pose3D -> Pose2 projection.
Plus supporting coverage for lidar_synth, the dataset loader, and config.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.fusion.config import FusionConfig, Mode
from slam_core.fusion.signature import Signature, project_pose3d_to_pose2
from slam_core.fusion.sync import SoftSync
from slam_core.fusion.lidar_synth import synthesize_2d_scan

DATASET = Path("datasets/tum/rgbd_dataset_freiburg1_room")


# --------------------------------------------------------------------------
# Pose3D -> Pose2 projection
# --------------------------------------------------------------------------

def _se3(rot_z: float = 0.0, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    c, s = math.cos(rot_z), math.sin(rot_z)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = t
    return T


def test_projection_identity():
    p = project_pose3d_to_pose2(np.eye(4))
    assert p == Pose2(0.0, 0.0, 0.0)


def test_projection_translation_drops_z():
    p = project_pose3d_to_pose2(_se3(t=(1.5, -2.0, 9.9)))
    assert p.x == pytest.approx(1.5)
    assert p.y == pytest.approx(-2.0)
    assert p.theta == pytest.approx(0.0)


def test_projection_extracts_yaw():
    p = project_pose3d_to_pose2(_se3(rot_z=0.7, t=(1.0, 2.0, 0.0)))
    assert p.theta == pytest.approx(0.7)
    assert p.x == pytest.approx(1.0)
    assert p.y == pytest.approx(2.0)


def test_projection_with_base_extrinsic():
    # base_T_cam = pure translation -> T_world_base = inv(base_T_cam).
    base_T_cam = _se3(t=(0.3, -0.1, 0.2))
    p = project_pose3d_to_pose2(np.eye(4), base_T_cam=base_T_cam)
    assert p.x == pytest.approx(-0.3)
    assert p.y == pytest.approx(0.1)
    assert p.theta == pytest.approx(0.0)


def test_projection_accepts_matrix_object():
    class _Iso:
        def __init__(self, m):
            self._m = m

        def matrix(self):
            return self._m

    p = project_pose3d_to_pose2(_Iso(_se3(rot_z=-0.4, t=(5.0, 6.0, 1.0))))
    assert p.theta == pytest.approx(-0.4)
    assert p.x == pytest.approx(5.0)
    assert p.y == pytest.approx(6.0)


def test_projection_rejects_bad_shape():
    with pytest.raises(ValueError):
        project_pose3d_to_pose2(np.eye(3))


# --------------------------------------------------------------------------
# SoftSync window logic
# --------------------------------------------------------------------------

def _scan(tag: float) -> np.ndarray:
    return np.full((1, 2), tag, dtype=float)


def test_sync_exact_match():
    s = SoftSync(tolerance_s=0.05)
    s.push_rgbd(1.0, "rgb", "depth")
    s.push_scan(1.0, _scan(1.0))
    # Need a later scan to finalize without flush.
    s.push_scan(1.06, _scan(1.06))
    out = s.try_pop()
    assert out is not None
    assert out.t == 1.0
    assert out.scan is not None and out.scan[0, 0] == pytest.approx(1.0)


def test_sync_picks_nearest_within_window():
    s = SoftSync(tolerance_s=0.05)
    s.push_rgbd(1.0, "rgb", "depth")
    s.push_scan(0.97, _scan(0.97))  # dt 0.03
    s.push_scan(1.02, _scan(1.02))  # dt 0.02 -> nearest
    s.push_scan(1.06, _scan(1.06))  # triggers finalize
    out = s.try_pop()
    assert out is not None
    assert out.scan[0, 0] == pytest.approx(1.02)


def test_sync_out_of_window_yields_none_scan():
    s = SoftSync(tolerance_s=0.05)
    s.push_rgbd(1.0, "rgb", "depth")
    s.push_scan(5.0, _scan(5.0))  # far past window, triggers finalize
    out = s.try_pop()
    assert out is not None
    assert out.t == 1.0
    assert out.scan is None


def test_sync_streaming_hold_then_release():
    s = SoftSync(tolerance_s=0.05)
    s.push_rgbd(1.0, "rgb", "depth")
    s.push_scan(1.01, _scan(1.01))  # inside window but no later scan yet
    assert s.try_pop() is None      # must hold: a closer scan could arrive
    s.push_scan(1.06, _scan(1.06))  # now safe to decide
    out = s.try_pop()
    assert out is not None
    assert out.scan[0, 0] == pytest.approx(1.01)  # 1.06 is outside tol


def test_sync_flush_drains_without_future_scan():
    s = SoftSync(tolerance_s=0.05)
    s.push_rgbd(2.0, "rgb", "depth")
    s.push_scan(2.01, _scan(2.01))
    assert s.try_pop() is None       # held pending a later scan
    out = s.try_pop(flush=True)      # end-of-stream drain
    assert out is not None
    assert out.scan[0, 0] == pytest.approx(2.01)


def test_sync_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        SoftSync(tolerance_s=0.0)


# --------------------------------------------------------------------------
# lidar_synth
# --------------------------------------------------------------------------

def _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def test_synth_constant_plane_forward_distance():
    depth = np.full((480, 640), 2.0)
    scan = synthesize_2d_scan(depth, _K(), noise_sigma=0.0)
    assert scan.shape[1] == 2
    assert scan.shape[0] > 0
    # forward coordinate (= camera Z) should equal the plane depth.
    assert np.allclose(scan[:, 0], 2.0, atol=1e-6)
    # the centre column (u == cx) maps to left == 0.
    assert np.min(np.abs(scan[:, 1])) == pytest.approx(0.0, abs=1e-6)


def test_synth_empty_row_returns_empty():
    depth = np.zeros((480, 640))
    scan = synthesize_2d_scan(depth, _K(), noise_sigma=0.0)
    assert scan.shape == (0, 2)


def test_synth_range_gate():
    depth = np.full((480, 640), 50.0)  # beyond range_max
    scan = synthesize_2d_scan(depth, _K(), range_max=10.0, noise_sigma=0.0)
    assert scan.shape == (0, 2)


def test_synth_num_beams_cap():
    depth = np.full((480, 640), 2.0)
    scan = synthesize_2d_scan(depth, _K(), num_beams=50, noise_sigma=0.0)
    assert scan.shape[0] <= 50


def test_synth_noise_is_reproducible():
    depth = np.full((480, 640), 2.0)
    a = synthesize_2d_scan(depth, _K(), noise_sigma=0.02,
                           rng=np.random.default_rng(0))
    b = synthesize_2d_scan(depth, _K(), noise_sigma=0.02,
                           rng=np.random.default_rng(0))
    assert np.allclose(a, b)


# --------------------------------------------------------------------------
# config + signature container
# --------------------------------------------------------------------------

def test_mode_values():
    assert {m.value for m in Mode} == {"orb", "lidar", "vlmain", "lvmain"}


def test_config_defaults():
    cfg = FusionConfig(mode=Mode.VLMAIN, dataset_path="d", output_dir="o")
    assert cfg.stm_size == 30
    assert cfg.wm_cap == 200
    assert cfg.ltm_cap == 1000
    assert cfg.sync_tolerance_s == pytest.approx(0.050)
    assert cfg.lidar_frontend == "scan_to_submap"


def test_signature_has_scan_flag():
    sig = Signature(id=0, timestamp=0.0, pose=Pose2(0, 0, 0))
    assert sig.has_scan is False
    sig.scan = np.zeros((3, 2))
    assert sig.has_scan is True


# --------------------------------------------------------------------------
# dataset loader (requires the TUM sequence on disk)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not DATASET.exists(), reason="TUM fr1_room not present")
def test_dataset_yields_synced_frames():
    from slam_core.fusion.dataset import FusionDataset

    ds = FusionDataset(DATASET, num_beams=180)
    assert len(ds) > 0

    frames = list(ds.iter_frames(max_frames=5))
    assert len(frames) == 5
    for fr in frames:
        assert fr.rgb.ndim == 3
        assert fr.depth.ndim == 2
        assert fr.scan is not None and fr.scan.shape[1] == 2
        # depth (scan) timestamp differs from rgb timestamp but stays within
        # the soft-sync tolerance the associator enforced.
        assert fr.scan_t != fr.rgb_t
        assert abs(fr.scan_t - fr.rgb_t) <= 0.05


@pytest.mark.skipif(not DATASET.exists(), reason="TUM fr1_room not present")
def test_dataset_softsync_end_to_end():
    from slam_core.fusion.dataset import FusionDataset

    ds = FusionDataset(DATASET, num_beams=180)
    sync = SoftSync(tolerance_s=0.05)
    finalized = 0
    matched = 0
    for fr in ds.iter_frames(max_frames=10):
        sync.push_rgbd(fr.rgb_t, fr.rgb, fr.depth)
        sync.push_scan(fr.scan_t, fr.scan)
        out = sync.try_pop()
        while out is not None:
            finalized += 1
            matched += int(out.scan is not None)
            out = sync.try_pop()
    out = sync.try_pop(flush=True)
    while out is not None:
        finalized += 1
        matched += int(out.scan is not None)
        out = sync.try_pop(flush=True)

    assert finalized == 10
    assert matched == 10  # synthetic scans land inside the window for every frame
