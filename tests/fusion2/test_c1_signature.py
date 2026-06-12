"""C1 — Signature payloads (copy-in, zero-copy views, lifetime) + LtmStore + memory gate."""
import gc
import math
import os

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")


def _payload(n_kpts=2000, n_scan=909, seed=0):
    rng = np.random.default_rng(seed)
    kpts = rng.uniform(0, 640, (n_kpts, 2)).astype(np.float32)
    des = rng.integers(0, 256, (n_kpts, 32), dtype=np.uint8)
    pts3d = rng.uniform(-5, 5, (n_kpts, 3)).astype(np.float32)
    scan = rng.uniform(0.1, 16.0, (n_scan, 2)).astype(np.float32)
    return kpts, des, pts3d, scan


def _make_sig(i=0, **kw):
    kpts, des, pts3d, scan = _payload(**kw)
    sig = fusion_core.Signature(i, 123.456 + i, kpts=kpts, des=des, pts3d=pts3d, scan_xy=scan)
    return sig, (kpts, des, pts3d, scan)


def test_roundtrip_equality():
    sig, (kpts, des, pts3d, scan) = _make_sig()
    assert sig.id == 0 and abs(sig.stamp - 123.456) < 1e-9
    np.testing.assert_array_equal(np.asarray(sig.kpts), kpts)
    np.testing.assert_array_equal(np.asarray(sig.des), des)
    np.testing.assert_array_equal(np.asarray(sig.pts3d), pts3d)
    np.testing.assert_array_equal(np.asarray(sig.scan_xy), scan)
    assert sig.has_visual and sig.has_scan
    assert sig.num_kpts == 2000 and sig.num_scan_points == 909


def test_copy_in_semantics():
    # Mutating the source numpy array after construction must NOT affect the Signature.
    sig, (kpts, _, _, _) = _make_sig(seed=1)
    before = np.asarray(sig.kpts).copy()
    kpts[:] = -1.0
    np.testing.assert_array_equal(np.asarray(sig.kpts), before)


def test_views_are_zero_copy_with_lifetime():
    sig, _ = _make_sig(seed=2)
    view = sig.des
    assert view.base is not None  # backed by the Signature, not a copy
    ref = np.asarray(view).copy()
    del sig
    gc.collect()
    np.testing.assert_array_equal(np.asarray(view), ref)  # view keeps owner alive


def test_modality_flags_and_empty_payloads():
    scan_only = fusion_core.Signature(7, 1.0, scan_xy=np.zeros((10, 2), np.float32))
    assert scan_only.has_scan and not scan_only.has_visual
    visual_only = fusion_core.Signature(
        8, 2.0, kpts=np.zeros((5, 2), np.float32),
        des=np.zeros((5, 32), np.uint8), pts3d=np.zeros((5, 3), np.float32))
    assert visual_only.has_visual and not visual_only.has_scan


def test_links():
    sig, _ = _make_sig(seed=3)
    sig.add_link(41, fusion_core.LinkType.NEIGHBOR, fusion_core.Pose2(0.1, 0.0, 0.05),
                 trans_info=1e5, rot_info=1e5)
    sig.add_link(7, fusion_core.LinkType.LOOP, fusion_core.Pose2(1.0, -0.5, math.pi / 4))
    links = sig.links
    assert [l.to_id for l in links] == [41, 7]
    assert links[1].type == fusion_core.LinkType.LOOP
    assert abs(links[0].rel_pose.x - 0.1) < 1e-12


def test_ltm_store_roundtrip():
    store = fusion_core.InRamLtmStore()
    sig, _ = _make_sig(i=5, seed=4)
    ref_des = np.asarray(sig.des).copy()
    store.store(sig)
    del sig
    gc.collect()
    assert store.contains(5) and store.size() == 1 and store.ids() == [5]
    loaded = store.load(5)
    np.testing.assert_array_equal(np.asarray(loaded.des), ref_des)
    assert store.load(999) is None
    store.erase(5)
    assert not store.contains(5) and store.size() == 0


def _rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("no VmRSS")


class _PyMapPointLike:
    """Per-point object shaped like visual_slam MapPoint (map_point.py:107-132,
    394-405): separate (3,) position array, separate (32,) descriptor copy, two
    threading.Lock instances, two dicts, and the scalar bookkeeping attributes."""

    def __init__(self, pos, des_row):
        import threading
        self._position = np.array(pos, dtype=np.float64)       # map_point.py:394
        self.des = np.array(des_row, dtype=np.uint8)           # map_point.py:452 (per-point copy)
        self._lock_pos = threading.Lock()                      # map_point.py:113
        self._lock_features = threading.Lock()                 # map_point.py:114
        self._observations = {}                                # map_point.py:118
        self._frame_views = {}                                 # map_point.py:119
        self._is_bad = False
        self._num_observations = 0
        self.num_times_visible = 1
        self.num_times_found = 1
        self.last_frame_id_seen = -1
        self.kf_ref = None
        self.replacement = None


class _PySignatureLikeOrb:
    """Replicates how the Python ORB stack actually stores a keyframe's payload:
    kps + kpsu as DUPLICATE lists of cv2.KeyPoint (frame.py:322-323) and one
    MapPoint-shaped object per point (the real per-point cost), plus the scan."""

    def __init__(self, kpts, des, pts3d, scan):
        import cv2
        self.kps = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in kpts]
        self.kpsu = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in kpts]
        self.points = [_PyMapPointLike(p, d) for p, d in zip(pts3d, des)]
        self.scan = scan.copy()
        self.meta = {"weight": 0, "stamp": 0.0, "links": []}


@pytest.mark.parametrize("n_sigs", [200])
def test_memory_gate_vs_python_objects(n_sigs):
    """C1 gate: C++ Signatures >= 10x smaller than the equivalent Python object graph."""
    n_kpts = 2000
    payloads = [_payload(n_kpts=n_kpts, seed=s) for s in range(8)]

    gc.collect()
    base = _rss_mb()
    cpp = [fusion_core.Signature(i, float(i), kpts=p[0], des=p[1], pts3d=p[2], scan_xy=p[3])
           for i, p in ((i, payloads[i % 8]) for i in range(n_sigs))]
    gc.collect()
    cpp_mb = _rss_mb() - base
    expected_payload_mb = sum(s.payload_bytes for s in cpp) / 1e6
    del cpp
    gc.collect()

    base = _rss_mb()
    pyobjs = [_PySignatureLikeOrb(*payloads[i % 8]) for i in range(n_sigs)]
    gc.collect()
    py_mb = _rss_mb() - base
    del pyobjs
    gc.collect()

    print(f"\n  C++ Signatures: {cpp_mb:.1f} MB ({n_sigs} sigs, payload {expected_payload_mb:.1f} MB)"
          f"\n  Python objects: {py_mb:.1f} MB"
          f"\n  ratio: {py_mb / max(cpp_mb, 1e-9):.1f}x")
    assert cpp_mb < expected_payload_mb * 1.5 + 20, "C++ storage should be ~= raw payload"
    assert py_mb / max(cpp_mb, 1e-9) >= 10.0, (
        f"memory gate failed: only {py_mb / max(cpp_mb, 1e-9):.1f}x")
