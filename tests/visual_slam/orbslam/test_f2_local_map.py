"""F2 isolation parity: C++ cppcore.build_local_map vs the Python tracking logic.

Builds a synthetic multi-keyframe scene (shared MapPoints -> covisibility +
spanning tree via update_connections), then a current frame whose matched points
vote some KFs, and checks the C++ expanding local-map build matches a faithful
Python reference of tracking._build_local_keyframes_from_votes +
_collect_local_points_from_keyframes (same local-KF list ORDER and local-point
list ORDER). Uses raw cpp_slam_core objects (no live KeyFrame needed).
"""
import sys, os
sys.path.insert(0, "/home/kaushik/slam_ws")
os.chdir("/home/kaushik/slam_ws")

import cv2
import numpy as np
import pytest

try:
    import cpp_slam_core
    _HAVE = hasattr(cpp_slam_core, "build_local_map")
except Exception:
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="cpp_slam_core.build_local_map unavailable")


def _make_kf(kid, n_kps):
    kf = cpp_slam_core.KeyFrame(kid=kid, frame_id=kid)
    kps = [cv2.KeyPoint(x=float(i % 50) * 12.0, y=float(i // 50) * 12.0, size=1.0, octave=i % 4)
           for i in range(n_kps)]
    des = np.random.randint(0, 255, (n_kps, 32), dtype=np.uint8)
    kf.init_feature_arrays(kps, des, None, None, n_kps)
    return kf


def _build_scene(n_kfs=6, n_shared=15):
    """Chain of KFs; consecutive pairs share n_shared MapPoints -> covisibility."""
    n_kps = n_shared * n_kfs
    kfs = [_make_kf(i, n_kps) for i in range(n_kfs)]
    shared = {}  # (i) -> list of MapPoints shared by kf[i],kf[i+1]
    for i in range(n_kfs - 1):
        mps = []
        for j in range(n_shared):
            idx = i * n_shared + j
            mp = cpp_slam_core.MapPoint([float(i) + 0.01 * j, 0.0, 3.0])
            mp.add_observation(kfs[i], idx)
            mp.add_observation(kfs[i + 1], idx)
            mps.append((mp, idx))
        shared[i] = mps
    for kf in kfs:
        kf.update_connections()
    return kfs, shared


def _py_build_local_map(f_cur, num_best, max_kfs):
    """Faithful Python mirror of tracking._build_local_keyframes_from_votes +
    _collect_local_points_from_keyframes (using get_points, like the C++)."""
    def bad(o):
        try: return o.is_bad()
        except Exception: return True
    # votes
    votes = {}
    for p in f_cur.get_matched_good_points():
        if p is None or bad(p):
            continue
        for kf, _ in p.observations():
            if kf is None or bad(kf):
                continue
            votes[kf] = votes.get(kf, 0) + 1
    counts = {kf: c for kf, c in votes.items() if not bad(kf)}
    local = list(counts.keys())

    def add(kf):
        if kf is None or bad(kf) or kf in counts:
            return False
        counts[kf] = 1
        local.append(kf)
        return True

    for kf in local:                       # transitive: iterates the growing list
        if len(local) >= max_kfs:
            break
        for n in kf.get_best_covisible_keyframes(num_best):
            if add(n):
                break
        for c in kf.get_children():
            if add(c):
                break
        add(kf.get_parent())
    local_kfs = [kf for kf, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_kfs]]
    # collect points (per-call dedup by identity)
    pts, seen = [], set()
    for kf in local_kfs:
        for p in kf.get_points():
            if p is None or bad(p):
                continue
            if p.get_replacement() is not None:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            pts.append(p)
    return local_kfs, pts


def test_build_local_map_parity():
    np.random.seed(0)
    kfs, shared = _build_scene(n_kfs=6, n_shared=15)

    # current frame: votes KFs 2 and 3 (set_point_match -> matched, NOT an observer,
    # matching the real pipeline where f_cur is a Frame, not a KF observer).
    f_cur = _make_kf(100, n_kps=15 * 6)
    for mp, idx in shared[2]:       # points shared by kf2,kf3
        f_cur.set_point_match(mp, idx)

    num_best, max_kfs, frame_id = 3, 80, 100
    cpp_kfs, cpp_pts = cpp_slam_core.build_local_map(f_cur, num_best, max_kfs, frame_id)
    py_kfs, py_pts = _py_build_local_map(f_cur, num_best, max_kfs)

    # local keyframes: same set + same order (by kid)
    assert [k.kid for k in cpp_kfs] == [k.kid for k in py_kfs], \
        f"local KF mismatch: cpp={[k.kid for k in cpp_kfs]} py={[k.kid for k in py_kfs]}"
    assert len(cpp_kfs) >= 3, "expansion should pull in neighbors/children/parent"

    # local points: same objects in the same order
    assert [id(p) for p in cpp_pts] == [id(p) for p in py_pts], \
        f"local point mismatch: cpp_n={len(cpp_pts)} py_n={len(py_pts)}"
    assert len(cpp_pts) > 0


def test_build_local_map_empty_frame():
    """No matched points -> empty local map (no crash)."""
    f_cur = _make_kf(200, n_kps=30)
    cpp_kfs, cpp_pts = cpp_slam_core.build_local_map(f_cur, 3, 80, 200)
    assert list(cpp_kfs) == [] and list(cpp_pts) == []


def test_mark_current_frame_matched_points_seen():
    """Native helper matches tracking._mark_current_frame_matched_points_seen effects."""
    f_cur = _make_kf(300, n_kps=4)
    good = cpp_slam_core.MapPoint([0.0, 0.0, 3.0])
    bad = cpp_slam_core.MapPoint([1.0, 0.0, 3.0])
    bad.set_bad()
    f_cur.set_point_match(good, 0)
    f_cur.set_point_match(bad, 1)

    visible_before = good.num_times_visible
    marked = cpp_slam_core.mark_current_frame_matched_points_seen(f_cur)

    assert marked == 1
    assert good.num_times_visible == visible_before + 1
    assert good.last_frame_id_seen == f_cur.id
    assert bad.last_frame_id_seen != f_cur.id


def test_build_mark_search_local_map_empty_frame():
    """Combined F3 helper preserves the empty-local-map case and exports timings."""
    f_cur = _make_kf(400, n_kps=8)
    scale_factors = np.ones(8, dtype=np.float32)
    result = cpp_slam_core.build_mark_search_local_map(
        f_cur,
        f_cur,
        3,
        80,
        f_cur.id,
        scale_factors,
        10.0,
        100.0,
        0.8,
        0.5,
        0.0,
        float("inf"),
        float(np.log(1.2)),
        8,
    )

    local_keyframes, local_points, found_count, found_fidxs, build_sec, mark_sec, search_sec = result
    assert list(local_keyframes) == []
    assert list(local_points) == []
    assert found_count == 0
    assert list(found_fidxs) == []
    assert build_sec >= 0.0
    assert mark_sec >= 0.0
    assert search_sec >= 0.0


if __name__ == "__main__":
    test_build_local_map_parity()
    test_build_local_map_empty_frame()
    test_mark_current_frame_matched_points_seen()
    test_build_mark_search_local_map_empty_frame()
    print("F2_LOCAL_MAP_PARITY_OK")
