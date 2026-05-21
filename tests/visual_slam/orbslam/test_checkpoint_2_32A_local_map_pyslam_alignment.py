import cv2
import g2o
import numpy as np

from tools.run_fr1_room_full_evaluation import write_csv
from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    Map,
    MapPoint,
    PinholeCamera,
    ProjectionMatcher,
    SensorType,
)
from visual_slam.orbslam.run_rgbd_slam import LOCAL_MAP_PROFILE_COLUMNS
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.tracking import Tracking


def setup_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def make_camera():
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )


def make_frame(frame_id=0, n=20, camera=None):
    camera = camera or make_camera()
    frame = Frame(
        camera=camera,
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(np.eye(4)),
        id=frame_id,
        timestamp=float(frame_id),
    )
    frame.kps = [cv2.KeyPoint(320.0 + i, 240.0, 20.0, 0.0, 1.0, 0) for i in range(n)]
    frame.kpsu = list(frame.kps)
    frame.des = np.asarray([np.full(32, i % 255, dtype=np.uint8) for i in range(n)], dtype=np.uint8)
    frame.depths = np.full(n, 2.0, dtype=np.float32)
    frame.uRs = np.full(n, 300.0, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(n, dtype=np.int32)
    frame.angles = np.zeros(n, dtype=np.float32)
    frame.sizes = np.full(n, 20.0, dtype=np.float32)
    frame.points = [None] * n
    frame.outliers = np.zeros(n, dtype=bool)
    frame.idxs = np.arange(n, dtype=np.int32)
    frame.kd = None
    frame.ensure_contiguous_arrays()
    return frame


def make_keyframe(kid, n=30):
    return KeyFrame(make_frame(frame_id=kid, n=n), kid=kid)


class FakeSlam:
    def __init__(self):
        self.map = Map()
        self.camera = make_camera()
        self.sensor_type = SensorType.RGBD
        self.local_mapping = None
        self.runtime_profiler = None


def make_tracking():
    tracking = object.__new__(Tracking)
    tracking.slam = FakeSlam()
    tracking.kf_ref = None
    tracking.kf_last = None
    tracking.local_keyframes = []
    tracking.local_points = []
    tracking.profile_local_map = False
    return tracking


def observe(point, keyframe, idx):
    point.add_observation(keyframe, idx)
    return point


def add_current_match(frame, idx, point):
    frame.points[idx] = point
    frame.outliers[idx] = False


def test_local_keyframe_voting_uses_current_frame_matched_points():
    tracking = make_tracking()
    kf_a = make_keyframe(1)
    kf_b = make_keyframe(2)
    current = make_frame(frame_id=10, n=4)

    p0 = observe(MapPoint(np.array([0.0, 0.0, 2.0])), kf_a, 0)
    p1 = observe(MapPoint(np.array([0.1, 0.0, 2.0])), kf_b, 1)
    p2 = observe(MapPoint(np.array([0.2, 0.0, 2.0])), kf_b, 2)
    add_current_match(current, 0, p0)
    add_current_match(current, 1, p1)
    add_current_match(current, 2, p2)

    votes = tracking._collect_local_keyframe_votes_from_current_frame(current)

    assert votes[kf_a] == 1
    assert votes[kf_b] == 2


def test_local_keyframe_voting_ignores_bad_points_and_bad_keyframes():
    tracking = make_tracking()
    good_kf = make_keyframe(1)
    bad_kf = make_keyframe(2)
    bad_kf._is_bad = True
    current = make_frame(frame_id=11, n=3)

    good = observe(MapPoint(np.array([0.0, 0.0, 2.0])), good_kf, 0)
    bad_point = observe(MapPoint(np.array([0.1, 0.0, 2.0])), good_kf, 1)
    bad_point.set_bad(map_no_lock=True)
    point_seen_by_bad_kf = observe(MapPoint(np.array([0.2, 0.0, 2.0])), bad_kf, 2)
    add_current_match(current, 0, good)
    add_current_match(current, 1, bad_point)
    add_current_match(current, 2, point_seen_by_bad_kf)

    votes = tracking._collect_local_keyframe_votes_from_current_frame(current)

    assert votes == {good_kf: 1}


def test_reference_keyframe_selected_by_max_vote_before_local_points():
    tracking = make_tracking()
    kf_a = make_keyframe(1)
    kf_b = make_keyframe(2)
    current = make_frame(frame_id=12, n=3)

    add_current_match(current, 0, observe(MapPoint(np.array([0.0, 0.0, 2.0])), kf_a, 0))
    add_current_match(current, 1, observe(MapPoint(np.array([0.1, 0.0, 2.0])), kf_b, 1))
    add_current_match(current, 2, observe(MapPoint(np.array([0.2, 0.0, 2.0])), kf_b, 2))
    tracking.f_cur = current

    tracking.update_local_map()

    assert tracking.kf_ref is kf_b
    assert current.kf_ref is kf_b
    assert tracking.local_points


def test_fallback_to_existing_reference_when_no_votes():
    tracking = make_tracking()
    kf_ref = make_keyframe(5)
    tracking.kf_ref = kf_ref
    tracking.f_cur = make_frame(frame_id=13, n=2)

    tracking.update_local_map()

    assert tracking.kf_ref is kf_ref
    assert tracking.f_cur.kf_ref is kf_ref


def test_local_keyframes_include_voted_keyframes():
    tracking = make_tracking()
    kf_a = make_keyframe(1)
    kf_b = make_keyframe(2)
    local = tracking._build_local_keyframes_from_votes({kf_a: 2, kf_b: 1}, make_frame(frame_id=14))

    assert kf_a in local
    assert kf_b in local


def test_local_keyframes_do_not_start_from_all_reference_covisibles():
    tracking = make_tracking()
    voted = make_keyframe(1)
    covisibles = [make_keyframe(i) for i in range(2, 8)]
    for i, kf in enumerate(covisibles):
        voted.add_connection(kf, 100 - i)

    local = tracking._build_local_keyframes_from_votes({voted: 5}, make_frame(frame_id=15), num_best=2)

    assert voted in local
    assert len([kf for kf in covisibles if kf in local]) == 1


def test_num_best_covisibility_keyframes_is_honored():
    tracking = make_tracking()
    voted = make_keyframe(1)
    best = make_keyframe(2)
    second = make_keyframe(3)
    voted.add_connection(best, 100)
    voted.add_connection(second, 90)

    local = tracking._build_local_keyframes_from_votes({voted: 3}, make_frame(frame_id=16), num_best=1)

    assert best in local
    assert second not in local


def test_get_best_covisible_keyframes_orders_by_weight():
    kf = make_keyframe(1)
    low = make_keyframe(2)
    high = make_keyframe(3)
    mid = make_keyframe(4)
    kf.add_connection(low, 2)
    kf.add_connection(high, 9)
    kf.add_connection(mid, 5)

    assert kf.get_best_covisible_keyframes(2) == [high, mid]


def test_parent_and_child_expansion_are_bounded():
    tracking = make_tracking()
    voted = make_keyframe(1)
    parent = make_keyframe(2)
    child_a = make_keyframe(3)
    child_b = make_keyframe(4)
    voted.set_parent(parent)
    voted.add_child(child_a)
    voted.add_child(child_b)

    local = tracking._build_local_keyframes_from_votes({voted: 4}, make_frame(frame_id=17), num_best=0)

    assert parent in local
    assert len([child for child in (child_a, child_b) if child in local]) == 1


def test_bad_keyframes_are_not_added_during_expansion():
    tracking = make_tracking()
    voted = make_keyframe(1)
    bad_neighbor = make_keyframe(2)
    good_neighbor = make_keyframe(3)
    bad_neighbor._is_bad = True
    voted.add_connection(bad_neighbor, 100)
    voted.add_connection(good_neighbor, 90)

    local = tracking._build_local_keyframes_from_votes({voted: 2}, make_frame(frame_id=18), num_best=2)

    assert bad_neighbor not in local
    assert good_neighbor in local


def test_local_points_are_unique_by_frame_marker():
    tracking = make_tracking()
    kf_a = make_keyframe(1)
    kf_b = make_keyframe(2)
    shared = MapPoint(np.array([0.0, 0.0, 2.0]))
    observe(shared, kf_a, 0)
    observe(shared, kf_b, 0)

    points = tracking._collect_local_points_from_keyframes([kf_a, kf_b], make_frame(frame_id=19))

    assert points == [shared]
    assert shared.last_track_reference_frame_id == 19


def test_local_point_marker_is_frame_specific():
    tracking = make_tracking()
    kf = make_keyframe(1)
    point = observe(MapPoint(np.array([0.0, 0.0, 2.0])), kf, 0)

    points_a = tracking._collect_local_points_from_keyframes([kf], make_frame(frame_id=20))
    points_b = tracking._collect_local_points_from_keyframes([kf], make_frame(frame_id=21))

    assert points_a == [point]
    assert points_b == [point]
    assert point.last_track_reference_frame_id == 21


def test_bad_points_are_skipped_when_collecting_local_points():
    tracking = make_tracking()
    kf = make_keyframe(1)
    good = observe(MapPoint(np.array([0.0, 0.0, 2.0])), kf, 0)
    bad = observe(MapPoint(np.array([0.1, 0.0, 2.0])), kf, 1)
    bad.set_bad(map_no_lock=True)

    points = tracking._collect_local_points_from_keyframes([kf], make_frame(frame_id=22))

    assert points == [good]


def test_already_matched_points_are_marked_seen_before_projection():
    tracking = make_tracking()
    current = make_frame(frame_id=23, n=2)
    point = MapPoint(np.array([0.0, 0.0, 2.0]))
    add_current_match(current, 0, point)

    marked = tracking._mark_current_frame_matched_points_seen(current)

    assert marked == 1
    assert point.last_frame_id_seen == current.id


def test_search_map_by_projection_skips_last_frame_seen_points_before_projection():
    setup_tracker()
    camera = make_camera()
    point = MapPoint(np.array([0.0, 0.0, 2.0]))
    point.set_descriptor(np.zeros(32, dtype=np.uint8))
    frame = make_frame(frame_id=24, n=1, camera=camera)
    frame.des[0] = np.zeros(32, dtype=np.uint8)
    frame.kd = None
    point.last_frame_id_seen = frame.id
    diagnostics = {}

    count, frame_idxs = ProjectionMatcher.search_map_by_projection(
        [point],
        frame,
        max_reproj_distance=10,
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )

    assert count == 0
    assert frame_idxs == []
    assert diagnostics["visible_projected_points"] == 0


def test_projection_diagnostics_count_rejected_already_seen_points():
    setup_tracker()
    camera = make_camera()
    point = MapPoint(np.array([0.0, 0.0, 2.0]))
    point.set_descriptor(np.zeros(32, dtype=np.uint8))
    frame = make_frame(frame_id=25, n=1, camera=camera)
    point.last_frame_id_seen = frame.id
    diagnostics = {}

    ProjectionMatcher.search_map_by_projection(
        [point],
        frame,
        max_reproj_distance=10,
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )

    assert diagnostics["input_local_points"] == 1
    assert diagnostics["rejected_already_seen"] == 1
    assert diagnostics["kd_candidate_count"] == 0


def test_local_map_profile_csv_has_required_columns(tmp_path):
    path = tmp_path / "local_map_profile.csv"
    row = {column: 0 for column in LOCAL_MAP_PROFILE_COLUMNS}

    write_csv(path, [row], LOCAL_MAP_PROFILE_COLUMNS)

    header = path.read_text().splitlines()[0].split(",")
    assert header == LOCAL_MAP_PROFILE_COLUMNS
