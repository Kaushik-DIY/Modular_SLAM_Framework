"""
=============================================================================
visual_slam/orbslam/slam/geometry_matchers.py

pySLAM-aligned projection matcher subset.

Reference:
- pySLAM: pyslam/slam/geometry_matchers.py

Implemented now:
- ProjectionMatcher.search_frame_by_projection
- ProjectionMatcher.search_keyframe_by_projection
- ProjectionMatcher.search_map_by_projection
- ProjectionMatcher.search_local_frames_by_projection
- ProjectionMatcher.search_all_map_by_projection
- ProjectionMatcher.search_and_fuse

Deferred:
- Sim3 loop-correction search
- Epipolar triangulation matcher
=============================================================================
"""

from __future__ import annotations

import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import (
    Frame,
    are_map_points_visible_in_frame,
    ensure_frame_feature_arrays,
)
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.utilities.geom_2views import computeF12, check_dist_epipolar_line


kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation


class ProjectionMatcher:
    @staticmethod
    def search_frame_by_projection(*args, **kwargs):
        return _search_frame_by_projection(*args, **kwargs)

    @staticmethod
    def search_keyframe_by_projection(*args, **kwargs):
        return _search_keyframe_by_projection(*args, **kwargs)

    @staticmethod
    def search_map_by_projection(*args, **kwargs):
        return _search_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_local_frames_by_projection(*args, **kwargs):
        return _search_local_frames_by_projection(*args, **kwargs)

    @staticmethod
    def search_all_map_by_projection(*args, **kwargs):
        return _search_all_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_more_map_points_by_projection(*args, **kwargs):
        raise NotImplementedError("Sim3 projection search is ported with loop closing.")

    @staticmethod
    def search_and_fuse(*args, **kwargs):
        return _search_and_fuse(*args, **kwargs)

    @staticmethod
    def search_and_fuse_for_loop_correction(*args, **kwargs):
        raise NotImplementedError("Loop-correction fusion is ported with loop closing.")

    @staticmethod
    def search_by_sim3(*args, **kwargs):
        raise NotImplementedError("Sim3 matching is ported with loop closing.")


class EpipolarMatcher:
    @staticmethod
    def search_frame_for_triangulation(
        f1,
        f2,
        idxs1=None,
        idxs2=None,
        max_descriptor_distance=None,
        is_monocular=True,
    ):
        """
        pySLAM-aligned epipolar matcher used by local mapping triangulation.

        It returns only matches where both keypoints currently have no assigned
        map point, which is the intended local-mapping triangulation input.
        """
        ensure_frame_feature_arrays(f1)
        ensure_frame_feature_arrays(f2)

        max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

        candidate1 = np.array(
            [i for i, p in enumerate(f1.points) if p is None],
            dtype=np.int32,
        )
        candidate2 = np.array(
            [i for i, p in enumerate(f2.points) if p is None],
            dtype=np.int32,
        )

        if idxs1 is not None and idxs2 is not None:
            idxs1 = np.asarray(idxs1, dtype=np.int32).reshape(-1)
            idxs2 = np.asarray(idxs2, dtype=np.int32).reshape(-1)

            if len(idxs1) != len(idxs2):
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            pair_candidates = [
                (int(i1), int(i2))
                for i1, i2 in zip(idxs1, idxs2)
                if i1 in set(candidate1) and i2 in set(candidate2)
            ]
        else:
            if len(candidate1) == 0 or len(candidate2) == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            matches = FeatureTrackerShared.feature_matcher.match(
                f1.img,
                f2.img,
                f1.des[candidate1],
                f2.des[candidate2],
                kps1=[f1.kps[i] for i in candidate1],
                kps2=[f2.kps[i] for i in candidate2],
            )

            if matches.idxs1 is None or matches.idxs2 is None:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

            pair_candidates = [
                (int(candidate1[i1]), int(candidate2[i2]))
                for i1, i2 in zip(matches.idxs1, matches.idxs2)
            ]

        if len(pair_candidates) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

        F12, _ = computeF12(f1, f2)

        out1 = []
        out2 = []

        level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2

        for i1, i2 in pair_candidates:
            if i1 < 0 or i1 >= len(f1.des) or i2 < 0 or i2 >= len(f2.des):
                continue

            d = FeatureTrackerShared.descriptor_distance(f1.des[i1], f2.des[i2])
            if d > max_descriptor_distance:
                continue

            octave2 = max(0, min(int(f2.octaves[i2]), len(level_sigmas2) - 1))
            sigma2 = float(level_sigmas2[octave2])

            if not check_dist_epipolar_line(f1.kpsu[i1].pt, f2.kpsu[i2].pt, F12, sigma2):
                continue

            out1.append(i1)
            out2.append(i2)

        if len(out1) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

        idxs1_out = np.asarray(out1, dtype=np.int32)
        idxs2_out = np.asarray(out2, dtype=np.int32)

        if FeatureTrackerShared.oriented_features:
            valid = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs1_out,
                idxs2_out,
                f1.angles,
                f2.angles,
            )
            idxs1_out = idxs1_out[valid]
            idxs2_out = idxs2_out[valid]

        return idxs1_out, idxs2_out, len(idxs1_out)


def _max_descriptor_distance(value):
    return Parameters.kMaxDescriptorDistance if value is None else value


def _valid_rotation_filter(idxs_ref, idxs_cur, ref_angles, cur_angles):
    if not (kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features):
        return np.asarray(idxs_ref, dtype=np.int32), np.asarray(idxs_cur, dtype=np.int32)

    if len(idxs_ref) == 0:
        return np.asarray(idxs_ref, dtype=np.int32), np.asarray(idxs_cur, dtype=np.int32)

    rot_histo = RotationHistogram(Parameters.kRotationHistogramLength if hasattr(Parameters, "kRotationHistogramLength") else 12)

    for match_idx, (idx_ref, idx_cur) in enumerate(zip(idxs_ref, idxs_cur)):
        rot = float(ref_angles[idx_ref]) - float(cur_angles[idx_cur])
        rot_histo.push(rot, match_idx)

    valid = rot_histo.get_valid_idxs()

    return np.asarray(idxs_ref, dtype=np.int32)[valid], np.asarray(idxs_cur, dtype=np.int32)[valid]


def _search_frame_by_projection(
    f_ref: Frame,
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    is_monocular=True,
    already_matched_ref_idxs=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

    ensure_frame_feature_arrays(f_ref)
    ensure_frame_feature_arrays(f_cur)

    matched_ref_idxs = np.array(
        [i for i, p in enumerate(f_ref.points) if p is not None and not f_ref.outliers[i]],
        dtype=np.int32,
    )

    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    if len(matched_ref_idxs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    matched_ref_points = [f_ref.points[i] for i in matched_ref_idxs]

    projs, depths = f_cur.project_map_points(matched_ref_points, f_cur.camera.is_stereo())
    is_visible = f_cur.are_in_image(projs[:, :2], depths)

    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]
    kp_ref_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[kp_ref_octaves]
    radiuses = max_reproj_distance * kp_ref_scale_factors

    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    idxs_ref = []
    idxs_cur = []

    cur_des = f_cur.des
    cur_points = f_cur.points
    cur_octaves = f_cur.octaves

    do_stereo_check = f_cur.uRs is not None and len(f_cur.uRs) > 0

    for j, (ref_idx, p_ref) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not is_visible[j]:
            continue

        kp_ref_octave = f_ref.octaves[ref_idx]
        best_dist = float("inf")
        best_k_idx = -1

        candidate_idxs = kd_cur_idxs[j]

        for h, kd_idx in enumerate(candidate_idxs):
            p_cur = cur_points[kd_idx]
            if p_cur is not None and p_cur.num_observations() > 0:
                continue

            kp_cur_octave = cur_octaves[kd_idx]
            if kp_cur_octave < (kp_ref_octave - 1) or kp_cur_octave > (kp_ref_octave + 1):
                continue

            if do_stereo_check and f_cur.uRs[kd_idx] >= 0:
                err_ur = abs(projs[j, 2] - f_cur.uRs[kd_idx])
                scale = FeatureTrackerShared.feature_manager.scale_factors[kp_cur_octave]
                if err_ur >= max_reproj_distance * scale:
                    continue

            descriptor_dist = p_ref.min_des_distance(cur_des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_k_idx = kd_idx

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if p_ref.add_frame_view(f_cur, best_k_idx):
                idxs_ref.append(int(ref_idx))
                idxs_cur.append(int(best_k_idx))

    idxs_ref, idxs_cur = _valid_rotation_filter(idxs_ref, idxs_cur, f_ref.angles, f_cur.angles)

    return idxs_ref, idxs_cur, len(idxs_cur)


def _search_keyframe_by_projection(
    kf_ref: KeyFrame,
    f_cur: Frame,
    max_reproj_distance,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    already_matched_ref_idxs=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

    assert kf_ref.is_keyframe, "[search_keyframe_by_projection] kf_ref must be a KeyFrame"

    ensure_frame_feature_arrays(kf_ref)
    ensure_frame_feature_arrays(f_cur)

    ref_mps = kf_ref.get_matched_points()
    if len(ref_mps) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    matched_ref_idxs = np.array(
        [i for i, p in enumerate(ref_mps) if p is not None and not p.is_bad()],
        dtype=np.int32,
    )

    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    matched_ref_points = [ref_mps[i] for i in matched_ref_idxs]
    if len(matched_ref_points) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    visible, projs, depths, dists = f_cur.are_visible(matched_ref_points, f_cur.camera.is_stereo())
    predicted_levels = MapPoint.predict_detection_levels(matched_ref_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    idxs_ref = []
    idxs_cur = []

    for j, (ref_idx, mp) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not visible[j]:
            continue

        predicted_level = predicted_levels[j]
        best_dist = float("inf")
        best_dist2 = float("inf")
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for idx2 in kd_cur_idxs[j]:
            if f_cur.points[idx2] is not None:
                continue

            kp_level = f_cur.octaves[idx2]
            if kp_level < predicted_level - 1 or kp_level > predicted_level + 1:
                continue

            descriptor_dist = mp.min_des_distance(f_cur.des[idx2])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = idx2
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if best_level == best_level2 and best_dist > best_dist2 * ratio_test:
                continue
            if mp.add_frame_view(f_cur, best_k_idx):
                idxs_ref.append(int(ref_idx))
                idxs_cur.append(int(best_k_idx))

    idxs_ref, idxs_cur = _valid_rotation_filter(idxs_ref, idxs_cur, kf_ref.angles, f_cur.angles)

    return idxs_ref, idxs_cur, len(idxs_cur)


def _search_map_by_projection(
    points: list[MapPoint],
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    far_points_threshold=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)

    if len(points) == 0:
        return 0, []

    ensure_frame_feature_arrays(f_cur)

    visibility_flags, projs, depths, dists = f_cur.are_visible(points, f_cur.camera.is_stereo())
    predicted_levels = MapPoint.predict_detection_levels(points, dists)

    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors

    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    if far_points_threshold is not None:
        visibility_flags = np.logical_and(visibility_flags, depths < far_points_threshold)

    idxs_and_pts = [
        (i, p)
        for i, p in enumerate(points)
        if visibility_flags[i]
        and p is not None
        and not p.is_bad()
        and p.last_frame_id_seen != f_cur.id
    ]

    found_pts_count = 0
    found_pts_fidxs = []

    for i, p in idxs_and_pts:
        p.increase_visible()
        predicted_level = predicted_levels[i]

        best_dist = float("inf")
        best_dist2 = float("inf")
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for kd_idx in kd_cur_idxs[i]:
            p_f = f_cur.points[kd_idx]
            if p_f is not None and p_f.num_observations() > 0:
                continue

            kp_level = f_cur.octaves[kd_idx]
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                continue

            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_k_idx > -1 and best_dist < max_descriptor_distance:
            if best_level == best_level2 and best_dist > best_dist2 * ratio_test:
                continue
            if p.add_frame_view(f_cur, best_k_idx):
                p.increase_found()
                found_pts_count += 1
                found_pts_fidxs.append(best_k_idx)

    return found_pts_count, found_pts_fidxs


def _search_local_frames_by_projection(
    map,
    f_cur,
    local_window_size=Parameters.kLocalBAWindowSize,
    max_descriptor_distance=None,
):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)
    frames = map.get_last_keyframes(local_window_size)
    frame_valid_points = set([p for f in frames for p in f.get_points() if p is not None])
    return _search_map_by_projection(
        list(frame_valid_points),
        f_cur,
        max_descriptor_distance=max_descriptor_distance,
    )


def _search_all_map_by_projection(map, f_cur, max_descriptor_distance=None):
    max_descriptor_distance = _max_descriptor_distance(max_descriptor_distance)
    return _search_map_by_projection(
        map.get_points().to_list() if hasattr(map.get_points(), "to_list") else list(map.get_points()),
        f_cur,
        max_descriptor_distance=max_descriptor_distance,
    )


def _search_and_fuse(
    points: list[MapPoint],
    keyframe: KeyFrame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
):
    max_descriptor_distance = 0.5 * _max_descriptor_distance(max_descriptor_distance)

    if len(points) == 0:
        return 0

    ensure_frame_feature_arrays(keyframe)

    good_points = [p for p in points if p is not None and not p.is_bad_or_is_in_keyframe(keyframe)]

    if len(good_points) == 0:
        return 0

    visible, projs, depths, dists = keyframe.are_visible(good_points, keyframe.camera.is_stereo())

    predicted_levels = MapPoint.predict_detection_levels(good_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_idxs = keyframe.kd.query_ball_point(projs[:, :2], radiuses)

    fused_pts_count = 0
    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

    for j, point in enumerate(good_points):
        if not visible[j]:
            continue

        predicted_level = predicted_levels[j]
        best_dist = float("inf")
        best_kd_idx = -1

        for kd_idx in kd_idxs[j]:
            kp_level = keyframe.octaves[kd_idx]
            if kp_level < predicted_level - 1 or kp_level > predicted_level:
                continue

            err = projs[j, :2] - np.array(keyframe.kpsu[kd_idx].pt, dtype=np.float64)
            chi2 = float(np.dot(err, err) * inv_level_sigmas2[kp_level])

            if chi2 > Parameters.kChi2Mono:
                continue

            descriptor_dist = point.min_des_distance(keyframe.des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_kd_idx = kd_idx

        if best_kd_idx > -1 and best_dist < max_descriptor_distance:
            existing = keyframe.get_point_match(best_kd_idx)

            if existing is not None:
                if existing.num_observations() > point.num_observations():
                    point.replace_with(existing)
                else:
                    existing.replace_with(point)
                    point.add_observation(keyframe, best_kd_idx)
            else:
                point.add_observation(keyframe, best_kd_idx)

            point.update_info()
            fused_pts_count += 1

    return fused_pts_count
