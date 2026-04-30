# pySLAM to visual_slam/orbslam module mapping

## Core SLAM
third_party/pyslam_reference/pyslam/slam/slam.py
-> visual_slam/orbslam/slam/slam.py

third_party/pyslam_reference/pyslam/slam/slam_commons.py
-> visual_slam/orbslam/slam/slam_commons.py

third_party/pyslam_reference/pyslam/slam/camera.py
-> visual_slam/orbslam/slam/camera.py

third_party/pyslam_reference/pyslam/slam/camera_pose.py
-> visual_slam/orbslam/slam/camera_pose.py

third_party/pyslam_reference/pyslam/slam/frame.py
-> visual_slam/orbslam/slam/frame.py

third_party/pyslam_reference/pyslam/slam/keyframe.py
-> visual_slam/orbslam/slam/keyframe.py

third_party/pyslam_reference/pyslam/slam/keyframe_data.py
-> visual_slam/orbslam/slam/keyframe_data.py

third_party/pyslam_reference/pyslam/slam/map_point.py
-> visual_slam/orbslam/slam/map_point.py

third_party/pyslam_reference/pyslam/slam/map.py
-> visual_slam/orbslam/slam/map.py

third_party/pyslam_reference/pyslam/slam/feature_tracker_shared.py
-> visual_slam/orbslam/slam/feature_tracker_shared.py

third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py
-> visual_slam/orbslam/slam/optimizer_g2o.py
NOTE: adapt g2o edge/camera API through visual_slam.g2o_compat.

third_party/pyslam_reference/pyslam/slam/motion_model.py
-> visual_slam/orbslam/slam/motion_model.py

third_party/pyslam_reference/pyslam/slam/rotation_histogram.py
-> visual_slam/orbslam/slam/rotation_histogram.py

third_party/pyslam_reference/pyslam/slam/geometry_matchers.py
-> visual_slam/orbslam/slam/geometry_matchers.py

third_party/pyslam_reference/pyslam/slam/tracking_core.py
-> visual_slam/orbslam/slam/tracking_core.py

third_party/pyslam_reference/pyslam/slam/tracking.py
-> visual_slam/orbslam/slam/tracking.py

third_party/pyslam_reference/pyslam/slam/local_mapping_core.py
-> visual_slam/orbslam/slam/local_mapping_core.py

third_party/pyslam_reference/pyslam/slam/local_mapping.py
-> visual_slam/orbslam/slam/local_mapping.py

third_party/pyslam_reference/pyslam/slam/global_bundle_adjustment.py
-> visual_slam/orbslam/slam/global_bundle_adjustment.py

third_party/pyslam_reference/pyslam/slam/relocalizer.py
-> visual_slam/orbslam/slam/relocalizer.py

## Local features
third_party/pyslam_reference/pyslam/local_features/feature_orbslam2.py
-> visual_slam/orbslam/local_features/feature_orbslam2.py

third_party/pyslam_reference/pyslam/local_features/feature_tracker.py
-> visual_slam/orbslam/local_features/feature_tracker.py

third_party/pyslam_reference/pyslam/local_features/feature_tracker_configs.py
-> visual_slam/orbslam/local_features/feature_tracker_configs.py

third_party/pyslam_reference/pyslam/local_features/feature_manager.py
-> visual_slam/orbslam/local_features/feature_manager.py

third_party/pyslam_reference/pyslam/local_features/feature_types.py
-> visual_slam/orbslam/local_features/feature_types.py

third_party/pyslam_reference/pyslam/local_features/feature_matcher.py
-> visual_slam/orbslam/local_features/feature_matcher.py

## Loop closing
third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py
-> visual_slam/orbslam/loop_closing/loop_closing.py

third_party/pyslam_reference/pyslam/loop_closing/loop_detecting_process.py
-> visual_slam/orbslam/loop_closing/loop_detecting_process.py

third_party/pyslam_reference/pyslam/loop_closing/loop_detector_configs.py
-> visual_slam/orbslam/loop_closing/loop_detector_configs.py

third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py
-> visual_slam/orbslam/loop_closing/loop_detector_dbow3.py

third_party/pyslam_reference/pyslam/loop_closing/loop_detector_vocabulary.py
-> visual_slam/orbslam/loop_closing/loop_detector_vocabulary.py

third_party/pyslam_reference/pyslam/loop_closing/loop_detector_database.py
-> visual_slam/orbslam/loop_closing/loop_detector_database.py

third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py
-> visual_slam/orbslam/loop_closing/keyframe_database.py
