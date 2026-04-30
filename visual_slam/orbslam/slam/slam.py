"""
=============================================================================
visual_slam/orbslam/slam/slam.py

pySLAM-aligned sparse ORB-SLAM system orchestrator.

Reference:
- pySLAM: pyslam/slam/slam.py

Responsibilities:
- own camera / feature tracker / map / local mapping / tracking
- initialize FeatureTrackerShared through init_feature_tracker()
- expose pySLAM-compatible track() delegation
- provide reset/quit/config-distribution hooks

Deferred:
- semantic mapping
- volumetric integration
- full loop closing
- system serialization/reload
=============================================================================
"""

from __future__ import annotations

from enum import Enum
import time
import traceback

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.local_mapping import LocalMapping
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.tracking import Tracking
from visual_slam.orbslam.utilities.logging import Printer


kVerbose = True


class SlamMode(Enum):
    SLAM = 0
    MAP_BROWSER = 1


class Slam:
    """
    pySLAM-style sparse SLAM system container.

    The construction order intentionally follows pySLAM:
        1. store camera/config/sensor metadata
        2. initialize feature tracker and FeatureTrackerShared
        3. create map
        4. create local mapping
        5. create tracking
        6. distribute optional config parameters
    """

    def __init__(
        self,
        camera,
        feature_tracker_config: dict | None = None,
        loop_detector_config=None,
        semantic_mapping_config=None,
        sensor_type: SensorType = SensorType.RGBD,
        environment_type=None,
        slam_mode: SlamMode = SlamMode.SLAM,
        config=None,
        headless: bool = True,
        viewer3d=None,
        start_local_mapping_thread: bool | None = None,
    ):
        self.camera = camera
        self.feature_tracker_config = feature_tracker_config or {}
        self.loop_detector_config = loop_detector_config
        self.semantic_mapping_config = semantic_mapping_config
        self.sensor_type = sensor_type
        self.environment_type = environment_type
        self.slam_mode = slam_mode
        self.headless = headless
        self.viewer3d = viewer3d

        self.feature_tracker = None
        self.init_feature_tracker(self.feature_tracker_config)

        self.map = Map()

        self.semantic_mapping = None
        self.loop_closing = None
        self.GBA = None
        self.GBA_on_demand = None
        self.volumetric_integrator = None

        self.local_mapping = LocalMapping(self)
        self.tracking = Tracking(self)

        self.reset_requested = False
        self.has_quit = False
        self.config = None

        self.set_config_params(config)

        # pySLAM starts local mapping on a separate thread when configured.
        # This port keeps local mapping sequential until the full online path is stable.
        self.start_local_mapping_thread = (
            Parameters.kLocalMappingOnSeparateThread
            if start_local_mapping_thread is None
            else bool(start_local_mapping_thread)
        )

        if self.start_local_mapping_thread:
            Printer.orange(
                "Slam: local-mapping thread start requested, but this ORB subset "
                "currently runs local mapping sequentially."
            )

    def set_config_params(self, config):
        self.config = config

        if config is None:
            return

        far_points_threshold = getattr(config, "far_points_threshold", None)
        use_fov_centers = getattr(config, "use_fov_centers_based_kf_generation", False)
        max_fov_centers_distance = getattr(config, "max_fov_centers_distance", -1)

        if self.tracking is not None:
            self.tracking.far_points_threshold = far_points_threshold
            self.tracking.use_fov_centers_based_kf_generation = use_fov_centers
            self.tracking.max_fov_centers_distance = max_fov_centers_distance

        if self.local_mapping is not None:
            self.local_mapping.far_points_threshold = far_points_threshold
            self.local_mapping.use_fov_centers_based_kf_generation = use_fov_centers
            self.local_mapping.max_fov_centers_distance = max_fov_centers_distance

    def init_feature_tracker(self, feature_tracker_config):
        """
        Initialize ORB2 feature tracker and FeatureTrackerShared.

        For now, this ORB-SLAM subset intentionally supports the ORB2 path only.
        Additional feature-manager choices can be added later if they are needed
        for comparison, but the thesis benchmark target is ORB/RGB-D.
        """
        if feature_tracker_config is None:
            feature_tracker_config = {}

        if "feature_tracker" in feature_tracker_config:
            feature_tracker = feature_tracker_config["feature_tracker"]
        else:
            feature_tracker = create_orb2_feature_tracker(**feature_tracker_config)

        self.feature_tracker = feature_tracker

        # pySLAM sets this with force=True during Slam initialization.
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

        if self.sensor_type == SensorType.STEREO:
            # Reserved for true stereo camera path. RGB-D target does not need it.
            try:
                feature_tracker_right = create_orb2_feature_tracker(**feature_tracker_config)
                FeatureTrackerShared.set_feature_tracker_right(feature_tracker_right, force=True)
            except TypeError:
                pass

    def request_reset(self):
        self.reset_requested = True

    def reset(self):
        if self.local_mapping is not None:
            self.local_mapping.request_reset()

        if self.tracking is not None:
            self.tracking.reset()

        if self.map is not None:
            self.map.reset()

        self.reset_requested = False

    def reset_session(self):
        self.reset()

    def quit(self):
        if self.has_quit:
            return

        self.has_quit = True

        # No background local mapping thread is started yet, but keep the hook.
        time.sleep(0.01)

    def __del__(self):
        try:
            self.quit()
        except Exception:
            pass

    def track(
        self,
        img,
        img_right=None,
        depth=None,
        img_id=None,
        timestamp=None,
        mask=None,
        mask_right=None,
    ):
        """
        pySLAM-compatible public tracking entry point.

        Delegates to Tracking.track() with the same argument order.
        """
        if self.reset_requested:
            self.reset()

        try:
            return self.tracking.track(
                img,
                img_right,
                depth,
                img_id,
                timestamp,
                mask,
                mask_right,
            )
        except Exception:
            Printer.red("Slam.track(): tracking exception")
            Printer.red(traceback.format_exc())
            raise

    def set_tracking_state(self, state: SlamState):
        self.tracking.state = state

    def bundle_adjust(self):
        return self.map.optimize()

    def get_final_trajectory(self):
        """
        Return pySLAM-style final trajectory containers available in Tracking.
        """
        return {
            "poses": list(self.tracking.poses),
            "timestamps": list(self.tracking.pose_timestamps),
            "history": self.tracking.tracking_history,
        }

    def get_tracking_state(self):
        return self.tracking.state

    def is_ok(self):
        return self.tracking.state == SlamState.OK
