"""
Local-mapping worker and queue manager.
This module receives new keyframes, runs local-map updates, and coordinates optional threading.
"""

from __future__ import annotations

from collections import defaultdict
from queue import Queue
from threading import Condition, RLock
import time
import traceback

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.geometry_matchers import EpipolarMatcher
from visual_slam.orbslam.slam.local_mapping_core import LocalMappingCore
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.utilities.geom_triangulation import triangulate_normalized_points
from visual_slam.orbslam.utilities.logging import Printer

import threading as _threading

try:
    import cpp_slam_core as _cpp_slam_core
    _CppLocalMappingCore = getattr(_cpp_slam_core, "LocalMappingCore", None)
    _CPP_LMC_AVAILABLE = _CppLocalMappingCore is not None
except ImportError:
    _CppLocalMappingCore = None
    _CPP_LMC_AVAILABLE = False

kVerbose = True
kUseLargeWindowBA = Parameters.kUseLargeWindowBA
kLocalMappingSleepTime = 5e-3


# Coordinate keyframe insertion and execution of local-mapping work.
class LocalMapping:
    print = staticmethod(lambda *args, **kwargs: None)

    def __init__(self, slam):
        self.slam = slam
        if _CPP_LMC_AVAILABLE:
            self.local_mapping_core = _CppLocalMappingCore(slam.map, slam.sensor_type.value)
            self._use_cpp_lmc = True
        else:
            self.local_mapping_core = LocalMappingCore(slam.map, slam.sensor_type)
            self._use_cpp_lmc = False

        self.queue = Queue()
        self.queue_condition = Condition()
        self.idle_condition = Condition()
        self.stop_mutex = RLock()
        self.reset_mutex = RLock()

        self.is_running = False
        self._is_idle = True
        self.stop_requested = False
        self.do_not_stop = False
        self.stopped = False
        self.reset_requested = False

        self.depth_cur = None
        self.img_cur_right = None
        self.img_cur = None

        self.mean_ba_chi2_error = None
        self.time_local_mapping = None

        self.far_points_threshold = None
        self.use_fov_centers_based_kf_generation = False
        self.max_fov_centers_distance = -1

        self.last_processed_kf_img_id = None
        self.last_num_triangulated_points = None
        self.total_num_triangulated_points = 0
        self.last_num_fused_points = None
        self.total_num_fused_points = 0
        self.last_num_culled_points = None
        self.total_num_culled_points = 0
        self.last_num_culled_keyframes = None
        self.total_num_culled_keyframes = 0

        self._thread: _threading.Thread | None = None

        self.init_print()

    def init_print(self):
        if kVerbose:
            LocalMapping.print = staticmethod(print)
        if not self._use_cpp_lmc and hasattr(LocalMappingCore, "print"):
            LocalMappingCore.print = LocalMapping.print

    @property
    def map(self):
        return self.slam.map

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def kf_cur(self):
        return self.local_mapping_core.kf_cur

    @kf_cur.setter
    def kf_cur(self, value):
        self.local_mapping_core.kf_cur = value

    @property
    def kid_last_BA(self):
        return self.local_mapping_core.kid_last_BA

    @kid_last_BA.setter
    def kid_last_BA(self, value):
        self.local_mapping_core.kid_last_BA = value

    @property
    def descriptor_distance_sigma(self):
        return self.slam.tracking.descriptor_distance_sigma

    def set_opt_abort_flag(self, value):
        self.local_mapping_core.set_opt_abort_flag(value)

    def interrupt_optimization(self):
        """Signal the BA optimizer to abort early so LM can become idle sooner."""
        self.set_opt_abort_flag(True)

    def is_stopped(self) -> bool:
        return bool(getattr(self, "stopped", False))

    def is_stop_requested(self) -> bool:
        return bool(getattr(self, "stop_requested", False))

    def push_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        with self.queue_condition:
            self.queue.put((keyframe, img, img_right, depth))
            self.queue_condition.notify_all()
        self.set_opt_abort_flag(True)

    def insert_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        self.push_keyframe(keyframe, img=img, img_right=img_right, depth=depth)

    def pop_keyframe(self, timeout=Parameters.kLocalMappingTimeoutPopKeyframe):
        with self.queue_condition:
            if self.queue.empty():
                self.queue_condition.wait(timeout=timeout)
            if self.queue.empty() or self.stop_requested:
                return None
            return self.queue.get(timeout=timeout)

    def queue_size(self):
        return self.queue.qsize()

    def is_idle(self):
        with self.idle_condition:
            return self._is_idle

    def set_idle(self, flag):
        with self.idle_condition:
            self._is_idle = bool(flag)
            self.idle_condition.notify_all()

    def wait_idle(self, print=print, timeout=None):
        with self.idle_condition:
            while not self._is_idle and self.is_running:
                ok = self.idle_condition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(f"LocalMapping: timeout {timeout}s reached")
                    return

    def request_reset(self):
        with self.reset_mutex:
            self.reset_requested = True

    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                while not self.queue.empty():
                    self.queue.get()
                self.reset_requested = False
                self.total_num_triangulated_points = 0
                self.total_num_fused_points = 0
                self.total_num_culled_points = 0
                self.total_num_culled_keyframes = 0
                self.last_num_triangulated_points = None
                self.local_mapping_core.reset()

    def step(self):
        if self.map.num_keyframes() <= 0:
            time.sleep(kLocalMappingSleepTime)
            return

        ret = self.pop_keyframe(timeout=0.0)

        if ret is None:
            self.set_idle(True)
            return

        self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret

        if self.kf_cur is None:
            self.set_idle(True)
            return

        self.last_processed_kf_img_id = getattr(self.kf_cur, "img_id", None)
        self.set_idle(False)

        try:
            self.do_local_mapping()
        except Exception as exc:
            LocalMapping.print(f"LocalMapping: encountered exception: {exc}")
            LocalMapping.print(traceback.format_exc())
            raise
        finally:
            self.set_idle(True)
            self.reset_if_requested()

    def do_local_mapping(self):
        LocalMapping.print("local mapping: starting...")
        time_start = time.time()

        if self.kf_cur is None:
            Printer.red("local mapping: no keyframe to process")
            return

        self.process_new_keyframe()

        num_culled_points = self.cull_map_points()
        self.last_num_culled_points = num_culled_points
        self.total_num_culled_points += num_culled_points

        total_new_pts = self.create_new_map_points()
        self.last_num_triangulated_points = total_new_pts
        self.total_num_triangulated_points += total_new_pts

        total_fused_pts = self.fuse_map_points()
        self.last_num_fused_points = total_fused_pts
        self.total_num_fused_points += total_fused_pts

        self.set_opt_abort_flag(False)
        self.local_BA()

        num_culled_keyframes = self.cull_keyframes()
        self.last_num_culled_keyframes = num_culled_keyframes
        self.total_num_culled_keyframes += num_culled_keyframes

        self.time_local_mapping = time.time() - time_start
        LocalMapping.print(f"local mapping elapsed time: {self.time_local_mapping}")

    def local_BA(self):
        if getattr(self.slam, "loop_closing", None) is not None:
            if self.slam.loop_closing.is_correcting():
                return

        err, num_kf_ref_tracked_points = self.local_mapping_core.local_BA()
        self.mean_ba_chi2_error = err

        if getattr(self.slam, "tracking", None) is not None:
            self.slam.tracking.num_kf_ref_tracked_points = num_kf_ref_tracked_points

    def large_window_BA(self):
        result = self.local_mapping_core.large_window_BA()
        if isinstance(result, tuple):
            return result[0]
        return result

    def process_new_keyframe(self):
        self.local_mapping_core.process_new_keyframe()

    def cull_map_points(self):
        return self.local_mapping_core.cull_map_points()

    def cull_keyframes(self):
        if self._use_cpp_lmc:
            return self.local_mapping_core.cull_keyframes()
        return self.local_mapping_core.cull_keyframes(
            self.use_fov_centers_based_kf_generation,
            self.max_fov_centers_distance,
        )

    def _get_local_mapping_neighbors(self):
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular
        else:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo

        if hasattr(self.map, "local_map") and hasattr(self.map.local_map, "get_best_neighbors"):
            return self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)

        return self.kf_cur.get_best_covisible_keyframes(num_neighbors)

    def create_new_map_points(self):
        total_new_pts = 0

        local_keyframes = [
            kf for kf in self._get_local_mapping_neighbors()
            if kf is not None and kf is not self.kf_cur and not kf.is_bad()
        ]

        for kf in local_keyframes:
            if not self.queue.empty():
                return total_new_pts

            idxs_cur, idxs, num_found_matches = EpipolarMatcher.search_frame_for_triangulation(
                self.kf_cur,
                kf,
                None,
                None,
                max_descriptor_distance=0.5 * self.descriptor_distance_sigma,
                is_monocular=(self.sensor_type == SensorType.MONOCULAR),
            )

            if num_found_matches == 0:
                continue

            pts3d, mask_pts3d = triangulate_normalized_points(
                self.kf_cur.pose(),
                kf.pose(),
                self.kf_cur.kpsn[idxs_cur],
                kf.kpsn[idxs],
            )

            new_pts_count, _, list_added_points = self.map.add_points(
                pts3d,
                mask_pts3d,
                self.kf_cur,
                kf,
                idxs_cur,
                idxs,
                self.img_cur,
                do_check=True,
                far_points_threshold=self.far_points_threshold,
            )

            total_new_pts += new_pts_count
            self.local_mapping_core.add_points(list_added_points)

        return total_new_pts

    def fuse_map_points(self):
        return self.local_mapping_core.fuse_map_points(self.descriptor_distance_sigma)

    # ------------------------------------------------------------------
    # Background-thread support
    # ------------------------------------------------------------------

    def start_thread(self):
        """Start local mapping on a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.is_running = True
        self.stop_requested = False
        self._thread = _threading.Thread(
            target=self._run_thread, daemon=True, name="LocalMapping"
        )
        self._thread.start()
        Printer.green("LocalMapping: background thread started")

    def stop_thread(self, timeout: float = 10.0):
        """Signal the background thread to stop and wait for it to exit."""
        self.is_running = False
        self.stop_requested = True
        with self.queue_condition:
            self.queue_condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                Printer.yellow(f"LocalMapping: thread did not exit within {timeout}s")
        self._thread = None
        Printer.green("LocalMapping: background thread stopped")

    def _run_thread(self):
        """Inner loop run by the background thread."""
        while self.is_running and not self.stop_requested:
            try:
                # Blocking pop with default 0.5 s timeout — thread sleeps when idle
                ret = self.pop_keyframe()
                if ret is None:
                    self.set_idle(True)
                    continue

                self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret
                if self.kf_cur is None:
                    self.set_idle(True)
                    continue

                self.last_processed_kf_img_id = getattr(self.kf_cur, "img_id", None)
                self.set_idle(False)
                try:
                    self.do_local_mapping()
                except Exception as exc:
                    LocalMapping.print(f"LocalMapping thread: {exc}")
                    LocalMapping.print(traceback.format_exc())
                finally:
                    self.set_idle(True)
                    self.reset_if_requested()
            except Exception as exc:
                Printer.red(f"LocalMapping thread outer: {exc}")
                time.sleep(kLocalMappingSleepTime)
        self.set_idle(True)
