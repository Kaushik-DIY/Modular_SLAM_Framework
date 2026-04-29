"""
=============================================================================
visual_slam/types.py

Core SLAM data structures: Frame, KeyFrame, MapPoint, Map.

IMPORTANT: This module follows pyslam's exact data structure design.
---------------------------------------------------------------------------
Ported from pyslam's frame.py, keyframe.py, map_point.py, and map.py.
The Python implementation mirrors pyslam's field names and relationships
to ensure compatibility when porting tracking/mapping/loop-closing modules.

Classes
-------
Frame
    Holds one RGB+Depth image pair with extracted features.
    
KeyFrame
    Extends Frame with SLAM bookkeeping (covisibility graph, BoW vectors).
    
MapPoint
    A 3D landmark observed from multiple keyframes.
    
Map
    Global map containing all keyframes and map points.

Design Notes
------------
- pose_world is g2o.Isometry3d (following our Phase 1 design)
- camera is CameraIntrinsics from slam_core.common.types3d
- keypoints are cv2.KeyPoint objects (OpenCV format)
- descriptors are np.ndarray with dtype=np.uint8 for ORB (Nx32 for ORB)

References
----------
pyslam GitHub: https://github.com/luigifreda/pyslam
Specifically: frame.py, keyframe.py, map_point.py, map.py

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from threading import Lock
import numpy as np
import cv2

try:
    import g2o
except ImportError:
    g2o = None
    print("WARNING: g2o not found. Frame poses will not work.")

from slam_core.common.types3d import CameraIntrinsics, Pose3D


# ===========================================================================
# Frame — One RGB+Depth Image Pair with Features
# ===========================================================================

class Frame:
    """
    Frame holds one RGB+Depth image pair with extracted features.
    
    This is the basic unit processed by the tracking front-end.
    A subset of frames become keyframes (stored in the map).
    
    Attributes
    ----------
    frame_id : int
        Unique frame identifier (incremented globally).
    timestamp : float
        Frame timestamp in seconds.
    image_rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype=np.uint8.
    image_depth : np.ndarray
        Depth image, shape (H, W), dtype=np.uint16.
        Raw depth values (divide by camera.depth_scale to get meters).
    camera : CameraIntrinsics
        Camera parameters for this frame.
    keypoints : List[cv2.KeyPoint]
        Detected keypoints (OpenCV format).
        Each cv2.KeyPoint has: pt (u,v), size, angle, response, octave.
    descriptors : np.ndarray
        Feature descriptors, shape (N, descriptor_size).
        For ORB: (N, 32), dtype=np.uint8.
    depths : np.ndarray
        Depth at each keypoint in meters, shape (N,).
        Computed by looking up image_depth at keypoint.pt and dividing by depth_scale.
    pose_world : g2o.Isometry3d or None
        Camera pose in world frame (T_world_from_camera).
        None if pose not yet estimated.
    map_point_matches : List[MapPoint or None]
        Matched map point for each keypoint index.
        Length = len(keypoints). None if no match.
    
    Methods
    -------
    from_images(timestamp, rgb, depth, camera) -> Frame
        Factory method to create a frame from images.
    
    Notes
    -----
    The pose convention follows pyslam:
        T_world_from_camera (not T_camera_from_world)
    
    This means: point_world = T_world_from_camera @ point_camera
    
    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>> from slam_core.common.types3d import CameraIntrinsics
    >>> 
    >>> # Create camera
    >>> cam = CameraIntrinsics(fx=517.3, fy=516.5, cx=318.6, cy=255.3,
    ...                        width=640, height=480, depth_scale=5000.0)
    >>> 
    >>> # Create frame from images
    >>> rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> depth = np.zeros((480, 640), dtype=np.uint16)
    >>> frame = Frame.from_images(timestamp=0.0, rgb=rgb, depth=depth, camera=cam)
    >>> 
    >>> # Features will be extracted later by FeatureTracker
    >>> frame.keypoints = []
    >>> frame.descriptors = np.array([], dtype=np.uint8)
    """
    
    # Class variable for global frame ID counter
    _next_id: int = 0
    _id_lock: Lock = Lock()
    
    def __init__(
        self,
        timestamp: float,
        image_rgb: np.ndarray,
        image_depth: np.ndarray,
        camera: CameraIntrinsics,
    ):
        """
        Initialize a Frame.
        
        Parameters
        ----------
        timestamp : float
            Frame timestamp in seconds.
        image_rgb : np.ndarray
            RGB image, shape (H, W, 3).
        image_depth : np.ndarray
            Depth image, shape (H, W).
        camera : CameraIntrinsics
            Camera parameters.
        """
        # Assign unique frame ID
        with Frame._id_lock:
            self.frame_id = Frame._next_id
            Frame._next_id += 1
        
        self.timestamp = timestamp
        self.image_rgb = image_rgb
        self.image_depth = image_depth
        self.camera = camera
        
        # Features (populated later by FeatureTracker)
        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: np.ndarray = np.array([], dtype=np.uint8)
        self.depths: np.ndarray = np.array([], dtype=np.float32)
        
        # Pose (estimated by Tracker)
        self.pose_world: Optional[Pose3D] = None  # g2o.Isometry3d or None
        
        # Map point associations (updated by Tracker and LocalMapping)
        self.map_point_matches: List[Optional[MapPoint]] = []
    
    @staticmethod
    def from_images(
        timestamp: float,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: CameraIntrinsics,
    ) -> Frame:
        """
        Factory method to create a Frame from images.
        
        Parameters
        ----------
        timestamp : float
            Frame timestamp.
        rgb : np.ndarray
            RGB image (H, W, 3).
        depth : np.ndarray
            Depth image (H, W).
        camera : CameraIntrinsics
            Camera parameters.
        
        Returns
        -------
        Frame
            New frame instance.
        """
        return Frame(timestamp, rgb, depth, camera)
    
    def __repr__(self) -> str:
        return (f"Frame(id={self.frame_id}, t={self.timestamp:.3f}, "
                f"kpts={len(self.keypoints)})")


# ===========================================================================
# KeyFrame — Frame + SLAM Bookkeeping
# ===========================================================================

class KeyFrame:
    """
    KeyFrame extends Frame with SLAM bookkeeping structures.
    
    KeyFrames are stored in the Map and used for:
    - Local mapping (triangulation, local BA)
    - Loop closing (place recognition, PGO)
    - Relocalization (when tracking is lost)
    
    Attributes
    ----------
    keyframe_id : int
        Unique keyframe identifier.
    frame : Frame
        The underlying frame data.
    is_bad : bool
        True if this keyframe has been culled (marked for removal).
    connected_keyframes : Dict[KeyFrame, int]
        Covisibility graph: {keyframe -> num_shared_map_points}.
    map_points : Set[MapPoint]
        Map points observed by this keyframe.
    bow_vector : Optional[dict]
        Bag-of-Words vector for loop closure (DBoW3).
        Format: {word_id: weight}.
    feature_vector : Optional[dict]
        Feature vector for loop closure (DBoW3).
        Format: {node_id: [feature_indices]}.
    
    Methods
    -------
    from_frame(frame, keyframe_id) -> KeyFrame
        Factory method to create a keyframe from a frame.
    add_connection(kf, weight)
        Add a covisibility edge to another keyframe.
    get_best_covisible_keyframes(n) -> List[KeyFrame]
        Get top N keyframes by shared map points.
    
    Notes
    -----
    The covisibility graph is built incrementally as map points are created.
    When a map point is observed by two keyframes, their connection weight
    (number of shared map points) is incremented.
    
    Examples
    --------
    >>> frame = Frame.from_images(0.0, rgb, depth, cam)
    >>> kf = KeyFrame.from_frame(frame, keyframe_id=0)
    >>> kf.is_bad = False
    >>> print(kf)
    KeyFrame(id=0, frame=0, kpts=0, mps=0)
    """
    
    # Class variable for global keyframe ID counter
    _next_id: int = 0
    _id_lock: Lock = Lock()
    
    def __init__(self, frame: Frame):
        """
        Initialize a KeyFrame from a Frame.
        
        Parameters
        ----------
        frame : Frame
            The frame to promote to keyframe.
        """
        # Assign unique keyframe ID
        with KeyFrame._id_lock:
            self.keyframe_id = KeyFrame._next_id
            KeyFrame._next_id += 1
        
        self.frame = frame
        self.is_bad = False
        
        # Covisibility graph
        self.connected_keyframes: Dict[KeyFrame, int] = {}
        self._connections_lock = Lock()
        
        # Map points observed by this keyframe
        self.map_points: Set[MapPoint] = set()
        self._map_points_lock = Lock()
        
        # DBoW3 vectors for loop closure (computed later)
        self.bow_vector: Optional[dict] = None
        self.feature_vector: Optional[dict] = None
    
    @staticmethod
    def from_frame(frame: Frame, keyframe_id: Optional[int] = None) -> KeyFrame:
        """
        Factory method to create a KeyFrame from a Frame.
        
        Parameters
        ----------
        frame : Frame
            The frame to convert.
        keyframe_id : int, optional
            If provided, override the auto-assigned ID (for loading saved maps).
        
        Returns
        -------
        KeyFrame
            New keyframe instance.
        """
        kf = KeyFrame(frame)
        if keyframe_id is not None:
            kf.keyframe_id = keyframe_id
        return kf
    
    def add_connection(self, kf: KeyFrame, weight: int) -> None:
        """
        Add or update a covisibility connection to another keyframe.
        
        Parameters
        ----------
        kf : KeyFrame
            The connected keyframe.
        weight : int
            Number of shared map points.
        """
        with self._connections_lock:
            self.connected_keyframes[kf] = weight
    
    def get_best_covisible_keyframes(self, n: int) -> List[KeyFrame]:
        """
        Get top N keyframes by covisibility (shared map points).
        
        Parameters
        ----------
        n : int
            Number of keyframes to return.
        
        Returns
        -------
        List[KeyFrame]
            Top N connected keyframes, sorted by weight (descending).
        """
        with self._connections_lock:
            sorted_kfs = sorted(
                self.connected_keyframes.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [kf for kf, weight in sorted_kfs[:n]]
    
    def __repr__(self) -> str:
        return (f"KeyFrame(id={self.keyframe_id}, frame={self.frame.frame_id}, "
                f"kpts={len(self.frame.keypoints)}, mps={len(self.map_points)})")


# ===========================================================================
# MapPoint — 3D Landmark
# ===========================================================================

class MapPoint:
    """
    MapPoint represents a 3D landmark observed from multiple keyframes.
    
    Map points are triangulated from stereo/RGBD observations or from
    multiple views (for monocular SLAM). They form the sparse 3D map.
    
    Attributes
    ----------
    point_id : int
        Unique map point identifier.
    position_world : np.ndarray
        3D position in world frame, shape (3,).
    observations : Dict[KeyFrame, int]
        Keyframes observing this point: {keyframe -> keypoint_index}.
    descriptor : np.ndarray
        Representative descriptor (median of all observations).
        Shape (descriptor_size,), e.g., (32,) for ORB.
    is_bad : bool
        True if this map point has been culled.
    found_ratio : float
        Tracking quality: num_found / num_visible.
        Used to cull unreliable map points.
    
    Methods
    -------
    add_observation(kf, keypoint_idx)
        Add an observation from a keyframe.
    remove_observation(kf)
        Remove an observation.
    compute_descriptor()
        Recompute representative descriptor from observations.
    
    Notes
    -----
    The found_ratio tracks how often this map point is successfully
    matched when it should be visible. Low ratios indicate unreliable
    points (e.g., moving objects, outliers) which get culled.
    
    Examples
    --------
    >>> mp = MapPoint(point_id=0, position_world=np.array([1.0, 2.0, 3.0]))
    >>> mp.add_observation(kf, keypoint_idx=5)
    >>> print(mp)
    MapPoint(id=0, pos=[1.0, 2.0, 3.0], obs=1)
    """
    
    # Class variable for global map point ID counter
    _next_id: int = 0
    _id_lock: Lock = Lock()
    
    def __init__(
        self,
        position_world: np.ndarray,
        point_id: Optional[int] = None,
    ):
        """
        Initialize a MapPoint.
        
        Parameters
        ----------
        position_world : np.ndarray
            3D position in world frame, shape (3,).
        point_id : int, optional
            If provided, use this ID (for loading saved maps).
        """
        # Assign unique map point ID
        if point_id is not None:
            self.point_id = point_id
        else:
            with MapPoint._id_lock:
                self.point_id = MapPoint._next_id
                MapPoint._next_id += 1
        
        self.position_world = position_world.copy()
        self.observations: Dict[KeyFrame, int] = {}
        self._observations_lock = Lock()
        
        self.descriptor: Optional[np.ndarray] = None
        self.is_bad = False
        
        # Tracking quality metrics (ORB-SLAM2: found_ratio = n_found / n_visible)
        self.found_ratio       = 1.0
        self._num_visible      = 0
        self._num_found        = 0
        # Public counters incremented by tracking (aliases for _num_visible/_num_found)
        self.found_in_frames   = 0
        self.visible_in_frames = 0
    
    def add_observation(self, kf: KeyFrame, keypoint_idx: int) -> None:
        """
        Add an observation of this map point from a keyframe.
        
        Parameters
        ----------
        kf : KeyFrame
            The observing keyframe.
        keypoint_idx : int
            Index of the keypoint in kf.frame.keypoints.
        """
        with self._observations_lock:
            self.observations[kf] = keypoint_idx
            
        # Add this map point to the keyframe's set
        with kf._map_points_lock:
            kf.map_points.add(self)
    
    def remove_observation(self, kf: KeyFrame) -> None:
        """
        Remove an observation from a keyframe.
        
        Parameters
        ----------
        kf : KeyFrame
            The keyframe to remove.
        """
        with self._observations_lock:
            if kf in self.observations:
                del self.observations[kf]
        
        with kf._map_points_lock:
            kf.map_points.discard(self)
    
    def compute_descriptor(self) -> None:
        """
        Compute representative descriptor as median of all observations.
        
        This is called after adding/removing observations to update
        the descriptor used for matching.
        """
        if len(self.observations) == 0:
            self.descriptor = None
            return
        
        # Collect all descriptors from observations
        descriptors = []
        with self._observations_lock:
            for kf, keypoint_idx in self.observations.items():
                if keypoint_idx >= len(kf.frame.descriptors):
                    continue
                desc = kf.frame.descriptors[keypoint_idx]
                descriptors.append(desc)
        
        if len(descriptors) == 0:
            self.descriptor = None
            return
        
        # Compute median descriptor (element-wise median)
        descriptors_array = np.array(descriptors, dtype=np.float32)
        median_desc = np.median(descriptors_array, axis=0)
        self.descriptor = median_desc.astype(np.uint8)
    
    def __repr__(self) -> str:
        return (f"MapPoint(id={self.point_id}, "
                f"pos=[{self.position_world[0]:.1f}, {self.position_world[1]:.1f}, "
                f"{self.position_world[2]:.1f}], obs={len(self.observations)})")


# ===========================================================================
# Map — Global SLAM Map
# ===========================================================================

class Map:
    """
    Map holds all keyframes and map points.
    
    This is the central data structure shared by:
    - Tracking (queries map points for matching)
    - Local Mapping (adds/removes keyframes and map points)
    - Loop Closing (performs PGO and GBA on the map)
    
    Attributes
    ----------
    keyframes : Dict[int, KeyFrame]
        All keyframes: {keyframe_id -> KeyFrame}.
    map_points : Dict[int, MapPoint]
        All map points: {point_id -> MapPoint}.
    
    Methods
    -------
    add_keyframe(kf)
        Add a keyframe to the map.
    add_map_point(mp)
        Add a map point to the map.
    remove_keyframe(kf)
        Remove a keyframe (marks as bad).
    remove_map_point(mp)
        Remove a map point (marks as bad).
    get_local_keyframes(current_kf, n) -> List[KeyFrame]
        Get spatially nearby keyframes.
    
    Notes
    -----
    Thread safety: All mutations are protected by locks since the map
    is accessed concurrently by Tracking, LocalMapping, and LoopClosing.
    
    Examples
    --------
    >>> slam_map = Map()
    >>> slam_map.add_keyframe(kf)
    >>> slam_map.add_map_point(mp)
    >>> print(f"Map: {len(slam_map.keyframes)} KFs, {len(slam_map.map_points)} MPs")
    Map: 1 KFs, 1 MPs
    """
    
    def __init__(self):
        """Initialize an empty map."""
        self.keyframes: Dict[int, KeyFrame] = {}
        self.map_points: Dict[int, MapPoint] = {}
        
        self._keyframes_lock = Lock()
        self._map_points_lock = Lock()
    
    def add_keyframe(self, kf: KeyFrame) -> None:
        """
        Add a keyframe to the map.
        
        Parameters
        ----------
        kf : KeyFrame
            Keyframe to add.
        """
        with self._keyframes_lock:
            self.keyframes[kf.keyframe_id] = kf
    
    def add_map_point(self, mp: MapPoint) -> None:
        """
        Add a map point to the map.
        
        Parameters
        ----------
        mp : MapPoint
            Map point to add.
        """
        with self._map_points_lock:
            self.map_points[mp.point_id] = mp
    
    def remove_keyframe(self, kf: KeyFrame) -> None:
        """
        Remove a keyframe from the map.
        
        Marks the keyframe as bad rather than deleting it,
        to preserve references from map points.
        
        Parameters
        ----------
        kf : KeyFrame
            Keyframe to remove.
        """
        kf.is_bad = True
        # Note: Do NOT delete from dict to preserve references
    
    def remove_map_point(self, mp: MapPoint) -> None:
        """
        Remove a map point from the map.
        
        Marks the map point as bad rather than deleting it.
        
        Parameters
        ----------
        mp : MapPoint
            Map point to remove.
        """
        mp.is_bad = True
    
    def get_local_keyframes(
        self,
        current_kf: KeyFrame,
        n: int = 10
    ) -> List[KeyFrame]:
        """
        Get local keyframes around the current keyframe.
        
        Returns keyframes that share map points with current_kf,
        sorted by covisibility (most shared points first).
        
        Parameters
        ----------
        current_kf : KeyFrame
            Reference keyframe.
        n : int
            Maximum number of keyframes to return.
        
        Returns
        -------
        List[KeyFrame]
            Local keyframes sorted by covisibility.
        """
        return current_kf.get_best_covisible_keyframes(n)
    
    def __repr__(self) -> str:
        return f"Map(keyframes={len(self.keyframes)}, map_points={len(self.map_points)})"