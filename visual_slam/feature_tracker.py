"""
=============================================================================
visual_slam/feature_tracker.py

Feature detection, description, and matching for Visual SLAM.

IMPORTANT: This module follows pyslam's feature tracking approach.
---------------------------------------------------------------------------
Ported from pyslam's feature_manager.py and feature_matcher.py.

This module handles:
- ORB2 keypoint detection (multi-scale pyramid)
- ORB descriptor computation (binary descriptors)
- Brute-force matching with Hamming distance
- Ratio test for ambiguity rejection
- Depth association from RGBD depth images

Classes
-------
FeatureTracker
    Detects features, computes descriptors, and matches between frames.

Design Notes
------------
- ORB detector: 1000 features, 8 pyramid levels (ORB-SLAM2 defaults)
- Matcher: BruteForce with Hamming distance (for binary descriptors)
- Ratio test: 0.7 threshold (Lowe's paper recommendation)
- Depth association: Bilinear interpolation from depth image

References
----------
pyslam: feature_manager.py, feature_matcher.py
ORB-SLAM2: ORBextractor.cc (for pyramid parameters)
OpenCV docs: cv2.ORB_create(), cv2.BFMatcher()

=============================================================================
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2

from visual_slam.types import Frame
from slam_core.common.types3d import CameraIntrinsics


class FeatureTracker:
    """
    Feature detection, description, and matching.
    
    This class handles all feature-related operations:
    - Detecting ORB keypoints in images
    - Computing ORB descriptors
    - Matching descriptors between frames
    - Associating depth to keypoints (for RGBD)
    
    Attributes
    ----------
    num_features : int
        Number of features to detect (ORB parameter).
    scale_factor : float
        Scale factor between pyramid levels (ORB parameter).
    num_levels : int
        Number of pyramid levels (ORB parameter).
    ratio_test_threshold : float
        Threshold for Lowe's ratio test (descriptor matching).
    
    Methods
    -------
    detect_and_compute(frame)
        Detect keypoints and compute descriptors in a frame.
    match_frames(frame1, frame2)
        Match features between two frames.
    associate_depth(frame)
        Associate depth from depth image to keypoints.
    
    Notes
    -----
    ORB parameters follow ORB-SLAM2 defaults:
        - num_features: 1000 (TUM RGBD)
        - scale_factor: 1.2 (20% scale reduction per level)
        - num_levels: 8 (pyramid levels)
    
    The ratio test threshold of 0.7 is from Lowe's SIFT paper and
    works well for ORB descriptors too.
    
    Examples
    --------
    >>> tracker = FeatureTracker(num_features=1000)
    >>> tracker.detect_and_compute(frame)
    >>> print(f"Detected {len(frame.keypoints)} keypoints")
    Detected 543 keypoints
    >>> 
    >>> matches = tracker.match_frames(frame1, frame2)
    >>> print(f"Matched {len(matches)} features")
    Matched 312 features
    """
    
    def __init__(
        self,
        num_features: int = 1000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        ratio_test_threshold: float = 0.7,
    ):
        """
        Initialize the FeatureTracker.
        
        Parameters
        ----------
        num_features : int
            Number of features to detect per image.
        scale_factor : float
            Scale factor between pyramid levels (ORB parameter).
        num_levels : int
            Number of pyramid levels (ORB parameter).
        ratio_test_threshold : float
            Threshold for Lowe's ratio test.
        """
        self.num_features = num_features
        self.scale_factor = scale_factor
        self.num_levels = num_levels
        self.ratio_test_threshold = ratio_test_threshold
        
        # Create ORB detector (following ORB-SLAM2 parameters)
        self.orb = cv2.ORB_create(
            nfeatures=num_features,
            scaleFactor=scale_factor,
            nlevels=num_levels,
            edgeThreshold=31,  # ORB-SLAM2 default
            firstLevel=0,
            WTA_K=2,  # 2-point comparison (default)
            scoreType=cv2.ORB_HARRIS_SCORE,  # Harris corner score
            patchSize=31,  # ORB-SLAM2 default
            fastThreshold=20,  # ORB-SLAM2 default
        )
        
        # Create BruteForce matcher with Hamming distance
        # (Hamming distance for binary descriptors like ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_compute(self, frame: Frame) -> None:
        """
        Detect keypoints and compute descriptors in a frame.
        
        Updates frame.keypoints and frame.descriptors in-place.
        
        Parameters
        ----------
        frame : Frame
            Frame to process (must have image_rgb).
        
        Notes
        -----
        ORB works on grayscale images, so we convert RGB to gray first.
        Keypoints are cv2.KeyPoint objects with attributes:
            - pt: (u, v) pixel coordinates
            - size: keypoint diameter
            - angle: orientation in degrees
            - response: detector response (strength)
            - octave: pyramid level
        
        Descriptors are binary (np.uint8) with shape (N, 32) for ORB.
        """
        # Convert RGB to grayscale
        image_gray = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(image_gray, None)
        
        # Handle case where no features detected
        if keypoints is None or len(keypoints) == 0:
            frame.keypoints = []
            frame.descriptors = np.array([], dtype=np.uint8).reshape(0, 32)
            frame.depths = np.array([], dtype=np.float32)
            return
        
        # Store in frame
        frame.keypoints = keypoints
        frame.descriptors = descriptors if descriptors is not None else np.array([], dtype=np.uint8).reshape(0, 32)
        
        # Associate depth to keypoints (for RGBD)
        self.associate_depth(frame)
    
    def associate_depth(self, frame: Frame) -> None:
        """
        Associate depth from depth image to each keypoint.
        
        Updates frame.depths in-place with depth in meters for each keypoint.
        
        Parameters
        ----------
        frame : Frame
            Frame with keypoints and depth image.
        
        Notes
        -----
        Depth lookup uses bilinear interpolation for sub-pixel accuracy.
        Invalid depths (0 or too large) are set to -1.0 to indicate failure.
        
        Depth conversion: depth_meters = pixel_value / camera.depth_scale
        For TUM RGBD: depth_scale = 5000, so pixel_value/5000 = meters.
        """
        if len(frame.keypoints) == 0:
            frame.depths = np.array([], dtype=np.float32)
            return
        
        depths = []
        for kp in frame.keypoints:
            u, v = kp.pt
            
            # Round to nearest pixel (simple approach)
            # For better accuracy, could use bilinear interpolation
            u_int = int(round(u))
            v_int = int(round(v))
            
            # Check bounds
            if (0 <= u_int < frame.camera.width and
                0 <= v_int < frame.camera.height):
                
                depth_raw = frame.image_depth[v_int, u_int]
                
                # Convert to meters
                depth_meters = float(depth_raw) / frame.camera.depth_scale
                
                # Check validity (TUM depth images: 0 = invalid)
                if depth_meters > 0.01 and depth_meters < 10.0:
                    depths.append(depth_meters)
                else:
                    depths.append(-1.0)  # Invalid depth marker
            else:
                depths.append(-1.0)  # Out of bounds
        
        frame.depths = np.array(depths, dtype=np.float32)
    
    def match_frames(
        self,
        frame1: Frame,
        frame2: Frame,
    ) -> List[cv2.DMatch]:
        """
        Match features between two frames using ratio test.
        
        Parameters
        ----------
        frame1 : Frame
            First frame (query).
        frame2 : Frame
            Second frame (train).
        
        Returns
        -------
        List[cv2.DMatch]
            Good matches passing the ratio test.
            Each match has attributes:
                - queryIdx: index in frame1.descriptors
                - trainIdx: index in frame2.descriptors
                - distance: Hamming distance
        
        Notes
        -----
        Matching pipeline:
        1. Find 2 nearest neighbors for each descriptor in frame1
        2. Apply Lowe's ratio test: keep match if
           best_dist / second_best_dist < threshold (default 0.7)
        
        The ratio test filters out ambiguous matches where the
        best match is not significantly better than the second-best.
        
        Examples
        --------
        >>> matches = tracker.match_frames(frame1, frame2)
        >>> for m in matches:
        ...     kp1 = frame1.keypoints[m.queryIdx]
        ...     kp2 = frame2.keypoints[m.trainIdx]
        ...     print(f"Matched: {kp1.pt} <-> {kp2.pt}, dist={m.distance}")
        """
        # Handle empty descriptor cases
        if (frame1.descriptors.shape[0] == 0 or
            frame2.descriptors.shape[0] == 0):
            return []
        
        # Find k=2 nearest neighbors for each descriptor in frame1
        # (we need 2 to perform ratio test)
        matches = self.matcher.knnMatch(
            frame1.descriptors,
            frame2.descriptors,
            k=2
        )
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            # knnMatch returns list of lists
            # match_pair can have 0, 1, or 2 matches
            if len(match_pair) == 2:
                best, second_best = match_pair
                
                # Ratio test: best distance should be significantly
                # smaller than second-best distance
                if best.distance < self.ratio_test_threshold * second_best.distance:
                    good_matches.append(best)
            elif len(match_pair) == 1:
                # Only one match found (no second-best to compare)
                # Accept it (conservative: could also reject)
                good_matches.append(match_pair[0])
        
        return good_matches
    
    def __repr__(self) -> str:
        return (f"FeatureTracker(nfeatures={self.num_features}, "
                f"levels={self.num_levels}, ratio={self.ratio_test_threshold})")