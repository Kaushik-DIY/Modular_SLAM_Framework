"""
=============================================================================
visual_slam/orbslam/local_features/feature_orbslam2.py

ORB2-compatible feature extractor wrapper.

Reference:
- pySLAM: pyslam/local_features/feature_orbslam2.py

pySLAM uses an external ORB-SLAM2 C++ ORBextractor. This self-contained port
uses OpenCV ORB while preserving the same class/module role. This is the only
deliberate dependency adaptation in this checkpoint.
=============================================================================
"""

from __future__ import annotations

import cv2
import numpy as np


class Orbslam2Feature2D:
    def __init__(
        self,
        num_features: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        deterministic: bool = False,
    ):
        self.num_features = int(num_features)
        self.scale_factor = float(scale_factor)
        self.num_levels = int(num_levels)
        self.ini_th_fast = int(ini_th_fast)
        self.min_th_fast = int(min_th_fast)
        self.deterministic = bool(deterministic)

        self._orb = cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.num_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.ini_th_fast,
        )

    def setMaxFeatures(self, num_features: int) -> None:
        self.num_features = int(num_features)
        self._orb.setMaxFeatures(self.num_features)

    def detect(self, image: np.ndarray, mask=None):
        return self._orb.detect(image, mask)

    def compute(self, image: np.ndarray, keypoints):
        return self._orb.compute(image, keypoints)

    def detectAndCompute(self, image: np.ndarray, mask=None):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(image, mask)
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        return keypoints, descriptors
