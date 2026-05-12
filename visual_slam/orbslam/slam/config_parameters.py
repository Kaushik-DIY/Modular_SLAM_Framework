"""
Central runtime parameters for the RGB-D SLAM pipeline.
This module collects feature, tracking, mapping, loop-closing, and optimization constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Collect the global constants used by tracking, mapping, and optimization.
class Parameters:
    # ================================================================
    # C++ core / runtime selection
    # ================================================================
    USE_CPP_CORE = False

    # ================================================================
    # Sparse SLAM threading
    # ================================================================
    kLocalMappingOnSeparateThread = False
    kTrackingWaitForLocalMappingToGetIdle = False
    kWaitForLocalMappingTimeout = 0.5
    kParallelLBAWaitIdleTimeout = 0.3

    # ================================================================
    # Feature management
    # ================================================================
    kNumFeatures = 2000
    kUseDynamicDesDistanceTh = True
    kUseDescriptorSigmaMadv2 = False

    kSigmaLevel0 = 1.0
    kFeatureMatchDefaultRatioTest = 0.7
    kKdtNmsRadius = 3
    kCheckFeaturesOrientation = True

    kORBNumLevels = 8
    kORBScaleFactor = 1.2
    kORBDeterministic = False
    kDescriptorSize = 32

    # ================================================================
    # Point triangulation / visibility
    # ================================================================
    kCosMaxParallaxInitializer = 0.99998
    kCosMaxParallax = 0.9998
    kMinRatioBaselineDepth = 0.01

    kViewingCosLimitForPoint = 0.5
    kScaleConsistencyFactor = 1.5
    kMaxDistanceToleranceFactor = 1.2
    kMinDistanceToleranceFactor = 0.8

    # ================================================================
    # Initializer
    # ================================================================
    kInitializerDesiredMedianDepth = 1
    kInitializerMinRatioDepthBaseline = 100
    kInitializerNumMinFeatures = 100
    kInitializerNumMinFeaturesStereo = 500
    kInitializerNumMinTriangulatedPoints = 150
    kInitializerNumMinTriangulatedPointsStereo = 100
    kInitializerFeatureMatchRatioTest = 0.9
    kInitializerNumMinNumPointsForPnPWithDepth = 15
    kInitializerUseCellCoverageCheck = True
    kInitializerUseMinFrameDistanceCheck = True

    # ================================================================
    # Tracking
    # ================================================================
    kUseMotionModel = True
    kUseSearchFrameByProjection = True
    kMinNumMatchedFeaturesSearchFrameByProjection = 20
    kUseEssentialMatrixFitting = False
    kMinNumMatchedFeaturesSearchReferenceFrame = 15
    kMaxNumOfKeyframesInLocalMap = 80
    kNumBestCovisibilityKeyFrames = 10
    kUseVisualOdometryPoints = True
    kMaxNumVisualOdometryPoints = 100
    kMaxNumStereoPointsOnNewKeyframe = 100
    kUseInterruptLocalMapping = False

    kMaxOutliersRatioInPoseOptimization = 0.9

    kUseMotionBlurDection = True
    kMotionBlurDetectionLalacianVarianceThreshold = 100.0
    kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography = 30

    # ================================================================
    # Keyframe generation
    # ================================================================
    kNumMinPointsForNewKf = 15
    kNumMinTrackedClosePointsForNewKfNonMonocular = 100
    kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70
    kThNewKfRefRatioMonocular = 0.9
    kThNewKfRefRatioStereo = 0.75
    kThNewKfRefRatioNonMonocular = 0.25
    kUseFeatureCoverageControlForNewKf = False
    kUseFovCentersBasedKfGeneration = False
    kMaxFovCentersDistanceForKfGeneration = 0.2

    # ================================================================
    # Keyframe culling
    # ================================================================
    kKeyframeCullingRedundantObsRatio = 0.9
    kKeyframeMaxTimeDistanceInSecForCulling = 0.5
    kKeyframeCullingMinNumPoints = 0

    # ================================================================
    # Stereo / RGB-D matching
    # ================================================================
    kStereoMatchingMaxRowDistance = 1.1
    kStereoMatchingShowMatchedPoints = False

    # ================================================================
    # Search matches by projection
    # ================================================================
    kMaxReprojectionDistanceFrame = 7
    kMaxReprojectionDistanceFrameNonStereo = 15
    kMaxReprojectionDistanceMap = 3
    kMaxReprojectionDistanceMapRgbd = 3
    kMaxReprojectionDistanceMapReloc = 5
    kMaxReprojectionDistanceFuse = 3
    kMaxReprojectionDistanceSim3 = 7.5

    kMatchRatioTestFrameByProjection = 0.9
    kMatchRatioTestMap = 0.8
    kMatchRatioTestEpipolarLine = 0.8

    kMaxDescriptorDistance = 0
    kMinDistanceFromEpipole = 10

    # ================================================================
    # Local Mapping
    # ================================================================
    kLocalMappingParallelKpsMatching = True
    kLocalMappingParallelKpsMatchingNumWorkers = 2
    kLocalMappingParallelFusePointsNumWorkers = 2
    kLocalMappingDebugAndPrintToFile = True
    kLocalMappingNumNeighborKeyFramesStereo = 10
    kLocalMappingNumNeighborKeyFramesMonocular = 20
    kLocalMappingTimeoutPopKeyframe = 0.5

    # ================================================================
    # Covisibility graph
    # ================================================================
    kMinNumOfCovisiblePointsForCreatingConnection = 15

    # ================================================================
    # Optimization engine
    # ================================================================
    kOptimizationAllUseGtsam = False
    kOptimizationFrontEndUseGtsam = False
    kOptimizationBundleAdjustUseGtsam = False
    kOptimizationLoopClosingUseGtsam = False

    # ================================================================
    # Bundle Adjustment
    # ================================================================
    kLocalBAWindowSize = 20
    kUseLargeWindowBA = False
    kEveryNumFramesLargeWindowBA = 10
    kLargeBAWindowSize = 20
    kUseParallelProcessLBA = False

    # ================================================================
    # Global Bundle Adjustment
    # ================================================================
    kUseGBA = False
    kGBADebugAndPrintToFile = True
    kGBAUseRobustKernel = True
    kGlobalBAIterations = 10
    kGlobalBAMinInlierEdges = 10

    # ================================================================
    # Loop closing
    # ================================================================
    kUseLoopClosing = True
    kMinDeltaFrameForMeaningfulLoopClosure = 10
    kMaxResultsForLoopClosure = 5
    kLoopDetectingTimeoutPopKeyframe = 0.5
    kLoopClosingDebugWithLoopDetectionImages = False
    kLoopClosingDebugWithSimmetryMatrix = True
    kLoopClosingDebugAndPrintToFile = True
    kLoopClosingDebugWithLoopConsistencyCheckImages = True
    kLoopClosingDebugShowLoopMatchedPoints = False
    kLoopClosingParallelKpsMatching = True
    kLoopClosingParallelKpsMatchingNumWorkers = 2
    kLoopClosingGeometryCheckerMinKpsMatches = 20
    kLoopClosingSE3GuidedMinSeedInliers = 8
    kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3 = 0.75
    kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3 = 45.0
    kLoopClosingSE3RansacMaxError = 0.25
    kLoopClosingSE3RansacIterations = 300
    kLoopClosingTh2 = 10
    kLoopClosingMaxReprojectionDistanceMapSearch = 10
    kLoopClosingMinNumMatchedMapPoints = 60
    kLoopClosingMaxReprojectionDistanceFuse = 4
    kLoopClosingFeatureMatchRatioTest = 0.75

    # information; this SE3 port keeps conservative non-identity weights to
    # distinguish structural, covisible, and loop constraints.
    kEssentialGraphSpanningTreeWeight = 1.0
    kEssentialGraphCovisibilityWeightScale = 0.01
    kEssentialGraphCovisibilityWeightMin = 0.5
    kEssentialGraphCovisibilityWeightMax = 5.0
    kEssentialGraphLoopEdgeWeight = 10.0

    # ================================================================
    # Relocalization
    # ================================================================
    kRelocalizationDebugAndPrintToFile = True
    kRelocalizationMinKpsMatches = 15
    kRelocalizationParallelKpsMatching = True
    kRelocalizationParallelKpsMatchingNumWorkers = 2
    kRelocalizationFeatureMatchRatioTest = 0.75
    kRelocalizationFeatureMatchRatioTestLarge = 0.9
    kRelocalizationPoseOpt1MinMatches = 10
    kRelocalizationDoPoseOpt2NumInliers = 50
    kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 10
    kRelocalizationMaxReprojectionDistanceMapSearchFine = 3

    # ================================================================
    # Common reprojection thresholds
    # ================================================================
    kChi2Mono = 5.991
    kChi2Stereo = 7.815
    kHuberMono = math.sqrt(kChi2Mono)
    kHuberStereo = math.sqrt(kChi2Stereo)
    kMinDepth = 1e-2

    # ================================================================
    # RGB-D helper defaults
    # ================================================================
    kDefaultRgbdBaselineMeters = 0.08


# Hold optional runtime overrides layered on top of the global parameters.
@dataclass
class OrbSlamSettings:
    """
    Small instance-level settings wrapper for runners.

    is only for future runner-level overrides.
    """

    sensor_type_name: str = "rgbd"
    num_features: int = Parameters.kNumFeatures
    num_levels: int = Parameters.kORBNumLevels
    scale_factor: float = Parameters.kORBScaleFactor
    deterministic: bool = Parameters.kORBDeterministic
    use_loop_closing: bool = Parameters.kUseLoopClosing
    use_local_mapping: bool = True
    use_relocalization: bool = True
