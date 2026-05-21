from visual_slam.orbslam.slam import (
    DatasetEnvironmentType,
    DatasetType,
    OrbSlamSettings,
    Parameters,
    SensorType,
    SlamState,
    get_sensor_type,
    is_depth_available,
    is_monocular,
    is_rgbd,
    is_stereo,
)


def test_sensor_type_values_match_pyslam():
    assert SensorType.MONOCULAR.value == 0
    assert SensorType.STEREO.value == 1
    assert SensorType.RGBD.value == 2

    assert DatasetEnvironmentType.INDOOR.value == 1
    assert DatasetEnvironmentType.OUTDOOR.value == 2

    assert DatasetType.TUM.value == 3


def test_sensor_type_helpers_match_pyslam_behavior():
    assert get_sensor_type("mono") == SensorType.MONOCULAR
    assert get_sensor_type("monocular") == SensorType.MONOCULAR
    assert get_sensor_type("stereo") == SensorType.STEREO
    assert get_sensor_type("rgbd") == SensorType.RGBD
    assert get_sensor_type("unknown") == SensorType.MONOCULAR

    assert is_monocular(SensorType.MONOCULAR)
    assert is_stereo(SensorType.STEREO)
    assert is_rgbd(SensorType.RGBD)
    assert is_depth_available(SensorType.STEREO)
    assert is_depth_available(SensorType.RGBD)
    assert not is_depth_available(SensorType.MONOCULAR)


def test_slam_state_values_match_pyslam():
    assert SlamState.NO_IMAGES_YET.value == 0
    assert SlamState.NOT_INITIALIZED.value == 1
    assert SlamState.OK.value == 2
    assert SlamState.LOST.value == 3
    assert SlamState.RELOCALIZE.value == 4
    assert SlamState.INIT_RELOCALIZE.value == 5


def test_parameters_match_pyslam_orb_rgbd_subset():
    assert Parameters.kNumFeatures == 2000
    assert Parameters.kSigmaLevel0 == 1.0
    assert Parameters.kFeatureMatchDefaultRatioTest == 0.7

    assert Parameters.kORBNumLevels == 8
    assert abs(Parameters.kORBScaleFactor - 1.2) < 1e-12
    assert Parameters.kORBDeterministic is True

    assert Parameters.kMinNumMatchedFeaturesSearchFrameByProjection == 20
    assert Parameters.kNumMinPointsForNewKf == 15
    assert Parameters.kNumMinTrackedClosePointsForNewKfNonMonocular == 100
    assert Parameters.kNumMaxNonTrackedClosePointsForNewKfNonMonocular == 70
    assert Parameters.kThNewKfRefRatioStereo == 0.90

    assert Parameters.kMaxReprojectionDistanceFrame == 7
    assert Parameters.kMaxReprojectionDistanceMapRgbd == 3
    assert Parameters.kLocalBAWindowSize == 20
    assert Parameters.kLoopClosingGeometryCheckerMinKpsMatches == 9
    assert Parameters.kRelocalizationMinKpsMatches == 15

    assert Parameters.kChi2Mono == 5.991
    assert Parameters.kChi2Stereo == 7.815
    assert Parameters.kMinDepth == 1e-2


def test_settings_wrapper_defaults():
    settings = OrbSlamSettings()
    assert settings.sensor_type_name == "rgbd"
    assert settings.num_features == Parameters.kNumFeatures
    assert settings.num_levels == Parameters.kORBNumLevels
    assert settings.scale_factor == Parameters.kORBScaleFactor
    assert settings.use_loop_closing is True
