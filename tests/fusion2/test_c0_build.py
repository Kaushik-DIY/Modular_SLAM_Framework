"""C0 — fusion_core module builds, imports, and coexists with sibling pybind modules."""
import math

import pytest

fusion_core = pytest.importorskip("fusion_core")


def test_hello_and_version():
    assert fusion_core.hello().startswith("fusion_core ")
    assert fusion_core.__version__


def test_pose2_compose_inverse():
    p = fusion_core.Pose2(1.0, 2.0, math.pi / 2)
    ident = p.compose(p.inverse())
    assert abs(ident.x) < 1e-12
    assert abs(ident.y) < 1e-12
    assert abs(ident.theta) < 1e-12


def test_g2o_se2_roundtrip():
    p = fusion_core.g2o_se2_roundtrip(0.5, -0.25, 0.3)
    assert abs(p.x - 0.5) < 1e-12
    assert abs(p.y + 0.25) < 1e-12
    assert abs(p.theta - 0.3) < 1e-12


def test_coexists_with_sibling_pybind_modules():
    # The C0 risk check: fusion_core (static g2o) + g2o pybind + slam_optimizer_core
    # (static g2o) + cpp_slam_core all loaded in ONE process without symbol clashes.
    import g2o  # noqa: F401
    import slam_optimizer_core  # noqa: F401
    import cpp_slam_core  # noqa: F401

    p = fusion_core.g2o_se2_roundtrip(1.0, 1.0, 1.0)
    assert abs(p.x - 1.0) < 1e-12
