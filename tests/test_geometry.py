import math

import pytest

from aegisrover.core.types import Pose2, Vec2, wrap_angle
from aegisrover.geometry.frames import chain, compose, inverse, relative, transform_point
from aegisrover.geometry.quaternion import from_yaw, rotate, to_yaw


def test_inverse_uses_transpose_of_forward_rotation():
    frame = Pose2(2.0, -1.0, 1.2)
    point = Vec2(1.5, -0.7)

    roundtrip = transform_point(inverse(frame), transform_point(frame, point))

    assert roundtrip.x == pytest.approx(point.x)
    assert roundtrip.y == pytest.approx(point.y)


def test_multilevel_frame_chain_round_trips_through_rotations():
    frames = [
        Pose2(1.0, 0.3, 0.5),
        Pose2(-0.4, 1.2, 1.7),
        Pose2(0.8, -0.6, -2.1),
    ]
    world = chain(frames)
    identity = compose(world, chain([inverse(f) for f in reversed(frames)]))

    assert identity.x == pytest.approx(0.0, abs=1e-12)
    assert identity.y == pytest.approx(0.0, abs=1e-12)
    assert wrap_angle(identity.yaw) == pytest.approx(0.0, abs=1e-12)


def test_relative_matches_inverse_pose_composition():
    parent = Pose2(3.0, -2.0, 0.8)
    child = Pose2(2.0, 1.0, 1.9)

    assert relative(parent, child) == compose(inverse(parent), child)
    assert compose(parent, relative(parent, child)).x == pytest.approx(child.x)
    assert compose(parent, relative(parent, child)).y == pytest.approx(child.y)
    assert wrap_angle(compose(parent, relative(parent, child)).yaw) == pytest.approx(
        wrap_angle(child.yaw), abs=1e-12
    )


def test_yaw_quaternion_and_vector_rotation_share_the_same_convention():
    for yaw in (0.0, 0.2, 1.1, -2.4, 3.0):
        q = from_yaw(yaw)

        assert wrap_angle(to_yaw(q) - yaw) == pytest.approx(0.0, abs=1e-12)

        x, y, z = rotate(q, (1.0, 0.0, 0.0))
        assert x == pytest.approx(math.cos(yaw), abs=1e-12)
        assert y == pytest.approx(math.sin(yaw), abs=1e-12)
        assert z == pytest.approx(0.0, abs=1e-12)
