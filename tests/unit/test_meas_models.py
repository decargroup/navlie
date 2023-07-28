from navlie.lib.states import (
    MatrixLieGroupState,
    SO3State,
    SE2State,
    SE3State,
    SE23State,
    CompositeState,
)
from navlie.lib.imu import IMUState
from navlie.lib.models import (
    Altitude,
    GlobalPosition,
    InvariantMeasurement,
    Magnetometer,
    PointRelativePosition,
    RangePoseToAnchor,
    RangePoseToPose,
    Gravitometer,
    GlobalVelocity,
)
from navlie.types import Measurement, MeasurementModel
from pymlg import SO3, SE3, SE2, SE3, SE23
import numpy as np
import pytest

np.random.seed(0)


def _jacobian_test(
    x: MatrixLieGroupState, model: MeasurementModel, atol=1e-6, rtol=1e-4
):
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_anchor_se2(direction):
    x = SE2State(
        SE2.random(),
        direction=direction,
    )
    model = RangePoseToAnchor([1, 2], [0.3, 0.1], 1)
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_anchor_se3(direction):
    x = SE3State(
        SE3.random(),
        direction=direction,
    )
    model = RangePoseToAnchor([1, 2, 0], [0.3, 0.1, 0], 1)
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_anchor_se23(direction):
    x = SE23State(
        SE23.random(),
        direction=direction,
    )
    model = RangePoseToAnchor([1, 2, 0], [0.3, 0.1, 0], 1)
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_to_pose_se2(direction):
    T12 = SE2State(SE2.Exp([0, 0, 0]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([0, 1, 0]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17], [-0.17, 0.17]]

    x.value[0].direction = direction
    x.value[1].direction = direction
    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_to_pose_se3(direction):
    T12 = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, state_id=2)
    T13 = SE3State(SE3.Exp([0, 1, 0, 1, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17, 0], [-0.17, 0.17, 0]]

    x.value[0].direction = direction
    x.value[1].direction = direction
    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_range_pose_to_pose_se23(direction):
    T12 = SE23State(SE23.random(), stamp=0.0, state_id=2)
    T13 = SE23State(SE23.random(), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17, 0], [-0.17, 0.17, 0]]

    x.value[0].direction = direction
    x.value[1].direction = direction
    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_global_position_se2(direction):
    x = SE2State(
        SE2.Exp([0.5, 1, 2]), stamp=0.0, state_id=2, direction=direction
    )
    model = GlobalPosition(np.identity(3))
    _jacobian_test(x, model, atol=1e-5)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_global_position_se3(direction):
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction=direction
    )
    model = GlobalPosition(np.identity(3))
    _jacobian_test(x, model, atol=1e-5)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_global_position_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = GlobalPosition(np.identity(3))
    _jacobian_test(x, model, atol=1e-5)

@pytest.mark.parametrize("direction", ["right", "left"])
def test_global_velocity_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = GlobalVelocity(np.identity(3))
    _jacobian_test(x, model, atol=1e-5)

@pytest.mark.parametrize("direction", ["right", "left"])
def test_global_velocity_imu_state(direction):
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0, 0, 0],
        [1, 2, 3],
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = GlobalVelocity(np.identity(3))
    _jacobian_test(x, model, atol=1e-5)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_altitude_se3(direction):
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction=direction
    )
    model = Altitude(1)
    _jacobian_test(x, model, atol=1e-5)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_altitude_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = Altitude(1)
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_gravity_so3(direction):
    x = SO3State(
        SO3.Exp([0.3, 0.1, 0.2]), stamp=0.0, state_id=2, direction=direction
    )
    model = Gravitometer(np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_gravity_se3(direction):
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction=direction
    )
    model = Gravitometer(np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_gravity_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = Gravitometer(np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_magnetometer_so3(direction):
    x = SO3State(
        SO3.Exp([0.3, 0.1, 0.2]), stamp=0.0, state_id=2, direction=direction
    )
    model = Magnetometer(np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_magnetometer_se3(direction):
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction=direction
    )
    model = Magnetometer(np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_magnetometer_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = Magnetometer(np.identity(3))
    _jacobian_test(x, model)


def test_invariant_magnetometer_so3():
    x = SO3State(
        SO3.Exp([0.3, 0.1, 0.2]), stamp=0.0, state_id=2, direction="left"
    )

    b = [1, 0, 0]
    y = np.array(b)
    model = Magnetometer(np.identity(3), magnetic_vector=y)
    meas = InvariantMeasurement(Measurement(y, 0.0, model), direction="right")
    jac = meas.model.jacobian(x)
    jac_test = -SO3.odot(b)
    assert np.allclose(jac, jac_test, atol=1e-6)


def test_invariant_magnetometer_se3():
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction="left"
    )

    b = [1, 0, 0]
    y = np.array(b)
    model = Magnetometer(np.identity(3), magnetic_vector=y)
    meas = InvariantMeasurement(Measurement(y, 0.0, model), direction="right")
    jac = meas.model.jacobian(x)
    jac_test = np.hstack((-SO3.odot(b), np.zeros((3, 3))))
    assert np.allclose(jac, jac_test, atol=1e-6)


def test_invariant_magnetometer_se23():
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction="left",
    )

    b = [1, 0, 0]
    y = np.array(b)
    model = Magnetometer(np.identity(3), magnetic_vector=y)
    meas = InvariantMeasurement(Measurement(y, 0.0, model), direction="right")

    jac = meas.model.jacobian(x)
    jac_test = np.hstack((-SO3.odot(b), np.zeros((3, 6))))
    assert np.allclose(jac, jac_test, atol=1e-6)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_landmark_relative_position_se3(direction):
    x = SE3State(
        SE3.Exp([0, 1, 2, 4, 5, 6]), stamp=0.0, state_id=2, direction=direction
    )
    model = PointRelativePosition([1, 2, 3], np.identity(3))
    _jacobian_test(x, model)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_landmark_relative_position_se23(direction):
    x = SE23State(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        stamp=0.0,
        state_id=2,
        direction=direction,
    )
    model = PointRelativePosition([1, 2, 3], np.identity(3))
    _jacobian_test(x, model)


if __name__ == "__main__":
    test_range_pose_to_pose_se3()
