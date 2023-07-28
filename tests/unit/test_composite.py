from navlie.lib.states import SE2State, CompositeState, VectorState
from navlie.types import StampedValue
from navlie.lib.models import (
    BodyFrameVelocity,
    CompositeProcessModel,
    RangeRelativePose,
    CompositeInput,
)
from pymlg import SE2
import numpy as np
import pickle
import os


def test_composite_process_model():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    Q = np.diag([0.1**2, 0.1**2, 0.001**2])
    x0 = CompositeState([T12, T13])
    process_model = CompositeProcessModel(
        [BodyFrameVelocity(Q), BodyFrameVelocity(Q)]
    )
    u = CompositeInput(
        [
            StampedValue(np.array([0.3, 1, 0]), 1),
            StampedValue(np.array([-0.3, 2, 0]), 1),
        ]
    )
    dt = 1
    jac = process_model.jacobian(x0, u, dt)
    jac_fd = process_model.jacobian_fd(x0, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-6)

def test_shared_input():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    Q = np.diag([0.1**2, 0.1**2, 0.001**2])
    X = CompositeState([T12, T13])
    x0 = CompositeState([X, X])
    composite_process_model = CompositeProcessModel([BodyFrameVelocity(Q), BodyFrameVelocity(Q)])
    process_model = CompositeProcessModel(
        [composite_process_model, composite_process_model],
        shared_input = True,
    )
    u = CompositeInput(
        [
            StampedValue(np.array([0.3, 1, 0]), 1),
            StampedValue(np.array([0.3, 1, 0]), 1),
        ]
    )
    dt = 1
    jac = process_model.jacobian(x0, u, dt)
    jac_fd = process_model.jacobian_fd(x0, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-6)

def test_composite_plus_jacobian():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    dx = np.array([i for i in range(x.dof)])
    jac = x.plus_jacobian(dx)
    jac_fd = x.plus_jacobian_fd(dx)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_composite_minus_jacobian():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x1 = CompositeState([T12, T13])
    x2 = CompositeState([T13, T12])
    jac = x1.minus_jacobian(x2)
    jac_fd = x1.minus_jacobian_fd(x2)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_relative_pose():

    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    my_tag = [0.17, 0.17]
    tag = [-0.16, 0.2]
    model = RangeRelativePose(my_tag, tag, T12.state_id, 0.1**2)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_composite_pickling():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    with open("test.pkl", "wb") as f:
        pickle.dump(x, f)

    with open("test.pkl", "rb") as f:
        y = pickle.load(f)

    os.remove("test.pkl")


def test_composite_add_and_remove_state():
    state_list = [
        SE2State(SE2.Exp([0.1, 0.2, 0.3]), stamp=0.0, state_id="p0"),
        VectorState(np.array([0.1, 0.2, 0.3]), stamp=0.0, state_id="l1"),
    ]

    state = CompositeState(state_list, stamp=0.0)

    new_state = VectorState(np.array([0.1, 0.2, 0.3]), stamp=0.0, state_id="l2")
    state.add_state(new_state)
    state.remove_state_by_id("l1")

    assert state.value[1].state_id == "l2"


def test_matrix_blocks_composite():
    state_list = [
        SE2State(SE2.Exp([0.1, 0.2, 0.3]), stamp=0.0, state_id="p0"),
        VectorState(np.array([0.1, 0.2, 0.3]), stamp=0.0, state_id="l1"),
    ]

    state = CompositeState(state_list, stamp=0.0)
    cov = np.random.rand(6, 6)

    cov_block_1 = state.get_matrix_block_by_ids(cov, "p0")
    cov_block_2 = state.get_matrix_block_by_ids(cov, "p0", "l1")
    cov_block_3 = state.get_matrix_block_by_ids(cov, "l1")

    assert np.allclose(cov_block_1, cov[0:3, 0:3])
    assert np.allclose(cov_block_2, cov[0:3, 3:6])
    assert np.allclose(cov_block_3, cov[3:6, 3:6])


if __name__ == "__main__":
    test_composite_minus_jacobian()
