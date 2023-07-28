"""Tests for the process and measurement residuals found in batch.py"""

from navlie.types import StampedValue, Measurement
from navlie.lib.models import SingleIntegrator, BodyFrameVelocity, GlobalPosition
from navlie.batch.estimator import PriorResidual, ProcessResidual, MeasurementResidual
from navlie.lib.states import VectorState, SE3State
from pymlg import SE3
import numpy as np

def test_prior_residual_vector():
    # Test vector state
    x = VectorState(np.array([1, 2, 3]))
    prior_state = x.copy()
    prior_covariance = np.identity(x.dof)
    keys = ["p"]
    prior_residual = PriorResidual(keys, prior_state, prior_covariance)
    error = prior_residual.evaluate([x])
    assert np.allclose(error, 0)

def test_prior_residual_se3():
    # Test process error for SE3State
    x = SE3State(SE3.random())
    prior_state = x.copy()
    prior_covariance = np.identity(x.dof)
    keys = ["p"]
    prior_residual = PriorResidual([keys], prior_state, prior_covariance)
    error = prior_residual.evaluate([x])
    assert np.allclose(error, 0)

def test_process_residual_vector_state():
    # Test proccess error for vector state
    u = StampedValue([1, 2, 3], 0.0)
    x1 = VectorState([4, 5, 6], stamp=0.0)
    dt = 0.1
    model = SingleIntegrator(np.identity(3))
    x2 = model.evaluate(x1.copy(), u, dt)
    x2.stamp += dt
    process_residual = ProcessResidual([1, 2], model, u)
    error = process_residual.evaluate([x1, x2])
    assert np.allclose(error, 0)

def test_process_residual_se3_state():
    x1 = SE3State(
        SE3.random(),
        direction="left",
        stamp=0.0
    )
    u = StampedValue(np.array([1, 2, 3, 4, 5, 6]), stamp=0.0)
    dt = 0.1
    model = BodyFrameVelocity(np.identity(6))
    x2 = model.evaluate(x1.copy(), u, dt)
    x2.stamp += dt
    process_residual = ProcessResidual([1, 2], model, u)
    error = process_residual.evaluate([x1, x2])
    assert np.allclose(error, 0)

def test_measurement_residual():
    x = SE3State(
        SE3.random(), stamp=0.0, state_id=2, direction="left",
    )
    model = GlobalPosition(np.identity(3))
    y_k = model.evaluate(x)
    meas = Measurement(value=y_k, model=model, stamp = 0.0)

    meas_residual = MeasurementResidual([1], meas)
    error = meas_residual.evaluate([x])
    assert np.allclose(error, 0)

if __name__ == "__main__":
    test_measurement_residual()