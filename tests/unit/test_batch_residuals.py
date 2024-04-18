"""Tests for the process and measurement residuals found in batch.py"""

from navlie.types import Measurement
from navlie.lib import (
    SingleIntegrator,
    BodyFrameVelocity,
    GlobalPosition,
    VectorInput,
)
from navlie.batch.estimator import (
    PriorResidual,
    ProcessResidual,
    MeasurementResidual,
)
from navlie.lib.states import VectorState, SE3State
from pymlg import SE3
import numpy as np
from navlie.batch.gaussian_mixtures import (
    GaussianMixtureResidual,
    MaxMixtureResidual,
    SumMixtureResidual,
    MaxSumMixtureResidual,
)


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
    u = VectorInput([1, 2, 3], 0.0)
    x1 = VectorState([4, 5, 6], stamp=0.0)
    dt = 0.1
    model = SingleIntegrator(np.identity(3))
    x2 = model.evaluate(x1.copy(), u, dt)
    x2.stamp += dt
    process_residual = ProcessResidual([1, 2], model, u)
    error = process_residual.evaluate([x1, x2])
    assert np.allclose(error, 0)


def test_process_residual_se3_state():
    x1 = SE3State(SE3.random(), direction="left", stamp=0.0)
    u = VectorInput(np.array([1, 2, 3, 4, 5, 6]), stamp=0.0)
    dt = 0.1
    model = BodyFrameVelocity(np.identity(6))
    x2 = model.evaluate(x1.copy(), u, dt)
    x2.stamp += dt
    process_residual = ProcessResidual([1, 2], model, u)
    error = process_residual.evaluate([x1, x2])
    assert np.allclose(error, 0)


def test_measurement_residual():
    x = SE3State(
        SE3.random(),
        stamp=0.0,
        state_id=2,
        direction="left",
    )
    model = GlobalPosition(np.identity(3))
    y_k = model.evaluate(x)
    meas = Measurement(value=y_k, model=model, stamp=0.0)

    meas_residual = MeasurementResidual([1], meas)
    error = meas_residual.evaluate([x])
    assert np.allclose(error, 0)


def test_jacobians_mixtures():
    weights = [0.3, 0.3, 0.5]
    covariances = [(lv1 + 1) * np.eye(2) for lv1 in range(3)]
    means = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]
    dims = 2
    component_residuals = []
    for lv1 in range(len(means)):
        prior_state = VectorState(means[lv1])
        component_residuals.append(PriorResidual([None], prior_state, covariances[lv1]))

    resid_dict = {
        "Max-Mixture": MaxMixtureResidual(component_residuals, weights),
        "Sum-Mixture": SumMixtureResidual(component_residuals, weights),
        "Max-Sum-Mixture": MaxSumMixtureResidual(component_residuals, weights, 10),
    }
    n_points = 20
    test_values = n_points * np.random.rand(dims, n_points)
    test_values = [test_values[:, lv1] for lv1 in range(n_points)]

    for key in ["Max-Mixture", "Sum-Mixture", "Max-Sum-Mixture"]:
        res: GaussianMixtureResidual = resid_dict[key]
        for x in test_values:
            test_state = VectorState(np.array([x]))
            jac_fd = res.jacobian_fd([test_state])
            _, jac_list = res.evaluate([test_state], [True])

            assert np.linalg.norm((jac_list[0] - jac_fd[0]), "fro") < 1e-5


if __name__ == "__main__":
    test_jacobians_mixtures()
    test_measurement_residual()
