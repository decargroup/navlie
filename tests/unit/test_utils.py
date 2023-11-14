from navlie.utils import (
    GaussianResult,
    GaussianResultList,
    schedule_sequential_measurements,
)
from navlie.types import StateWithCovariance, Measurement
from navlie.lib.states import (
    SE23State,
    SE3State,
    SO3State,
)
from pymlg import SE23, SE3, SO3
import numpy as np

from navlie.utils import jacobian
from navlie.batch.residuals import Residual, MeasurementResidual
import numpy as np
import pytest
from navlie.lib.models import RangePoseToAnchor, GlobalPosition

def test_gaussian_result_indexing():
    # Construct a dummy GaussianResultList
    x_true = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    x_hat = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    cov = [(i + 1) * np.eye(9) for i in range(10)]
    x_cov = [StateWithCovariance(x, c) for x, c in zip(x_hat, cov)]
    gr = [GaussianResult(x, t) for x, t in zip(x_cov, x_true)]
    grl = GaussianResultList(gr)

    slc = 2
    grl_test = grl[:, slc]
    e_test = grl.error[:, slc]
    cov_test = grl.covariance[:, slc, slc]
    nees_test = [e**2 / c for e, c in zip(e_test, cov_test)]
    nees_test = np.array(nees_test).squeeze()
    assert np.all(grl_test.covariance == cov_test)
    assert np.all(grl_test.error == grl.error[:, slc])
    assert np.allclose(grl_test.nees, nees_test)


def test_gaussian_result_slicing():
    # Construct a dummy GaussianResultList
    x_true = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    x_hat = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    cov = [(i + 1) * np.eye(9) for i in range(10)]
    x_cov = [StateWithCovariance(x, c) for x, c in zip(x_hat, cov)]
    gr = [GaussianResult(x, t) for x, t in zip(x_cov, x_true)]
    grl = GaussianResultList(gr)

    slc = slice(0, 3)
    grl_test = grl[:, slc]
    e_test = grl.error[:, slc]
    cov_test = grl.covariance[:, slc, slc]
    nees_test = [
        e.reshape((1, -1)) @ np.linalg.inv(c) @ e.reshape((-1, 1))
        for e, c in zip(e_test, cov_test)
    ]
    nees_test = np.array(nees_test).squeeze()
    assert np.all(grl_test.covariance == cov_test)
    assert np.all(grl_test.error == grl.error[:, slc])
    assert np.allclose(grl_test.nees, nees_test)

def test_gaussian_result_list_slicing_equivalency():
    # Construct a dummy GaussianResultList
    x_true = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    x_hat = [SE23State(SE23.random(), stamp=i) for i in range(10)]
    cov = [(i + 1) * np.eye(9) for i in range(10)]
    x_cov = [StateWithCovariance(x, c) for x, c in zip(x_hat, cov)]
    gr = [GaussianResult(x, t) for x, t in zip(x_cov, x_true)]
    results = GaussianResultList(gr)    

    results[0:10] # returns the first 10 time steps
    results[:, 0] # returns the first degree of freedom
    results[0:10, 0] # returns the first degree of freedom for the first 10 time steps
    results[0:10, [0, 1]] # returns the first two degrees of freedom for the first 10 time steps
    results[:, 3:] # returns the all but the first three degrees of freedom
    
    assert np.all(results[0:10].error == results[0:10, :].error)
    assert np.all(results[:, [0,1,2]].error == results[:, 0:3].error)
    assert np.all(results[:, 3:].error == results[:, 3:9].error)


@pytest.mark.parametrize(
    "method, threshold", [("forward", 1e-6), ("central", 1e-10), ("cs", 1e-16)]
)
def test_jacobian_linear_numpy(method, threshold):
    x = np.array([1, 2]).reshape((-1, 1))
    A = np.array([[1, 2], [3, 4]])

    def fun(x):
        return A @ x

    J_test = jacobian(fun, x, method=method)

    assert np.allclose(J_test, A, atol=threshold)


@pytest.mark.parametrize(
    "method, threshold", [("forward", 1e-6), ("central", 1e-10), ("cs", 1e-16)]
)
def test_jacobian_nonlinear_numpy(method, threshold):
    x = np.array([1, 2]).reshape((-1, 1))
    A = np.array([[1, 2], [3, 4]])

    def fun(x):
        return 1 / np.sqrt(x.T @ A.T @ A @ x)

    J_test = jacobian(fun, x, method=method)
    J_true = (-x.T @ A.T @ A) / ((x.T @ A.T @ A @ x) ** (3 / 2))

    assert np.allclose(J_test, J_true, atol=threshold)


def test_jacobian_so3():
    x = np.array([0.1, 0.2, 0.3])

    def fun(x):
        return SO3State(SO3.Exp(x), direction="left")

    J_test = jacobian(fun, x, method="forward")
    J_true = SO3.left_jacobian(x)
    assert np.allclose(J_test, J_true, atol=1e-6)


def test_jacobian_x2():
    x = np.array([1.0, 2.0]).reshape((-1, 1))

    def fun(x):
        return x.T @ x

    J_test = jacobian(fun, x, method="cs", step_size=1e-15)

    assert np.allclose(J_test, 2 * x.T)


def test_meas_scheduling():
    x = SE3State(
        SE3.random(),
    )
    model = RangePoseToAnchor([1, 2, 0], [0.3, 0.1, 0], 1)
    model_list = [model] * 5

    freq = 50

    offset_list, new_freq = schedule_sequential_measurements(model_list, freq)

    assert new_freq == 10
    assert (np.array(offset_list) == np.array([0, 1, 2, 3, 4]) / freq).all()

def test_residual_jacobian_fd():
    # Test the finite difference for a measurement residual.

    meas_model = GlobalPosition(np.identity(3))
    measurement = Measurement(value=np.array([0.1, 0.2, 0.3]), stamp=0.0, model=meas_model)
    residual = MeasurementResidual(["pose"], measurement)

    # Create an SE(3) state
    se3_state = SE3State(value=SE3.random(), direction="right")
    jac_list = residual.jacobian_fd([se3_state])
    jac_numerical = jac_list[0]

    # Evaluate the Jacobian of the measurement model itself
    jac_analytical = meas_model.jacobian(se3_state)

    # For measurement residuals, the Jacobian of the residual should just be the 
    # negative of the analytical measurement model Jacobian
    assert (np.allclose(-jac_analytical, jac_numerical))



if __name__ == "__main__":
    # just for debugging purposes
    test_jacobian_x2()
