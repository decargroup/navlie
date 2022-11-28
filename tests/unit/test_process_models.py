from pynav.lib.models import SingleIntegrator, DoubleIntegrator, DoubleIntegratorWithBias
from pynav.lib.states import VectorState
from pynav.types import StampedValue
import numpy as np

def test_single_integrator_jacobian():
    u = StampedValue([1, 2, 3], 0)
    x = VectorState([4, 5, 6])
    model = SingleIntegrator(np.identity(3))
    jac = model.jacobian(x, u, 0.1)
    jac_test = model.jacobian_fd(x, u, 0.1)
    assert np.allclose(jac, jac_test)

def test_single_integrator_covariance():
    u = StampedValue([1, 2, 3], 0)
    x = VectorState([4, 5, 6])
    model = SingleIntegrator(np.identity(3))
    cov = model.covariance(x, u, 0.1)
    L = model.input_jacobian_fd(x, u, 0.1)
    cov_test = L @ np.identity(3) @ L.T
    assert np.allclose(cov, cov_test)

def test_double_integrator_jacobian():
    u = StampedValue([1, 2, 3], 0)
    x = VectorState([1, 2, 3, 4, 5, 6])
    model = DoubleIntegrator(np.identity(3))
    jac = model.jacobian(x, u, 0.1)
    jac_test = model.jacobian_fd(x, u, 0.1)
    assert np.allclose(jac, jac_test)

def test_double_integrator_covariance():
    u = StampedValue([1, 2, 3], 0)
    x = VectorState([1, 2, 3, 4, 5, 6])
    model = DoubleIntegrator(np.identity(3))
    cov = model.covariance(x, u, 0.1)
    L = model.input_jacobian_fd(x, u, 0.1)
    cov_test = L @ np.identity(3) @ L.T
    assert np.allclose(cov, cov_test)

def test_double_integrator_with_bias_jacobian():
    u = StampedValue([1, 2, 3], 0)
    x = VectorState([1, 2, 3, 4, 5, 6, 7, 8, 9])
    model = DoubleIntegratorWithBias(0.2*np.identity(6))
    jac = model.jacobian(x, u, 0.1)
    jac_test = model.jacobian_fd(x, u, 0.1)
    assert np.allclose(jac, jac_test)

def test_double_integrator_with_bias_covariance():
    u = StampedValue([1, 2, 3, 4, 5, 6], 0)
    x = VectorState([1, 2, 3, 4, 5, 6, 7, 8, 9])
    Q = 0.2*np.identity(6)
    model = DoubleIntegratorWithBias(Q)
    cov = model.covariance(x, u, 0.1)
    L = model.input_jacobian_fd(x, u, 0.1)
    cov_test = L @ (Q) @ L.T
    assert np.allclose(cov, cov_test)

if __name__ == "__main__":
    test_double_integrator_with_bias_covariance()