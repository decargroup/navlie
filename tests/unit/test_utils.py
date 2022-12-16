from pynav.utils import jacobian
import numpy as np
import pytest
from pylie import SO3
from pynav.lib.states import SO3State



@pytest.mark.parametrize(
    "method, threshold", [("forward", 1e-6), ("central", 1e-10), ("cs", 1e-16)]
)
def test_jacobian_linear_numpy(method, threshold):

    x = np.array([1, 2]).reshape((-1,1))
    A = np.array([[1, 2], [3, 4]])

    def fun(x):
        return A @ x

    J_test = jacobian(fun, x, method=method)

    assert np.allclose(J_test, A, atol=threshold)



@pytest.mark.parametrize(
    "method, threshold", [("forward", 1e-6), ("central", 1e-10), ("cs", 1e-16)]
)
def test_jacobian_nonlinear_numpy(method, threshold):

    x = np.array([1, 2]).reshape((-1,1))
    A = np.array([[1, 2], [3, 4]])

    def fun(x):
        return 1/np.sqrt(x.T @ A.T @ A @ x)

    J_test = jacobian(fun, x, method=method)
    J_true = (- x.T @ A.T @ A)/((x.T @ A.T @ A @ x)**(3/2))

    assert np.allclose(J_test, J_true, atol=threshold)


def test_jacobian_so3():
    x = np.array([0.1, 0.2, 0.3])

    def fun(x):
        return SO3State(SO3.Exp(x), direction="left")

    J_test = jacobian(fun, x, method="forward")
    J_true = SO3.left_jacobian(x)
    assert np.allclose(J_test, J_true, atol=1e-6)

if __name__=="__main__":
    # just for debugging purposes
    test_jacobian_so3()