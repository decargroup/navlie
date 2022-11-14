import numpy as np
import pynav.utils as utils

from scipy.linalg import expm

np.set_printoptions(precision=5, suppress=True)
np.random.seed(0)


def test_van_loans():
    N = 3
    A_c = np.random.rand(N, N)
    L_c = np.random.rand(N, N)
    Q_c = np.identity(N)
    dt = 0.1

    A_d, Q_d = utils.van_loans(A_c, L_c, Q_c, dt)

    # Compare to A_d computed through matrix exponential
    assert np.allclose(A_d, expm(A_c * dt))

def test_van_loan_double_integrator():
    A_c = np.array([[0 ,1],[0, 0]])
    L_c = np.array([0,1]).reshape((-1,1))
    Q_c = np.array([1])
    dt = 0.1

    A_d, Q_d = utils.van_loans(A_c, L_c, Q_c, dt)

    # Compare to analytical solution
    A_d_test = np.array([[1, dt], [0, 1]])
    Q_d_test = np.array([[1/3*dt**3, 1/2*dt**2],[1/2*dt**2, dt]])

    
    assert np.allclose(A_d, A_d_test)
    assert np.allclose(Q_d, Q_d_test)


if __name__ == "__main__":
    test_van_loan_double_integrator()
