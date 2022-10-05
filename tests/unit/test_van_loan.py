import numpy as np
import pynav.utils as utils

from scipy.linalg import expm

np.set_printoptions(precision=2, suppress=True)
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


if __name__ == "__main__":
    test_van_loans()
