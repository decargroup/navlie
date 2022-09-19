from pynav.lib.states import SE23State
from pynav.lib.models import Imu, ImuKinematics
from pylie import SE23
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def test_U_matrix_inverse():
    model = ImuKinematics()
    dt = 0.1
    u = Imu([1, 2, 3], [4, 5, 6], 0)
    U = model._U_matrix(u, dt)
    U_inv = model._U_matrix_inv(u, dt)
    U_inv_test = np.linalg.inv(U)
    assert np.allclose(U_inv, U_inv_test)
    assert np.allclose(U.dot(U_inv), np.eye(5))


def test_G_matrix_inverse():
    model = ImuKinematics()
    dt = 0.1
    G = model._G_matrix(dt)
    G_inv = model._G_matrix_inv(dt)
    G_inv_test = np.linalg.inv(G)
    assert np.allclose(G_inv, G_inv_test)
    assert np.allclose(G.dot(G_inv), np.eye(5))


def test_left_jacobian():
    model = ImuKinematics()
    dt = 0.1
    u = Imu([1, 2, 3], [4, 5, 6], 0)
    x = SE23State(SE23.random(), 0, direction="left")
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_U_adjoint():
    model = ImuKinematics()
    dt = 0.1
    u = Imu([1, 2, 3], [2, 3, 1], 0)
    U = model._U_matrix(u, dt)
    U_adj = model._adjoint_U(U)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(U_adj @ xi)
    U_inv = np.linalg.inv(U)
    test2 = U @ SE23.wedge(xi) @ U_inv
    assert np.allclose(test1, test2)


def test_U_adjoint_inv():
    model = ImuKinematics()
    dt = 0.1
    u = Imu([1, 2, 3], [2, 3, 1], 0)
    U = model._U_matrix(u, dt)
    U_inv = model._U_matrix_inv(u, dt)

    U_inv_adj = model._adjoint_U(U_inv)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(U_inv_adj @ xi)
    test2 = U_inv @ SE23.wedge(xi) @ U
    assert np.allclose(test1, test2)


def test_G_adjoint():
    model = ImuKinematics()
    dt = 0.1
    G = model._G_matrix(dt)
    G_adj = model._adjoint_U(G)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(G_adj @ xi)
    G_inv = np.linalg.inv(G)
    test2 = G @ SE23.wedge(xi) @ G_inv
    assert np.allclose(test1, test2)

def test_G_adjoint_inv():
    model = ImuKinematics()
    dt = 0.1
    G = model._G_matrix(dt)
    G_inv = model._G_matrix_inv(dt)

    G_inv_adj = model._adjoint_U(G_inv)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(G_inv_adj @ xi)
    test2 = G_inv @ SE23.wedge(xi) @ G
    assert np.allclose(test1, test2)

def test_right_jacobian():
    model = ImuKinematics()
    dt = 0.1
    u = Imu([1, 2, 3], [2,3,1], 0)
    x = SE23State(SE23.Exp([1,2,3,4,5,6,7,8,9]), 0, direction="right")
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


if __name__ == "__main__":
    test_right_jacobian()
    print("All tests passed!")
