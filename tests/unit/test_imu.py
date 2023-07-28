from navlie.lib.states import SE23State
from navlie.lib.imu import (
    IMUState,
    IMU,
    IMUKinematics,
    N_matrix,
    U_matrix,
    U_matrix_inv,
    adjoint_IE3,
    G_matrix,
    G_matrix_inv,
)
from pymlg import SE23, SO3
import numpy as np
from math import factorial
import pytest


np.set_printoptions(precision=5, suppress=True, linewidth=200)


def test_N_matrix():
    phi = np.array([1, 2, 3])
    N = N_matrix(phi)
    N_test = np.sum(
        [
            (2 / factorial(n + 2)) * np.linalg.matrix_power(SO3.wedge(phi), n)
            for n in range(100)
        ],
        axis=0,
    )
    assert np.allclose(N, N_test)


def test_U_matrix_inverse_se23():
    dt = 0.1
    u = IMU([1, 2, 3], [4, 5, 6], 0)
    U = U_matrix(u.gyro, u.accel, dt)
    U_inv = U_matrix_inv(u.gyro, u.accel, dt)
    U_inv_test = np.linalg.inv(U)
    assert np.allclose(U_inv, U_inv_test)
    assert np.allclose(U @ U_inv, np.eye(5))


def test_G_matrix_inverse_se23():
    g = np.array([0, 0, -9.81])
    dt = 0.1
    G = G_matrix(g, dt)
    G_inv = G_matrix_inv(g, dt)
    G_inv_test = np.linalg.inv(G)
    assert np.allclose(G_inv, G_inv_test)
    assert np.allclose(G.dot(G_inv), np.eye(5))




def test_U_adjoint_se23():
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    U = U_matrix(u.gyro, u.accel, dt)
    U_adj = adjoint_IE3(U)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(U_adj @ xi)
    U_inv = np.linalg.inv(U)
    test2 = U @ SE23.wedge(xi) @ U_inv
    assert np.allclose(test1, test2)


def test_U_adjoint_inv_se23():
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    U = U_matrix(u.gyro, u.accel, dt)
    U_inv = U_matrix_inv(u.gyro, u.accel, dt)

    U_inv_adj = adjoint_IE3(U_inv)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(U_inv_adj @ xi)
    test2 = U_inv @ SE23.wedge(xi) @ U
    assert np.allclose(test1, test2)


def test_G_adjoint_se23():
    g = np.array([0, 0, -9.81])
    dt = 0.1
    G = G_matrix(g, dt)
    G_adj = adjoint_IE3(G)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(G_adj @ xi)
    G_inv = np.linalg.inv(G)
    test2 = G @ SE23.wedge(xi) @ G_inv
    assert np.allclose(test1, test2)


def test_G_adjoint_inv_se23():
    g = np.array([0, 0, -9.81])
    dt = 0.1
    G = G_matrix(g, dt)
    G_inv = G_matrix_inv(g, dt)

    G_inv_adj = adjoint_IE3(G_inv)
    xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test1 = SE23.wedge(G_inv_adj @ xi)
    test2 = G_inv @ SE23.wedge(xi) @ G
    assert np.allclose(test1, test2)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_imu_kinematics_jacobian_se23(direction):
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [4, 5, 6], 0)
    x = SE23State(SE23.random(), 0, direction=direction)
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)

@pytest.mark.parametrize("direction", ["right", "left"])
def test_imu_kinematics_jacobian_imu(direction):
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction=direction,
    )
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-3)

@pytest.mark.parametrize("direction", ["right", "left"])
def test_imu_group_jacobian(direction):
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction=direction,
    )
    dx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0.2, 0.3, 4, 5, 6])
    jac = x.plus_jacobian(dx)
    jac_fd = x.plus_jacobian_fd(dx)
    assert np.allclose(jac, jac_fd, atol=1e-6)

if __name__ == "__main__":
    test_imu_kinematics_jacobian_imu("left")
    print("All tests passed!")
