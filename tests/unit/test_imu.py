from pynav.lib.states import SE23State
from pynav.lib.imu import (
    IMUState,
    IMU,
    IMUKinematics,
    N_matrix,
    RelativeIMUKinematics,
    U_matrix,
    U_matrix_inv,
    adjoint_IE3,
    G_matrix,
    G_matrix_inv,
    M_matrix,
)
from pylie import SE23, SO3
import numpy as np
from math import factorial

np.set_printoptions(precision=5, linewidth=200)


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


def test_left_jacobian_se23():
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [4, 5, 6], 0)
    x = SE23State(SE23.random(), 0, direction="left")
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


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


def test_right_jacobian_se23():
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = SE23State(SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, direction="right")
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_left_jacobian_imu():
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction="left",
    )
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-3)


def test_right_jacobian_imu():
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction="right",
    )
    jac = model.jacobian(x, u, dt)
    jac_fd = model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-3)


def test_imu_group_jacobian_right():
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction="right",
    )
    dx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0.2, 0.3, 4, 5, 6])
    jac = x.jacobian(dx)
    jac_fd = x.jacobian_fd(dx)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_imu_group_jacobian_left():
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        [0.1, 0.2, 0.3],
        [4, 5, 6],
        0,
        direction="left",
    )
    dx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0.2, 0.3, 4, 5, 6])
    jac = x.jacobian(dx)
    jac_fd = x.jacobian_fd(dx)
    assert np.allclose(jac, jac_fd, atol=1e-6)

def test_bias_jacobian_term():
    dt = 0.1
    phi = dt * np.array([[1, 1, 1]])
    a = np.array([4, 5, 6])

    def func(phi):
        J_inv = SO3.left_jacobian_inv(phi)
        N = N_matrix(phi)
        return J_inv @ N @ a.reshape((-1, 1))

    h = 1e-8

    # Finite difference just that term
    jac_test = np.zeros((3, 3))
    for i in range(3):
        dx = np.zeros((3,))
        dx[i] = h
        jac_test[:, i] = np.ravel((func(dt * (phi + dx)) - func(dt * phi)) / h)


    # Analytical approximation
    N = N_matrix(dt * phi)
    a = a.reshape((-1, 1))
    Om = SO3.wedge(phi * dt)
    A = SO3.wedge(a)
    jac2 = (
        -(1 / 360)
        * (dt**3)
        * (Om @ Om @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(Om @ Om @ a))
        + (1 / 6) * dt * A
    )
    assert(np.allclose(jac_test, jac2, atol=1e-3))

def test_relative_imu_equivalence():
    model = IMUKinematics(np.identity(6))
    dt = 0.1
    u1 = IMU([1, 2, 3], [2, 3, 1], 0, state_id=1)
    u2 = IMU([0.4,0.5,0.6], [1,1,1], 0, state_id=2)
    x1 = SE23State(SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]), state_id=1)
    x2 = SE23State(SE23.Exp(0.1*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),  state_id=1)
    T12 = SE23.inverse(x1.value) @ x2.value
    x12 = SE23State(T12.copy(), 0, direction="right")
    relmodel = RelativeIMUKinematics(np.identity(6), np.identity(6), id1=1, id2=2)
    for i in range(10):
        x1 = model.evaluate(x1, u1, dt)
        x2 = model.evaluate(x2, u2, dt)
        x12 = relmodel.evaluate(x12, u1, dt)
        x12 = relmodel.evaluate(x12, u2, dt)

    assert np.allclose(SE23.inverse(x1.value) @ x2.value, x12.value)

def test_relative_imu_jacobian():

    dt = 0.1
    u1 = IMU([1, 2, 3], [2, 3, 1], 0, state_id=1)
    u2 = IMU([0.4,0.5,0.6], [1,1,1], 0, state_id=2)
    x1 = SE23State(SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]), state_id=1)
    x2 = SE23State(SE23.Exp(0.1*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])), state_id=1)
    T12 = SE23.inverse(x1.value) @ x2.value
    x12 = SE23State(T12.copy(), 0, direction="right")
    relmodel = RelativeIMUKinematics(np.identity(6), np.identity(6), id1=1, id2=2)

    x12.direction="right"
    jac = relmodel.jacobian(x12, u1, dt)
    jac_fd = relmodel.jacobian_fd(x12, u1, dt)
    assert np.allclose(jac, jac_fd, atol=1e-8)

    jac = relmodel.jacobian(x12, u2, dt)
    jac_fd = relmodel.jacobian_fd(x12, u2, dt)
    assert np.allclose(jac, jac_fd, atol=1e-5)

    x12.direction="left"
    jac = relmodel.jacobian(x12, u1, dt)
    jac_fd = relmodel.jacobian_fd(x12, u1, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)

    jac = relmodel.jacobian(x12, u2, dt)
    jac_fd = relmodel.jacobian_fd(x12, u2, dt)
    assert np.allclose(jac, jac_fd, atol=1e-8)

if __name__ == "__main__":
    test_relative_imu_jacobian()
    print("All tests passed!")
