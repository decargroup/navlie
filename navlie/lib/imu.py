from pylie import SO3, SE23
import numpy as np
from navlie.types import ProcessModel, Input
from typing import Any, List, Tuple
from navlie.lib.states import CompositeState, VectorState, SE23State
from math import factorial


class IMU(Input):
    """
    Data container for an IMU reading.
    """

    def __init__(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        stamp: float,
        bias_gyro_walk: np.ndarray = [0, 0, 0],
        bias_accel_walk: np.ndarray = [0, 0, 0],
        state_id: Any = None,
        covariance: np.ndarray = None,
    ):
        super().__init__(dof=12, stamp=stamp, covariance=covariance)
        self.gyro = np.array(gyro).ravel()  #:np.ndarray: Gyro reading
        self.accel = np.array(
            accel
        ).ravel()  #:np.ndarray: Accelerometer reading

        if bias_accel_walk is None:
            bias_accel_walk = np.zeros((3, 1))
        else:
            #:np.ndarray: driving input for gyro bias random walk
            self.bias_gyro_walk = np.array(bias_gyro_walk).ravel()

        if bias_gyro_walk is None:
            bias_gyro_walk = np.zeros((3, 1))
        else:
            #:np.ndarray: driving input for accel bias random walk
            self.bias_accel_walk = np.array(bias_accel_walk).ravel()

        self.state_id = state_id  #:Any: State ID associated with the reading

    def plus(self, w: np.ndarray):
        """
        Modifies the IMU data. This is used to add noise to the IMU data.

        Parameters
        ----------
        w : np.ndarray with size 12
            w[0:3] is the gyro noise, w[3:6] is the accel noise,
            w[6:9] is the gyro bias walk noise, w[9:12] is the accel bias walk
            noise
        """
        new = self.copy()
        w = w.ravel()
        new.gyro = new.gyro + w[0:3]
        new.accel = new.accel + w[3:6]
        new.bias_gyro_walk = new.bias_gyro_walk + w[6:9]
        new.bias_accel_walk = new.bias_accel_walk + w[9:12]
        return new

    def copy(self):
        if self.covariance is None:
            cov_copy = None 
        else:
            cov_copy = self.covariance.copy()
        return IMU(
            self.gyro.copy(),
            self.accel.copy(),
            self.stamp,
            self.bias_gyro_walk.copy(),
            self.bias_accel_walk.copy(),
            self.state_id,
            cov_copy,
        )

    def __repr__(self):
        s = [
            f"IMU(stamp={self.stamp}, state_id={self.state_id})",
            f"    gyro: {self.gyro.ravel()}",
            f"    accel: {self.accel.ravel()}",
        ]

        if np.any(self.bias_accel_walk) or np.any(self.bias_gyro_walk):
            s.extend(
                [
                    f"    gyro_bias_walk: {self.bias_gyro_walk.ravel()}",
                    f"    accel_bias_walk: {self.bias_accel_walk.ravel()}",
                ]
            )

        return "\n".join(s)

    @staticmethod
    def random():
        return IMU(
            np.random.normal(size=3),
            np.random.normal(size=3),
            0.0,
            np.random.normal(size=3),
            np.random.normal(size=3),
        )


class IMUState(CompositeState):
    def __init__(
        self,
        nav_state: np.ndarray,
        bias_gyro: np.ndarray,
        bias_accel: np.ndarray,
        stamp: float = None,
        state_id: Any = None,
        direction="right",
    ):
        """
        Instantiate and IMUState object.

        Parameters
        ----------
        nav_state : np.ndarray with shape (5, 5)
            The navigation state stored as an element of SE_2(3).
            Contains orientation, velocity, and position.
        bias_gyro : np.ndarray with size 3
            Gyroscope bias
        bias_accel : np.ndarray with size 3
            Accelerometer bias
        stamp : float, optional
            Timestamp, by default None
        state_id : Any, optional
            Unique identifier, by default None
        direction : str, optional
            Direction of the perturbation for the nav state, by default "right"
        """
        nav_state = SE23State(nav_state, stamp, "pose", direction)
        bias_gyro = VectorState(bias_gyro, stamp, "gyro_bias")
        bias_accel = VectorState(bias_accel, stamp, "accel_bias")

        state_list = [nav_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)

        # Just for type hinting
        self.value: List[SE23State, VectorState, VectorState] = self.value

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0].attitude

    @attitude.setter
    def attitude(self, C: np.ndarray):
        self.value[0].attitude = C

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0].velocity

    @velocity.setter
    def velocity(self, v: np.ndarray):
        self.value[0].velocity = v

    @property
    def position(self) -> np.ndarray:
        return self.value[0].position

    @position.setter
    def position(self, r: np.ndarray):
        self.value[0].position = r

    @property
    def bias(self) -> np.ndarray:
        """Bias vector with in order [gyro_bias, accel_bias]"""
        return np.concatenate(
            [self.value[1].value.ravel(), self.value[2].value.ravel()]
        )

    @bias.setter
    def bias(self, new_bias: np.ndarray) -> np.ndarray:
        bias_gyro = new_bias[0:3]
        bias_accel = new_bias[3:6]
        self.value[1].value = bias_gyro
        self.value[2].value = bias_accel

    @property
    def bias_gyro(self) -> np.ndarray:
        return self.value[1].value

    @bias_gyro.setter
    def bias_gyro(self, gyro_bias: np.ndarray):
        self.value[1].value = gyro_bias.ravel()

    @property
    def bias_accel(self) -> np.ndarray:
        return self.value[2].value

    @bias_accel.setter
    def bias_accel(self, accel_bias: np.ndarray):
        self.value[2].value = accel_bias.ravel()

    @property
    def nav_state(self) -> np.ndarray:
        return self.value[0].value

    @property
    def pose(self) -> np.ndarray:
        return self.value[0].pose

    @pose.setter
    def pose(self, pose):
        self.value[0].pose = pose

    @property
    def direction(self) -> str:
        return self.value[0].direction

    @direction.setter
    def direction(self, direction: str) -> None:
        self.value[0].direction = direction

    def copy(self):
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
        return IMUState(
            self.nav_state.copy(),
            self.bias_gyro.copy(),
            self.bias_accel.copy(),
            self.stamp,
            self.state_id,
            self.direction,
        )

    def jacobian_from_blocks(
        self,
        attitude: np.ndarray = None,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        bias_gyro: np.ndarray = None,
        bias_accel: np.ndarray = None,
    ):
        for jac in [attitude, position, velocity, bias_gyro, bias_accel]:
            if jac is not None:
                dim = jac.shape[0]
                break

        nav_jacobian = self.value[0].jacobian_from_blocks(
            attitude=attitude, position=position, velocity=velocity
        )
        if bias_gyro is None:
            bias_gyro = np.zeros((dim, 3))
        if bias_accel is None:
            bias_accel = np.zeros((dim, 3))

        return np.hstack([nav_jacobian, bias_gyro, bias_accel])


def get_unbiased_imu(x: IMUState, u: IMU) -> IMU:
    """
    Removes bias from the measurement.

    Parameters
    ----------
    x : IMUState
        Contains the biases
    u : IMU
        IMU data correupted by bias

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        unbiased gyro and accelerometer measurements
    """

    u = u.copy()
    if hasattr(x, "bias_gyro"):
        u.gyro = u.gyro.ravel() - x.bias_gyro.ravel()
    if hasattr(x, "bias_accel"):
        u.accel = u.accel.ravel() - x.bias_accel.ravel()

    return u


def N_matrix(phi_vec: np.ndarray):
    """
    The N matrix from Barfoot 2nd edition, equation 9.211
    """
    if np.linalg.norm(phi_vec) < SO3._small_angle_tol:
        return np.identity(3)
    else:
        phi = np.linalg.norm(phi_vec)
        a = phi_vec / phi
        a = a.reshape((-1, 1))
        a_wedge = SO3.wedge(a)
        c = (1 - np.cos(phi)) / phi**2
        s = (phi - np.sin(phi)) / phi**2
        N = 2 * c * np.identity(3) + (1 - 2 * c) * (a @ a.T) + 2 * s * a_wedge
        return N


def M_matrix(phi_vec):
    phi_mat = SO3.wedge(phi_vec)
    M = np.sum(
        [
            (2 / factorial(n + 2)) * np.linalg.matrix_power(phi_mat, n)
            for n in range(100)
        ],
        axis=0,
    )
    return M


def adjoint_IE3(X):
    """
    Adjoint matrix of the "Incremental Euclidean Group".
    """
    R = X[:3, :3]
    c = X[3, 4]
    a = X[:3, 3].reshape((-1, 1))
    b = X[:3, 4].reshape((-1, 1))
    Ad = np.zeros((9, 9))
    Ad[:3, :3] = R
    Ad[3:6, :3] = SO3.wedge(a) @ R
    Ad[3:6, 3:6] = R

    Ad[6:9, :3] = -SO3.wedge(c * a - b) @ R
    Ad[6:9, 3:6] = -c * R
    Ad[6:9, 6:9] = R
    return Ad


def inverse_IE3(X):
    """
    Inverse matrix on the "Incremental Euclidean Group".
    """

    R = X[:3, :3]
    c = X[3, 4]
    a = X[:3, 3].reshape((-1, 1))
    b = X[:3, 4].reshape((-1, 1))
    X_inv = np.identity(5)
    X_inv[:3, :3] = R.T
    X_inv[:3, 3] = np.ravel(-R.T @ a)
    X_inv[:3, 4] = np.ravel(R.T @ (c * a - b))
    X_inv[3, 4] = np.ravel(-c)
    return X_inv


def U_matrix(omega, accel, dt: float):
    phi = omega * dt
    O = SO3.Exp(phi)
    J = SO3.left_jacobian(phi)
    a = accel.reshape((-1, 1))
    V = N_matrix(phi)
    U = np.identity(5)
    U[:3, :3] = O
    U[:3, 3] = np.ravel(dt * J @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * V @ a)
    U[3, 4] = dt
    return U


def U_tilde_matrix(omega, accel, dt: float):
    phi = omega * dt
    O = SO3.Exp(phi)
    J = SO3.left_jacobian(phi)
    a = accel.reshape((-1, 1))
    V = N_matrix(phi)
    U = np.identity(5)
    U[:3, :3] = O
    U[:3, 3] = np.ravel(dt * J @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * V @ a)
    return U


def delta_matrix(dt: float):
    U = np.identity(5)
    U[3, 4] = dt
    return U


def U_matrix_inv(omega, accel, dt: float):
    return inverse_IE3(U_matrix(omega, accel, dt))


def G_matrix(gravity, dt):
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G


def G_matrix_inv(gravity, dt):
    return inverse_IE3(G_matrix(gravity, dt))


def L_matrix(unbiased_gyro, unbiased_accel, dt: float) -> np.ndarray:
    """
    Computes the jacobian of the nav state with respect to the input.

    Since the noise and bias are both additive to the input, they have the
    same jacobians.
    """

    a = unbiased_accel
    om = unbiased_gyro
    omdt = om * dt
    J_att_inv_times_N = SO3.left_jacobian_inv(omdt) @ N_matrix(omdt)
    xi = np.zeros((9,))
    xi[:3] = dt * om
    xi[3:6] = dt * a
    xi[6:9] = (dt**2 / 2) * J_att_inv_times_N @ a
    J = SE23.left_jacobian(-xi)
    Om = SO3.wedge(omdt)
    OmOm = Om @ Om
    A = SO3.wedge(a)
    # See Barfoot 2nd edition, equation 9.247
    Up = dt * np.eye(9, 6)
    Up[6:9, 0:3] = (
        -0.5
        * (dt**2 / 2)
        * (
            (1 / 360)
            * (dt**3)
            * (OmOm @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(OmOm @ a))
            - (1 / 6) * dt * A
        )
    )
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N

    L = J @ Up
    return L


class IMUKinematics(ProcessModel):
    """
    The IMU Kinematics refer to the following continuous time model:

    .. math::

        \\dot{\mathbf{r}} &= \mathbf{v}

        \\dot{\mathbf{v}} &= \mathbf{C}\mathbf{a} +  \mathbf{g}

        \\dot{\mathbf{C}} &= \mathbf{C}\mathbf{\omega}^\wedge

    Using :math:`SE_2(3)` extended poses, it can be shown that the
    discrete-time IMU kinematics are given by:

    .. math::
        \mathbf{T}_{k} = \mathbf{G}_{k-1} \mathbf{T}_{k-1} \mathbf{U}_{k-1}

    where :math:`\mathbf{T}_{k}` is the pose at time :math:`k`, :math:`\mathbf{G}_{k-1}`
    is a matrix that depends on the gravity vector, and :math:`\mathbf{U}_{k-1}` is a matrix
    that depends on the IMU measurements.

    The :math:`\mathbf{G}_{k-1}` and :math:`\mathbf{U}_{k-1}` matrices are
    not quite elements of :math:`SE_2(3)`, but instead belong to a new group
    named here the "Incremental Euclidean Group" :math:`IE(3)`.

    """

    def __init__(self, Q: np.ndarray, gravity=None):
        """
        Parameters
        ----------
        Q : np.ndarray
            Discrete-time noise matrix.
        g_a : np.ndarray
            Gravity vector resolved in the inertial frame.
            If None, default value is set to [0; 0; -9.80665].
        """
        self._Q = Q

        if gravity is None:
            gravity = np.array([0, 0, -9.80665])

        self._gravity = np.array(gravity).ravel()

    def evaluate(self, x: IMUState, u: IMU, dt: float) -> IMUState:
        """
        Propagates an IMU state forward one timestep from an IMU measurement.

        The continuous-time IMU equations are discretized using the assumption
        that the IMU measurements are constant between two timesteps.

        Parameters
        ----------
        x : IMUState
            Current IMU state
        u : IMU
            IMU measurement,
        dt : float
            timestep.

        Returns
        -------
        IMUState
            Propagated IMUState.
        """
        x = x.copy()

        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)

        x.pose = G @ x.pose @ U

        # Propagate the biases forward in time using random walk
        if hasattr(u, "bias_gyro_walk") and hasattr(x, "bias_gyro"):
            x.bias_gyro = x.bias_gyro.ravel() + dt * u.bias_gyro_walk.ravel()

        if hasattr(u, "bias_accel_walk") and hasattr(x, "bias_accel"):
            x.bias_accel = x.bias_accel.ravel() + dt * u.bias_accel_walk.ravel()

        return x

    def jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        """
        Returns the Jacobian of the IMU kinematics model with respect
        to the full state
        """

        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U_inv = U_matrix_inv(u_no_bias.gyro, u_no_bias.accel, dt)

        # Jacobian of process model wrt to pose
        if x.direction == "right":
            jac_pose = adjoint_IE3(U_inv)
        elif x.direction == "left":
            jac_pose = adjoint_IE3(G)

        jac_kwargs = {}

        if hasattr(x, "bias_gyro"):
            # Jacobian of pose wrt to bias
            jac_bias = -self._get_input_jacobian(x, u, dt)

            # Jacobian of bias random walk wrt to pose
            jac_pose = np.vstack([jac_pose, np.zeros((6, jac_pose.shape[1]))])

            # Jacobian of bias random walk wrt to biases
            jac_bias = np.vstack([jac_bias, np.identity(6)])
            jac_gyro = jac_bias[:, :3]
            jac_accel = jac_bias[:, 3:6]
            jac_kwargs["bias_gyro"] = jac_gyro
            jac_kwargs["bias_accel"] = jac_accel

        jac_kwargs["attitude"] = jac_pose[:, :3]
        jac_kwargs["velocity"] = jac_pose[:, 3:6]
        jac_kwargs["position"] = jac_pose[:, 6:9]

        return x.jacobian_from_blocks(**jac_kwargs)

    def covariance(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        # Jacobian of pose wrt to noise
        L_pn = self._get_input_jacobian(x, u, dt)

        # Jacobian of bias random walk wrt to noise
        L_bn = np.zeros((6, 6))

        if hasattr(x, "bias_gyro"):
            # Jacobian of pose wrt to bias random walk
            L_pw = np.zeros((9, 6))

            # Jacobian of bias wrt to bias random walk
            L_bw = dt * np.identity(6)

            L = np.block([[L_pn, L_pw], [L_bn, L_bw]])

        else:
            L = np.hstack([[L_pn, L_bn]])

        return L @ self._Q @ L.T

    def _get_input_jacobian(self, x: IMUState, u: IMU, dt: float) -> np.ndarray:
        """
        Computes the jacobian of the nav state with respect to the input.

        Since the noise and bias are both additive to the input, they have the
        same jacobians.
        """
        # Get unbiased inputs
        u_no_bias = get_unbiased_imu(x, u)

        G = G_matrix(self._gravity, dt)
        U = U_matrix(u_no_bias.gyro, u_no_bias.accel, dt)
        L = L_matrix(u_no_bias.gyro, u_no_bias.accel, dt)

        if x.direction == "right":
            jac = L
        elif x.direction == "left":
            jac = SE23.adjoint(G @ x.pose @ U) @ L
        return jac
