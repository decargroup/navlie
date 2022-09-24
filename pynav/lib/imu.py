from pylie import SO2, SO3, SE2, SE3, SE23
from pylie.numpy.base import MatrixLieGroup
import numpy as np
from ..types import State, ProcessModel
from typing import Any, List, Tuple
from .states import CompositeState, VectorState, SE23State


class IMU:
    """
    Data container for an IMU reading.
    """

    def __init__(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        stamp: float,
        bias_gyro_walk=[0, 0, 0],
        bias_accel_walk=[0, 0, 0],
    ):
        self.gyro = np.array(gyro).ravel()
        self.accel = np.array(accel).ravel()
        self.bias_gyro_walk = np.array(bias_gyro_walk).ravel()
        self.bias_accel_walk = np.array(bias_accel_walk).ravel()
        self.stamp = stamp

    def plus(self, w: np.ndarray):
        w = w.ravel()
        self.gyro += w[0:3]
        self.accel += w[3:6]
        self.bias_gyro_walk += w[6:9]
        self.bias_accel_walk += w[9:12]

    def copy(self):
        return IMU(
            self.gyro.copy(),
            self.accel.copy(),
            self.stamp,
            self.bias_gyro_walk.copy(),
            self.bias_accel_walk.copy(),
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
        """Instantiate and IMUState object.

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
        nav_state = SE23State(nav_state, stamp, state_id, direction)
        bias_gyro = VectorState(bias_gyro, stamp, state_id)
        bias_accel = VectorState(bias_accel, stamp, state_id)

        state_list = [nav_state, bias_gyro, bias_accel]
        super().__init__(state_list, stamp, state_id)
        self.direction = direction

        # Just for type hinting
        self.value: List[SE23State, VectorState, VectorState] = self.value

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0].attitude

    @attitude.setter
    def attitude(self, C):
        self.value[0].attitude = C

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0].velocity

    @velocity.setter
    def velocity(self, v):
        self.value[0].velocity = v

    @property
    def position(self) -> np.ndarray:
        return self.value[0].position

    @position.setter
    def position(self, r):
        self.value[0].position = r

    @property
    def bias_gyro(self) -> np.ndarray:
        return self.value[1].value

    @bias_gyro.setter
    def bias_gyro(self, gyro_bias):
        self.value[1].value = gyro_bias.ravel()

    @property
    def bias_accel(self) -> np.ndarray:
        return self.value[2].value

    @bias_accel.setter
    def bias_accel(self, accel_bias):
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

    def plus(self, dx: np.ndarray, new_stamp: float = None):
        """
        Updates the value of each of the IMU state, given a perturbation dx.
        """
        if dx.shape[0] != 15:
            raise ValueError("Perturbation must be dimension 15!")

        for i, s in enumerate(self._slices):
            sub_dx = dx[s]
            self.value[i].plus(sub_dx)

        return self.copy()

    def minus(self, x: "IMUState") -> np.ndarray:
        dx = []
        for i, v in enumerate(x.value):
            dx.append(self.value[i].minus(x.value[i]).reshape((-1, 1)))

        return np.vstack(dx)

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


class IMUKinematics(ProcessModel):
    def __init__(self, Q: np.ndarray, g_a=[0, 0, -9.80665]):
        """Instantiate the IMUKinematic model.

        Parameters
        ----------
        Q : np.ndarray
            Discrete-time noise matrix.
        g_a : np.ndarray
            Gravity vector resolved in the inertial frame.
            If None, default value is set to [0; 0; -9.81].
        """
        self._Q = Q

        self._gravity = np.array(g_a).ravel()

    def evaluate(self, x: IMUState, u: IMU, dt: float) -> IMUState:
        """Propagates an IMU state forward one timestep from an IMU measurement.

        The continuous-time IMU equations are discretized using the assumption
        that the acceleration in the inertial frame is constant between two timesteps.

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

        # Get unbiased inputs
        unbiased_gyro, unbiased_accel = self._get_unbiased_imu(x, u)

        G = self._G_matrix(dt)
        U = self._U_matrix(unbiased_gyro, unbiased_accel, dt)

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
        unbiased_gyro, unbiased_accel = self._get_unbiased_imu(x, u)

        G = self._G_matrix(dt)
        U_inv = self._U_matrix_inv(unbiased_gyro, unbiased_accel, dt)

        # Jacobian of process model wrt to pose
        if x.direction == "right":
            jac_pose = self._adjoint_IE3(U_inv)
        elif x.direction == "left":
            jac_pose = self._adjoint_IE3(G)

        jac_kwargs = {}

        if hasattr(x, "bias_gyro"):
            jac_bias = self._get_input_jacobian(x, u, dt)

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
        unbiased_gyro, unbiased_accel = self._get_unbiased_imu(x, u)

        G = self._G_matrix(dt)
        U = self._U_matrix(unbiased_gyro, unbiased_accel, dt)

        a = unbiased_accel
        om = unbiased_gyro
        N = self._N_matrix(om * dt)
        J_att_inv = SO3.left_jacobian_inv(om * dt)
        xi = np.vstack(
            [
                dt * om,
                dt * a,
                (dt**2 / 2) * J_att_inv @ N @ a,
            ]
        )
        J = SE23.left_jacobian(-xi)

        # See Barfoot 2nd edition.
        # TODO: These jacobians seem to rely on a small dt assumption.
        # We will have to talk to barfoot
        XI = np.zeros((9, 6))
        XI[0:3, 0:3] = -dt * np.eye(3)
        XI[3:6, 3:6] = -dt * np.eye(3)
        XI[6:9, 0:3] = -(dt**3) * (
            SO3.wedge(N @ a) / 4 - J_att_inv @ SO3.wedge(a) / 6
        )
        XI[6:9, 3:6] = -(dt**2 / 2) * J_att_inv @ N

        jac_right = J @ XI
        if x.direction == "right":
            jac = jac_right
        elif x.direction == "left":
            jac = SE23.adjoint(G @ x.pose @ U) @ jac_right
        return jac

    def _get_unbiased_imu(self, x: IMUState, u: IMU) -> Tuple[np.ndarray, np.ndarray]:
        """Removes bias from the measurement.

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
        if hasattr(x, "bias_gyro"):
            bias_gyro = x.bias_gyro.reshape((-1, 1))
        else:
            bias_gyro = 0

        if hasattr(x, "bias_accel"):
            bias_accel = x.bias_accel.reshape((-1, 1))
        else:
            bias_accel = 0

        unbiased_gyro = u.gyro.reshape((-1, 1)) - bias_gyro
        unbiased_accel = u.accel.reshape((-1, 1)) - bias_accel

        return unbiased_gyro, unbiased_accel

    @staticmethod
    def _N_matrix(phi_vec: np.ndarray):
        if np.linalg.norm(phi_vec) < SO3._small_angle_tol:
            return np.identity(3)
        else:
            phi = np.linalg.norm(phi_vec)
            a = phi_vec / phi
            a = a.reshape((-1, 1))
            a_wedge = SO3.wedge(a)
            N = (
                2 * (1 - np.cos(phi)) / phi**2 * np.identity(3)
                + (1 - 2 * (1 - np.cos(phi)) / phi**2) * (a @ a.T)
                + 2 * ((phi - np.sin(phi)) / phi**2) * a_wedge
            )
            return N

    def _U_matrix(self, omega, accel, dt: float):
        phi = omega * dt
        O = SO3.Exp(phi)
        J = SO3.left_jacobian(phi)
        a = accel.reshape((-1, 1))
        V = self._N_matrix(phi)
        U = np.identity(5)
        U[:3, :3] = O
        U[:3, 3] = np.ravel(dt * J @ a)
        U[:3, 4] = np.ravel(dt**2 / 2 * V @ a)
        U[3, 4] = dt
        return U

    def _U_matrix_inv(self, omega, accel, dt: float):
        phi = omega * dt
        O = SO3.Exp(phi)
        V = self._N_matrix(phi)
        J = SO3.left_jacobian(phi)
        a = accel.reshape((-1, 1))
        U_inv = np.identity(5)
        U_inv[:3, :3] = O.T
        U_inv[:3, 3] = np.ravel(-dt * O.T @ J @ a)
        U_inv[:3, 4] = np.ravel(dt**2 * O.T @ (J - V / 2) @ a)
        U_inv[3, 4] = -dt
        return U_inv

    def _G_matrix(self, dt):
        G = np.identity(5)
        G[:3, 3] = dt * self._gravity
        G[:3, 4] = -0.5 * dt**2 * self._gravity
        G[3, 4] = -dt
        return G

    def _G_matrix_inv(self, dt):
        G_inv = np.identity(5)
        G_inv[:3, 3] = -dt * self._gravity
        G_inv[:3, 4] = -0.5 * dt**2 * self._gravity
        G_inv[3, 4] = dt
        return G_inv

    @staticmethod
    def _adjoint_IE3(X):
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

    @staticmethod
    def _inverse_IE3(X):
        """
        Inverse matrix on the "Incremental Euclidean Group".
        """

        R = X[:3, :3]
        c = X[3, 4]
        a = X[:3, 3].reshape((-1, 1))
        b = X[:3, 4].reshape((-1, 1))
        X_inv = np.identity(5)
        X_inv[:3, :3] = R.T
        X_inv[:3, 3] = -R.T @ a
        X_inv[:3, 4] = -R.T @ (c * a - b)
        X_inv[3, 4] = -c
        return X_inv

# TODO: consolidate and simplify some of the matrix functions. They will
# be needed for preintegration as well, so make them standalone functions
# in this module. 