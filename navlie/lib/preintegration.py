from abc import ABC, abstractmethod
from typing import List, Any, Callable, Tuple
from navlie.types import StampedValue, ProcessModel, Input
from navlie.lib.imu import (
    IMU,
    IMUState,
    U_matrix,
    adjoint_IE3,
    inverse_IE3,
    L_matrix,
    G_matrix,
)
from navlie.lib.states import MatrixLieGroupState, VectorState
import numpy as np
from pylie import SO3, SE3, SE2, SE23, MatrixLieGroup


class RelativeMotionIncrement(Input):
    __slots__ = ["stamps", "covariance", "state_id"]

    def __init__(self, dof: int):
        #:int: Degrees of freedom of the RMI
        self.dof = dof

        #:List[float, float]: the two timestamps i, j associated with the RMI
        self.stamps = [None, None]

        #:np.ndarray: the covariance matrix of the RMI
        self.covariance = None

        #:Any: an ID associated with the RMI
        self.state_id = None

        self.value = None

    @property
    def stamp(self):
        """
        The later timestamp :math:`j` of the RMI.
        """
        return self.stamps[1]

    @abstractmethod
    def increment(self, u, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        pass

    @abstractmethod
    def new(self) -> "RelativeMotionIncrement":
        pass

    def symmetrize(self):
        """
        Symmetrize the covariance matrix of the RMI.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update_bias(self, new_bias):
        """
        Update the bias of the RMI such that the new `.value` attribute of this
        RMI instance incorporates the new bias values.
        """
        raise NotImplementedError()


class IMUIncrement(RelativeMotionIncrement):
    __slots__ = [
        "original_value",
        "original_bias",
        "new_bias",
        "stamps",
        "covariance",
        "input_covariance",
        "bias_jacobian",
        "gravity",
        "state_id",
        "_estimating_bias",
    ]

    def __init__(
        self,
        input_covariance: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        state_id: Any = None,
        gravity=None,
    ):
        """
        Initializes an "identity" IMU RMI.

        Parameters
        ----------
        input_covariance : np.ndarray with shape (12, 12) or (6, 6)
            If a 6 x 6 array is provided, this is the covariance of gyro, accel
            measurements. If a 12 x 12 array is provided, this is the covariance
            of gyro, accel, gyro bias random walk, and accel bias random walk.
        """
        # TODO: add tests for when (6,6) input covariance is given
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])

        if input_covariance.shape[0] == 12:
            self.dof = 15
            self._estimating_bias = True
            self.covariance = np.zeros((15, 15))
        elif input_covariance.shape[0] == 6:
            super().__init__(dof=9)
            self.dof = 9
            self._estimating_bias = False
            self.covariance = np.zeros((9, 9))
        else:
            raise ValueError("Input covariance must be 12x12 or 6x6")

        self.original_value = np.identity(5)
        self.original_bias = np.hstack(
            [np.array(gyro_bias).ravel(), np.array(accel_bias).ravel()]
        )
        self.stamps = [None, None]
        self.input_covariance = input_covariance
        self.bias_jacobian = np.zeros((9, 6))
        self.gravity = np.array(gravity).ravel()
        self.state_id = state_id
        self.new_bias = self.original_bias

    @property
    def gyro_bias(self):
        return self.original_bias[:3]

    @property
    def accel_bias(self):
        return self.original_bias[3:]

    @property
    def value(self):
        db = self.new_bias.reshape((-1, 1)) - self.original_bias.reshape(
            (-1, 1)
        )
        return self.original_value @ SE23.Exp(self.bias_jacobian @ db)

    def increment(self, u: IMU, dt: float):
        u = u.copy()
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_gyro = u.gyro - self.gyro_bias
        unbiased_accel = u.accel - self.accel_bias

        U = U_matrix(unbiased_gyro, unbiased_accel, dt)
        self.original_value = self.original_value @ U

        U_inv = inverse_IE3(U)
        A = adjoint_IE3(U_inv)
        L = L_matrix(unbiased_gyro, unbiased_accel, dt)
        Q = self.input_covariance

        A_full = np.zeros((15, 15))
        A_full[0:9, 0:9] = A
        A_full[0:9, 9:15] = -L
        A_full[9:15, 9:15] = np.identity(6)

        L_full = np.zeros((15, 12))
        L_full[0:9, 0:6] = L
        L_full[9:15, 6:12] = dt * np.identity(6)

        if not self._estimating_bias:
            A_full = A_full[0:9, 9:9]
            L_full = L_full[0:9, 0:6]

        self.covariance = (
            A_full @ self.covariance @ A_full.T + L_full @ Q @ L_full.T
        )
        self.bias_jacobian = A @ self.bias_jacobian - L
        self.symmetrize()

    def update_bias(self, new_bias: np.ndarray):
        """
        Updates the RMI given new bias values

        Parameters
        ----------
        new_gyro_bias: np.ndarray with size 3
            new gyro bias value
        new_accel_bias: np.ndarray with size 3
            new accel bias value
        """
        # TODO. this needs some tests.
        # TODO. bias jacobian also needs some tests.
        if self.original_bias is None:
            raise ValueError("Cannot update bias of RMI without original bias")

        self.new_bias = new_bias

    def plus(self, w: np.ndarray) -> "IMUIncrement":
        """
        Adds noise to the RMI

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        RelativeMotionIncrement
            The updated RMI
        """
        new = self.copy()
        new.original_value = new.original_value @ SE23.Exp(w[0:9])
        return new

    def copy(self) -> "IMUIncrement":
        """
        Returns
        -------
        IMUIncrement
            A copy of the RMI
        """
        new = IMUIncrement(
            self.input_covariance,
            self.gyro_bias,
            self.accel_bias,
            self.state_id,
            self.gravity,
        )
        new.original_value = self.original_value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

    def new(
        self, new_bias: np.ndarray = None, gyro_bias=None, accel_bias=None
    ) -> "IMUIncrement":
        """
        Parameters
        ----------
        new_bias : np.ndarray
            The new bias value stacked as [gyro_bias, accel_bias]
        gyro_bias : np.ndarray
            The new gyro bias to use in this RMI (overwrites previous argument)
        accel_bias : np.ndarray
            The new accel bias to use in this RMI (overwrites previous argument)

        Returns
        -------
        IMUIncrement
            A copy of the RMI with reinitialized values
        """
        gyro_bias = self.gyro_bias
        accel_bias = self.accel_bias

        if new_bias is not None:
            gyro_bias = new_bias[:3]
            accel_bias = new_bias[3:]

        if gyro_bias is not None:
            gyro_bias = self.gyro_bias

        if accel_bias is not None:
            accel_bias = self.accel_bias

        new = IMUIncrement(
            self.input_covariance,
            gyro_bias,
            accel_bias,
            self.state_id,
            self.gravity,
        )
        return new


class PreintegratedIMUKinematics(ProcessModel):
    def __init__(self, gravity=None):
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])
        self.gravity = gravity

    def evaluate(self, x: IMUState, rmi: IMUIncrement, dt=None) -> IMUState:
        x = x.copy()
        rmi = rmi.copy()

        rmi.update_bias(x.bias)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        x.pose = DG @ x.pose @ DU
        return x

    def jacobian(self, x: IMUState, rmi: IMUIncrement, dt=None) -> np.ndarray:
        x = x.copy()
        rmi = rmi.copy()
        rmi.update_bias(x.bias)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        J = SE23.right_jacobian(
            rmi.bias_jacobian @ (x.bias - rmi.original_bias)
        )
        A = np.identity(15)
        if x.direction == "right":
            A[0:9, 0:9] = adjoint_IE3(inverse_IE3(DU))
            A[0:9, 9:15] = J @ rmi.bias_jacobian
        elif x.direction == "left":
            Ad = SE23.adjoint(DG @ x.pose @ DU)
            A[0:9, 0:9] = adjoint_IE3(DG)
            A[0:9, 9:15] = Ad @ J @ rmi.bias_jacobian
        return A

    def covariance(self, x: IMUState, rmi: IMUIncrement, dt=None) -> np.ndarray:
        x = x.copy()
        rmi = rmi.copy()
        rmi.update_bias(x.bias)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        update_bias_vec = rmi.bias_jacobian @ (x.bias - rmi.original_bias)

        if x.direction == "right":
            L = np.identity(15)
            L[0:9, 0:9] = SE23.adjoint(SE23.Exp(-update_bias_vec))
        elif x.direction == "left":
            Ad = SE23.adjoint(DG @ x.pose @ DU)
            L = np.identity(15)
            L[0:9, 0:9] = Ad @ SE23.adjoint(SE23.Exp(-update_bias_vec))
        return L @ rmi.covariance @ L.T


class BodyVelocityIncrement(RelativeMotionIncrement):
    """
    This is a general preintegration class for any process model of the form

    .. math::
        \mathbf{T}_{k} = \mathbf{T}_{k-1} \exp(\Delta t \mathbf{u}_{k-1}^\wedge).

    Preintegration is trivially done with

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{U}_{ij},

    where :math:`\Delta \mathbf{U}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{U}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{u}_{k}^\wedge).
    """
    # TODO. add bias update tests
    # TODO. add ability to also propagate bias uncertainty. 
    def __init__(
        self,
        group: MatrixLieGroup,
        Q: np.ndarray,
        bias: np.ndarray = None,
        state_id=None,
    ):
        self.dof = group.dof
        self.bias = bias
        self.group = group
        self.covariance = np.zeros((group.dof, group.dof))
        self.input_covariance = Q
        self.original_value: np.ndarray = group.identity()
        self.bias_jacobian = np.zeros((group.dof, group.dof))
        self.stamps = [None, None]
        self.state_id = state_id

        if bias is None:
            bias = np.zeros((group.dof))

        bias = np.array(bias).ravel()
        self.bias = bias
        self.new_bias = bias

    def increment(self, u: StampedValue, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_velocity = u.value - self.bias

        # Increment the value
        U = self.group.Exp(unbiased_velocity * dt)
        self.original_value = self.original_value @ U

        # Increment the covariance
        A = self.group.adjoint(self.group.inverse(U))
        L = dt * self.group.left_jacobian(-unbiased_velocity * dt)
        self.covariance = (
            A @ self.covariance @ A.T + L @ self.input_covariance @ L.T
        )

        # Increment the bias jacobian
        self.bias_jacobian = A @ self.bias_jacobian + L
        self.symmetrize()

    def update_bias(self, new_bias: np.ndarray):
        """
        Internally updates the RMI given a new bias.

        Parameters
        ----------
        new_bias : np.ndarray
            New bias values
        """
        if self.bias is None:
            raise ValueError("Cannot update bias of RMI without original bias")

        self.new_bias = new_bias

    @property
    def value(self):
        """
        Returns
        -------
        numpy.ndarray
            The RMI matrix :math:`\Delta \mathbf{U}_{ij}`.
        """
        db = self.new_bias - self.bias

        return self.original_value @ self.group.Exp(self.bias_jacobian @ db)

    def plus(self, w: np.ndarray):
        """
        Adds noise to the RMI

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        RelativeMotionIncrement
            The updated RMI
        """
        new = self.copy()
        new.original_value = new.original_value @ new.group.Exp(w)

        return new

    def copy(self):
        """
        Returns
        -------
        RelativeMotionIncrement
            A copy of the RMI
        """
        new = self.new()
        new.original_value = self.original_value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

    def new(self):
        """
        Returns
        -------
        RelativeMotionIncrement
            A copy of the RMI
        """
        new = self.__class__(
            self.group, self.input_covariance, self.bias, self.state_id
        )
        return new


class PreintegratedBodyVelocity(ProcessModel):
    """
    Process model that performs prediction of the state given an RMI
    :math:`\Delta \mathbf{U}_{ij}` according to the equation

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{U}_{ij}.

    The covariance is also propagated accordingly
    """

    def __init__(self):
        pass

    def evaluate(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt=None
    ) -> MatrixLieGroupState:
        x = x.copy()
        if rmi.stamps[-1] is not None and rmi.stamps[0] is not None:
            x.stamp = rmi.stamps[-1]
        x.value = x.value @ rmi.value
        return x

    def jacobian(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt=None
    ) -> np.ndarray:
        dof = x.dof
        if rmi.input_covariance.shape[0] == dof:
            estimating_bias = False
            total_dof = x.dof
        elif rmi.input_covariance.shape[0] == 2 * dof:
            estimating_bias = True
            total_dof = 2 * x.dof
        else:
            raise ValueError("Input covariance has incorrect shape")

        A = np.identity(total_dof)

        if x.direction == "right":
            A[0:dof, 0:dof] = x.group.adjoint(x.group.inverse(rmi.value))
            if estimating_bias:
                A[0:dof, dof:total_dof] = rmi.bias_jacobian

        elif x.direction == "left":
            Ad = x.group.adjoint(rmi.value)
            if estimating_bias:
                A[0:dof, dof:total_dof] = Ad @ rmi.bias_jacobian

        return A

    def covariance(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt=None
    ) -> np.ndarray:
        dof = x.dof
        if rmi.input_covariance.shape[0] == dof:
            total_dof = x.dof
        elif rmi.input_covariance.shape[0] == 2 * dof:
            total_dof = 2 * x.dof
        else:
            raise ValueError("Input covariance has incorrect shape")

        if x.direction == "right":
            L = np.identity(total_dof)
        elif x.direction == "left":
            L = np.identity(total_dof)
            Ad = x.group.adjoint(x.value @ rmi.value)
            L[0:dof, 0:dof] = Ad

        return L @ rmi.covariance @ L.T


class AngularVelocityIncrement(BodyVelocityIncrement):
    """
    This is a preintegration class for angular velocity measurements, on only
    attitude. Give a rotation matrix :math:`\mathbf{C}_{k}`, the preintegrated
    process model is of the form

    .. math::
        \mathbf{C}_{j} = \mathbf{C}_{i} \Delta \mathbf{\Omega}_{ij},

    where :math:`\Delta \mathbf{\Omega}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{\Omega}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{\omega}_{k}^\wedge)

    and :math:`\mathbf{\omega}_{k}` is the angular velocity measurement at time :math:`k`.

    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id=None):
        super().__init__(SO3, Q, bias, state_id=state_id)

    def new(self) -> "AngularVelocityIncrement":
        """
        Returns
        -------
        AngularVelocityIncrement
            A new AngularVelocityIncrement with reinitialized values
        """
        new = self.__class__(
            self.input_covariance, self.bias, self.state_id
        )
        return new


class WheelOdometryIncrement(BodyVelocityIncrement):
    """
    This is a preintegration class for wheel odometry measurements on SE(n).
    The preintegrated process model is of the form

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{V}_{ij},

    where :math:`\Delta \mathbf{V}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{V}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{\\varpi}_{k}^\wedge)

    and :math:`\mathbf{\\varpi}_{k} = [\mathbf{\omega}_k^T \;
    \mathbf{v}_k^T]^T` is the angular and translational velocity of the
    robot at time :math:`k`.
    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id=None):
        if Q.shape == (6, 6):
            group = SE3
        elif Q.shape == (3, 3):
            group = SE2
        else:
            raise ValueError("Input covariance has incorrect shape")

        super().__init__(group, Q, bias, state_id=state_id)

    def new(self) -> "WheelOdometryIncrement":
        """
        Returns
        -------
        WheelOdometryIncrement
            A new WheelOdometryIncrement with reinitialized values
        """
        new = self.__class__(
            self.input_covariance, self.bias, self.state_id
        )
        return new


# Alternate names for classes
class PreintegratedAngularVelocity(PreintegratedBodyVelocity):
    pass


class PreintegratedWheelOdometry(PreintegratedBodyVelocity):
    pass


class LinearIncrement(RelativeMotionIncrement):
    """
    This class preintegrates any process model of the form

    .. math::
        \mathbf{x}_{k} = \mathbf{A}_{k-1} \mathbf{x}_{k-1}
        + \mathbf{B}_{k-1}(\mathbf{u}_{k-1} + \mathbf{w}_{k-1})

    where :math:`\mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0},
    \mathbf{Q}_{k-1})`. By directly interating this equation, it can be shown
    that

    .. math::
        \mathbf{x}_j - \left(\prod_{k=i}^{j-1} \mathbf{A}_k \\right) \mathbf{x}_i
        + \sum_{k=i}^{j-1} \left(\prod_{\ell=k+1}^{j-1} \mathbf{A}_\ell\\right)
        \mathbf{B}_k \mathbf{u}_k

    which can be rewritten as

    .. math::
        \mathbf{x}_j = \mathbf{A}_{ij} \mathbf{x}_i + \Delta \mathbf{u}_{ij} + \mathbf{w}_{ij}

    This class will compute the preintegrated quantities :math:`\mathbf{A}_{ij},
    \Delta \\bar{\mathbf{u}}_{ij}, \mathbf{Q}_{ij}` where :math:`\mathbf{w}_{ij} \sim \mathcal{N}(\mathbf{0},
    \mathbf{Q}_{ij})`.

    """

    def __init__(
        self,
        input_covariance: np.ndarray,
        state_matrix: Callable[[StampedValue, float], np.ndarray],
        input_matrix: Callable[[StampedValue, float], np.ndarray],
        dof: int,
        bias: np.ndarray = None,
        state_id: Any = None,
    ):
        """

        Parameters
        ----------
        input_covariance : numpy.ndarray
            Covariance associated with the input. If a bias is also supplied,
            then this should also contain the covariance of the bias random walk.
        state_matrix : Callable[[StampedValue, float], numpy.ndarray]
            The state transition matrix, supplied as a function of the input
            and time interval `dt`.
        input_matrix : Callable[[StampedValue, float], numpy.ndarray]
            The input matrix, supplied as a function of the input and time
            interval `dt`.
        dof : int
            the total dof of the state
        bias : numpy.ndarray, optional
            If provided, this bias will be subtracted from the input values.
            Furthermore, the covariance associated with this RMI will also
            be augmented to include a bias state.
        state_id : Any, optional
            Optional container for other identifying information, by default None.
        """
        self.dof = dof
        self._input_covariance = input_covariance
        self._state_matrix = state_matrix
        self._input_matrix = input_matrix
        self.state_id = state_id

        #:numpy.ndarray: The RMI value before a new bias correction.
        self.original_value = [
            np.identity(dof),
            np.zeros((dof, 1)),
        ]
        if bias is not None:
            bias = np.array(bias).ravel()

            #:numpy.ndarray: Covariance matrix :math:`\mathbf{Q}_{ij}`.
            self.covariance = np.zeros((dof + bias.size, dof + bias.size))

            #:numpy.ndarray: The bias jacobian :math:`\mathbf{B}_{ij}`.
            self.bias_jacobian = np.zeros((dof, bias.size))
        else:
            self.covariance = np.zeros((dof, dof))
            self.bias_jacobian = None

        #:numpy.ndarray: The bias value used when computing the RMI
        self.original_bias = bias

        #:numpy.ndarray: The bias value used to modify the original RMIs
        self.new_bias = self.original_bias
        self.stamps = [None, None]

    def increment(self, u: StampedValue, dt: float):
        u = u.copy()

        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        # Remove bias is supplied
        if self.original_bias is not None:
            u.value = u.value.ravel() - self.original_bias.ravel()

        # Get the state and input matrices, increment the RMI value
        A = self._state_matrix(u, dt)
        B = self._input_matrix(u, dt)

        if self.original_bias is not None:
            bias = self.original_bias
            A_full = np.zeros((self.dof + bias.size, self.dof + bias.size))
            A_full[: self.dof, : self.dof] = A
            A_full[self.dof :, self.dof :] = np.identity(bias.size)
            A_full[: self.dof, self.dof :] = -B

            B_full = np.zeros((self.dof + bias.size, 2 * bias.size))
            B_full[: self.dof, : bias.size] = B
            B_full[self.dof :, bias.size :] = dt * np.identity(bias.size)
            if u.value.size == 2 * bias.size:
                u.value = u.value[: bias.size]
        else:
            A_full = A
            B_full = B

        self.original_value[0] = A @ self.original_value[0]
        self.original_value[1] = A @ self.original_value[1].reshape(
            (-1, 1)
        ) + B @ u.value.reshape((-1, 1))

        self.covariance = (
            A_full @ self.covariance @ A_full.T
            + B_full @ self._input_covariance @ B_full.T
        )

        if self.bias_jacobian is None:
            self.bias_jacobian = np.zeros((self.dof, u.dof))

        self.bias_jacobian = A @ self.bias_jacobian - B

    def update_bias(self, new_bias):
        """
        Update the bias of the RMI.

        Parameters
        ----------
        new_bias : np.ndarray
            The new bias value
        """
        if self.original_bias is None:
            raise ValueError("Cannot update bias of RMI without original bias")

        self.new_bias = new_bias

    @property
    def value(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The two matrices :math:`\mathbf{A}_{ij}, \Delta\\bar{\mathbf{u}}_{ij}`
        that make up the RMI.
        """

        if self.original_bias is None or self.new_bias is None:
            return self.original_value
        else:
            delta_bias = (self.new_bias - self.original_bias).reshape((-1, 1))
            out = self.original_value
            out[1] = (
                self.original_value[1].reshape((-1, 1))
                + self.bias_jacobian @ delta_bias
            )
            return out

    def plus(self, w: np.ndarray) -> "LinearIncrement":
        """
        Increment the RMI itself

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        LinearIncrement
            The updated RMI
        """
        new = self.copy()
        new.original_value[1] = new.original_value[1] + w
        return new

    def copy(self) -> "LinearIncrement":
        """
        Returns
        -------
        LinearIncrement
            A copy of the RMI
        """
        new = self.new()
        new.original_value = [
            self.original_value[0].copy(),
            self.original_value[1].copy(),
        ]
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        new.new_bias = self.new_bias
        return new

    def new(self, new_bias=None) -> "LinearIncrement":
        """
        Returns
        -------
        LinearIncrement
            A copy of the RMI with reinitialized values
        """
        if new_bias is None:
            new_bias = self.original_bias

        new = LinearIncrement(
            self._input_covariance,
            self._state_matrix,
            self._input_matrix,
            self.dof,
            new_bias,
            self.state_id,
        )
        return new


class PreintegratedLinearModel(ProcessModel):
    """
    Process model that applies a preintegrated `LinearIncrement` to predict
    a state forward in time using the equation

    .. math::
        \mathbf{x}_k = \mathbf{A}_{ij} \mathbf{x}_i + \Delta \mathbf{u}_{ij}
    """

    def __init__(self):
        pass

    def evaluate(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> VectorState:

        x = x.copy()
        A_ij = rmi.value[0]
        Du_ij = rmi.value[1]

        if rmi.original_bias is not None:
            x_i = x.value[0 : -rmi.original_bias.size].reshape((-1, 1))
        else:
            x_i = x.value.reshape((-1, 1))

        x_j = A_ij @ x_i + Du_ij

        if rmi.original_bias is not None:
            x_j = np.vstack(
                (x_j, x.value[-rmi.original_bias.size :].reshape((-1, 1)))
            )

        x.value = x_j.ravel()
        return x

    def jacobian(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> np.ndarray:

        if rmi.original_bias is not None:
            A_ij = rmi.value[0]
            B_ij = rmi.bias_jacobian
            state_dof = x.dof - rmi.original_bias.size
            A = np.zeros((x.dof, x.dof))
            A[:state_dof, :state_dof] = A_ij
            A[state_dof:, state_dof:] = np.identity(rmi.original_bias.size)
            A[:state_dof, state_dof:] = B_ij
        else:
            A = rmi.value[0]

        return A

    def covariance(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> np.ndarray:
        return rmi.covariance
