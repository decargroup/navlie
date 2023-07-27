from navlie.types import (
    Measurement,
    ProcessModel,
    MeasurementModel,
    StampedValue,
    Input,
)
from navlie.lib.states import (
    CompositeState,
    MatrixLieGroupState,
    VectorState,
)
from pylie import SO2, SO3
import numpy as np
from typing import List, Any
from scipy.linalg import block_diag


class SingleIntegrator(ProcessModel):
    """
    The single-integrator process model is a process model of the form

    .. math::

        \dot{\mathbf{x}} = \mathbf{u} 

    where :math:`\mathbf{u}\in \mathbb{R}^n` is a simple velocity input.
    In discrete time the process model is simply

    .. math::
        \mathbf{x}_k = \mathbf{x}_{k-1} + \Delta t \mathbf{u}_{k-1}.
    
    """

    def __init__(self, Q: np.ndarray):
        """
        Parameters
        ----------
        Q : np.ndarray
            Square matrix representing the discrete-time covariance of the input 
            noise.

        Raises
        ------
        ValueError
            If `Q` is not a square matrix.
        """

        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = Q.shape[0]

    def evaluate(
        self, x: VectorState, u: StampedValue, dt: float
    ) -> np.ndarray:
        x = x.copy()
        x.value = x.value + dt * u.value
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        return np.identity(self.dim)

    def covariance(self, x, u, dt) -> np.ndarray:
        return dt**2 * self._Q


class DoubleIntegrator(ProcessModel):
    """
    The double-integrator process model is a second-order point kinematic model
    given in continuous time by

    .. math::
        \dot{\mathbf{r}} = \mathbf{v}
        \dot{\mathbf{v}} = \mathbf{u}

    where :math:`\mathbf{u}` is the input.
    """

    def __init__(self, Q: np.ndarray):
        """
        inputs:
            Q: Discrete time covariance on the input u.
        """
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = Q.shape[0]

    def evaluate(
        self, x: VectorState, u: StampedValue, dt: float
    ) -> np.ndarray:
        """
        Evaluate discrete-time process model
        """
        x_new = x.copy()
        Ad = self.jacobian(None, None, dt)
        Ld = self.input_jacobian(dt)
        u = np.atleast_1d(u.value)
        x_new.value = (
            Ad @ x.value.reshape((-1, 1)) + Ld @ u[: self.dim].reshape((-1, 1))
        ).ravel()
        return x_new

    def jacobian(self, x, u, dt) -> np.ndarray:
        """
        Discrete-time state Jacobian
        """

        Ad = np.identity(2 * self.dim)
        Ad[0 : self.dim, self.dim :] = dt * np.identity(self.dim)
        return Ad

    def covariance(self, x, u, dt) -> np.ndarray:
        """
        Discrete-time covariance on process model
        """

        Ld = self.input_jacobian(dt)
        return Ld @ self._Q @ Ld.T

    def input_jacobian(self, dt):
        """
        Discrete-time input Jacobian
        """
        Ld = np.zeros((2 * self.dim, self.dim))
        Ld[0 : self.dim, :] = 0.5 * dt**2 * np.identity(self.dim)
        Ld[self.dim :, :] = dt * np.identity(self.dim)
        return Ld


class DoubleIntegratorWithBias(DoubleIntegrator):
    """
    The double-integrator process model, but with an additional bias on the input.

        x_dot = v
        v_dot = u - b
        b_dot = w (noise)

    """

    def __init__(self, Q: np.ndarray):
        """

        Parameters
        ----------
        Q : np.ndarray
            Q: Discrete time covariance on the input u.

        Raises
        ------
        ValueError
            if Q is not an n x n matrix.
        """
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = int(Q.shape[0] / 2)

    def evaluate(
        self, x: VectorState, u: StampedValue, dt: float
    ) -> VectorState:
        """
        Evaluate discrete-time process model
        """
        x = x.copy()
        u = u.copy()
        Ad = super().jacobian(x, u, dt)
        Ld = super().input_jacobian(dt)

        pv = x.value[0 : 2 * self.dim].reshape((-1, 1))
        bias = x.value[2 * self.dim :].reshape((-1, 1))
        accel = u.value[: self.dim].reshape((-1, 1)) - bias

        # If the input contains extra dimensions, we assume that they are the
        # random walk input being used for data generation.
        if u.value.size > self.dim:
            walk = u.value[self.dim :].reshape((-1, 1))
        else:
            walk = np.zeros((self.dim, 1))

        pv = (Ad @ pv + Ld @ accel).ravel()
        x.value[0 : 2 * self.dim] = pv
        x.value[2 * self.dim :] = (bias + walk * dt).ravel()
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        """
        Discrete-time state Jacobian
        """
        Ad = super().jacobian(x, u, dt)
        Ld = super().input_jacobian(dt)

        A = np.block(
            [
                [Ad, -Ld],
                [np.zeros((self.dim, 2 * self.dim)), np.identity(self.dim)],
            ]
        )
        return A

    def covariance(self, x, u, dt) -> np.ndarray:
        """
        Discrete-time covariance on process model
        """

        L = self.input_jacobian(dt)
        return L @ self._Q @ L.T

    def input_jacobian(self, dt):
        Ld = super().input_jacobian(dt)
        L = np.zeros((3 * self.dim, 2 * self.dim))
        L[0 : 2 * self.dim, 0 : self.dim] = Ld
        L[2 * self.dim :, self.dim :] = dt * np.identity(self.dim)
        return L


class OneDimensionalPositionVelocityRange(MeasurementModel):
    """
    A 1D range measurement for a state consisting of position and velocity
    """

    # TODO. We should remove this. Double integrator and RangePointToAnchor should
    # satisfy these needs
    def __init__(self, R: float):
        self._R = np.array(R)

    def evaluate(self, x: VectorState) -> np.ndarray:
        return x.value[0]

    def jacobian(self, x: VectorState) -> np.ndarray:
        return np.array([1, 0]).reshape(1, -1)

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R


class BodyFrameVelocity(ProcessModel):
    """
    The body-frame velocity process model assumes that the input contains
    both translational and angular velocity measurements, both relative to
    a local reference frame, but resolved in the robot body frame.

    .. math::
        \mathbf{T}_k = \mathbf{T}_{k-1} \exp(\Delta t \mathbf{u}_{k-1}^\wedge)

    This is commonly the process model associated with SE(n).
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> MatrixLieGroupState:
        x = x.copy()
        x.value = x.value @ x.group.Exp(u.value * dt)
        return x

    def jacobian(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u.value * dt))
        elif x.direction == "left":
            return np.identity(x.dof)

    def covariance(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u.value * dt)
        elif x.direction == "left":
            Ad = x.group.adjoint(x.value @ x.group.Exp(u.value * dt))
            L = dt * Ad @ x.group.left_jacobian(-u.value * dt)

        return L @ self._Q @ L.T


class RelativeBodyFrameVelocity(ProcessModel):
    def __init__(self, Q1: np.ndarray, Q2: np.ndarray):
        self._Q1 = Q1
        self._Q2 = Q2

    def evaluate(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> MatrixLieGroupState:
        x = x.copy()
        u = u.value.reshape((2, round(u.value.size / 2)))
        x.value = x.group.Exp(-u[0] * dt) @ x.value @ x.group.Exp(u[1] * dt)
        return x

    def jacobian(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        u = u.value.reshape((2, round(u.value.size / 2)))
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u[1] * dt))
        else:
            raise NotImplementedError(
                "TODO: left jacobian not yet implemented."
            )

    def covariance(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        u = u.value.reshape((2, round(u.value.size / 2)))
        u1 = u[0]
        u2 = u[1]
        if x.direction == "right":
            L1 = (
                dt
                * x.group.adjoint(x.value @ x.group.Exp(u2 * dt))
                @ x.group.left_jacobian(dt * u1)
            )
            L2 = dt * x.group.left_jacobian(-dt * u2)
            return L1 @ self._Q1 @ L1.T + L2 @ self._Q2 @ L2.T
        else:
            raise NotImplementedError(
                "TODO: left covariance not yet implemented."
            )


class LinearMeasurement(MeasurementModel):
    def __init__(self, C: np.ndarray, R: np.ndarray):
        self._C = C
        self._R = R

    def evaluate(self, x: VectorState) -> np.ndarray:
        return self._C @ x.value.reshape((-1, 1))

    def jacobian(self, x: VectorState) -> np.ndarray:
        return self._C

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R


class CompositeInput(Input):
    # TODO: add tests to new methods
    def __init__(self, input_list: List[Input]) -> None:
        self.input_list = input_list

    @property
    def dof(self) -> int:
        return sum([input.dof for input in self.input_list])

    @property
    def stamp(self) -> float:
        return self.input_list[0].stamp
    
    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of inputs.
        """
        return [x.state_id for x in self.input_list].index(state_id)

    def add_input(self, input: Input):
        """
        Adds an input to the composite input.
        """
        self.input_list.append(input)

    def remove_input_by_id(self, state_id):
        """
        Removes a given input by ID.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list.pop(idx)

    def get_input_by_id(self, state_id) -> Input:
        """
        Get input object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx]

    def get_dof_by_id(self, state_id) -> int:
        """
        Get degrees of freedom of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].dof

    def get_stamp_by_id(self, state_id) -> float:
        """
        Get timestamp of sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.input_list[idx].stamp

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx].stamp = stamp

    def set_input_by_id(self, input: Input, state_id):
        """
        Set the whole sub-input by id.
        """
        idx = self.get_index_by_id(state_id)
        self.input_list[idx] = input

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all subinputs.
        """
        for input in self.input_list:
            input.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeInput object back into a list of inputs.
        """
        return self.input_list

    def copy(self) -> "CompositeInput":
        return CompositeInput([input.copy() for input in self.input_list])

    def plus(self, w: np.ndarray):
        new = self.copy()
        temp = w
        for i, input in enumerate(self.input_list):
            new.input_list[i] = input.plus(temp[: input.dof])
            temp = temp[input.dof :]

        return new


class CompositeProcessModel(ProcessModel):
    """
    Should this be called a StackedProcessModel?
    # TODO: Add documentation and tests
    """

    # TODO: This needs to be expanded and/or changed. We have come across the
    # following use cases:
    # 1. Applying a completely seperate process model to each sub-state.
    # 2. Applying a single process model to each sub-state (seperately).
    # 3. Applying a single process model to one sub-state, and leaving the rest
    #    unchanged.
    # 4. Applying process model A to some sub-states, and process model B to
    #    other sub-states
    # 5. Receiving a CompositeInput, which is a list of synchronously-received
    #    inputs, and applying each input to the corresponding sub-state.
    # 6. Receiving the state-specific input asynchronously, applying to the
    #    corresponding sub-state, and leaving the rest unchanged. Typically happens
    #    with case 3.

    # What they all have in common: list of decoupled process models, one per
    # substate. For coupled process models, the user will have to define their
    # own process model from scratch.

    def __init__(
        self, 
        model_list: List[ProcessModel],
        shared_input: bool = False,
    ):
        self._model_list = model_list
        self._shared_input = shared_input

    def evaluate(
        self, 
        x: CompositeState, 
        u: CompositeInput, 
        dt: float, 
    ) -> CompositeState:
        x = x.copy()
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            x.value[i] = self._model_list[i].evaluate(x_sub, u_sub, dt)

        return x

    def jacobian(
        self, 
        x: CompositeState, 
        u: CompositeInput, 
        dt: float,
    ) -> np.ndarray:
        jac = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            jac.append(self._model_list[i].jacobian(x_sub, u_sub, dt))

        return block_diag(*jac)

    def covariance(
        self, 
        x: CompositeState, 
        u: CompositeInput, 
        dt: float,
    ) -> np.ndarray:
        cov = []
        for i, x_sub in enumerate(x.value):
            if self._shared_input:
                u_sub = u
            else:
                u_sub = u.input_list[i]
            cov.append(self._model_list[i].covariance(x_sub, u_sub, dt))

        return block_diag(*cov)


class CompositeMeasurementModel(MeasurementModel):
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState.
    """

    def __init__(self, model: MeasurementModel, state_id):
        self.model = model
        self.state_id = state_id

    def __repr__(self):
        return f"{self.model}(of substate {self.state_id})"

    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        jac_sub = self.model.jacobian(x_sub)
        jac = np.zeros((jac_sub.shape[0], x.dof))
        slc = x.get_slice_by_id(self.state_id)
        jac[:, slc] = jac_sub
        return jac

    def covariance(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        return self.model.covariance(x_sub)


class CompositeMeasurement(Measurement):
    def __init__(self, y: Measurement, state_id: Any):
        """
        Converts a standard Measurement into a CompositeMeasurement, which
        replaces the model with a CompositeMeasurementModel.

        Parameters
        ----------
        y : Measurement
            Measurement to be converted.
        state_id : Any
            ID of the state that the measurement will be assigned to,
            as per the CompositeMeasurementModel.
        """
        model = CompositeMeasurementModel(y.model, state_id)
        super().__init__(
            value=y.value, stamp=y.stamp, model=model, state_id=y.state_id
        )


class RangePointToAnchor(MeasurementModel):
    """
    Range measurement from a point state to an anchor (which is also another
    point).
    """

    def __init__(self, anchor_position: List[float], R: float):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = np.array(R)
        self.dim = self._r_cw_a.size

    def evaluate(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.ravel()[0 : self.dim]
        y = np.linalg.norm(self._r_cw_a - r_zw_a)
        return y

    def jacobian(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.ravel()[0 : self.dim]
        r_zc_a: np.ndarray = r_zw_a - self._r_cw_a
        y = np.linalg.norm(r_zc_a)

        jac = np.zeros((1, x.dof))
        jac[0, : self.dim] = r_zc_a.reshape((1, -1)) / y
        return jac

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R


class PointRelativePosition(MeasurementModel):
    def __init__(
        self,
        landmark_position: np.ndarray,
        R: np.ndarray,
    ):
        self._landmark_position = np.array(landmark_position).ravel()
        self._R = R

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        """Evaluates the measurement model of a landmark from a given pose."""
        r_zw_a = x.position.reshape((-1, 1))
        C_ab = x.attitude
        r_pw_a = self._landmark_position.reshape((-1, 1))
        return C_ab.T @ (r_pw_a - r_zw_a)

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position.reshape((-1, 1))
        C_ab = x.attitude
        r_pw_a = self._landmark_position.reshape((-1, 1))
        y = C_ab.T @ (r_pw_a - r_zw_a)

        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(y), position=-np.identity(r_zw_a.shape[0])
            )

        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=-C_ab.T @ SO3.odot(r_pw_a), position=-C_ab.T
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R


class InvariantPointRelativePosition(MeasurementModel):
    def __init__(self, y: np.ndarray, model: PointRelativePosition):
        self.y = y.ravel()
        self.measurement_model = model

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        """Computes the right-invariant innovation.


        Parameters
        ----------
        x : MatrixLieGroupState
            Evaluation point of the innovation.

        Returns
        -------
        np.ndarray
            Residual.
        """
        y_hat = self.measurement_model.evaluate(x)
        e: np.ndarray = y_hat.ravel() - self.y.ravel()
        z = x.attitude @ e

        return z

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        """Compute the Jacobian of the innovation directly.

        Parameters
        ----------
        x : MatrixLieGroupState
            Matrix Lie group state containing attitude and position

        Returns
        -------
        np.ndarray
            Jacobian of the innovation w.r.t the state
        """

        if x.direction == "left":
            jac_attitude = SO3.cross(self.measurement_model._landmark_position)
            jac_position = -np.identity(3)
        else:
            raise NotImplementedError("Right jacobian not implemented.")

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )

        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:

        R = np.atleast_2d(self.measurement_model.covariance(x))
        M = x.attitude
        cov = M @ R @ M.T

        return cov


class RangePoseToAnchor(MeasurementModel):
    """
    Range measurement from a pose state to an anchor.
    """

    def __init__(
        self,
        anchor_position: List[float],
        tag_body_position: List[float],
        R: float,
    ):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = R
        self._r_tz_b = np.array(tag_body_position).flatten()

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        return np.linalg.norm(r_tc_a)

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        rho = r_tc_a / np.linalg.norm(r_tc_a)

        if x.direction == "right":
            jac_attitude = rho.T @ C_ab @ att_group.odot(self._r_tz_b)
            jac_position = rho.T @ C_ab
        elif x.direction == "left":
            jac_attitude = rho.T @ att_group.odot(C_ab @ self._r_tz_b + r_zw_a)
            jac_position = rho.T @ np.identity(r_zw_a.size)

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )
        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R


class RangePoseToPose(MeasurementModel):
    """
    Range model given two absolute poses of rigid bodies, each containing a tag.
    """
    # TODO. tag_body_positions should be optional. argh but this will be 
    # a breaking change since the argument order needs to be different.
    def __init__(
        self, tag_body_position1, tag_body_position2, state_id1, state_id2, R
    ):
        self.tag_body_position1 = np.array(tag_body_position1).flatten()
        self.tag_body_position2 = np.array(tag_body_position2).flatten()
        self.state_id1 = state_id1
        self.state_id2 = state_id2
        self._R = R

    def evaluate(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )
        return np.array(np.linalg.norm(r_t1t2_a.flatten()))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )

        if C_a1.shape == (2, 2):
            att_group = SO2
        elif C_a1.shape == (3, 3):
            att_group = SO3

        rho: np.ndarray = (
            r_t1t2_a / np.linalg.norm(r_t1t2_a.flatten())
        ).reshape((-1, 1))

        if x1.direction == "right":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ C_a1 @ att_group.odot(r_t1_1),
                position=rho.T @ C_a1,
            )
        elif x1.direction == "left":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ att_group.odot(C_a1 @ r_t1_1 + r_1w_a),
                position=rho.T @ np.identity(r_t1_1.size),
            )

        if x2.direction == "right":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ C_a2 @ att_group.odot(r_t2_2),
                position=-rho.T @ C_a2,
            )
        elif x2.direction == "left":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ att_group.odot(C_a2 @ r_t2_2 + r_2w_a),
                position=-rho.T @ np.identity(r_t2_2.size),
            )

        return x.jacobian_from_blocks(
            {self.state_id1: jac1, self.state_id2: jac2}
        )

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R


class RangeRelativePose(CompositeMeasurementModel):
    """
    Range model given a pose of another body relative to current pose. This
    model operates on a CompositeState where it is assumed that the neighbor
    relative pose is stored as a substate somewhere inside the composite state
    with a state_id matching the `nb_state_id` supplied to this model.
    """

    def __init__(
        self,
        tag_body_position: np.ndarray,
        nb_tag_body_position: np.ndarray,
        nb_state_id: Any,
        R: np.ndarray,
    ):
        """

        Parameters
        ----------
        tag_body_position : numpy.ndarray
            Position of tag in body frame of Robot 1.
        nb_tag_body_position : numpy.ndarray
            Position of 2nd tag in body frame of Robot 2.
        nb_state_id : Any
            State ID of Robot 2.
        R : float or numpy.ndarray
            covariance associated with range measurement
        """

        model = RangePoseToAnchor(tag_body_position, nb_tag_body_position, R)
        super(RangeRelativePose, self).__init__(model, nb_state_id)

    def __repr__(self):
        return f"RangeRelativePose (of substate {self.state_id})"



class GlobalPosition(MeasurementModel):
    """
    Global, world-frame, or "absolute" position measurement.
    """

    def __init__(self, R: np.ndarray):
        self.R = R

    def evaluate(self, x: MatrixLieGroupState):
        return x.position

    def jacobian(self, x: MatrixLieGroupState):
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        if x.direction == "right":
            return x.jacobian_from_blocks(position=x.attitude)
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=att_group.odot(x.position),
                position=np.identity(x.position.size),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R


class GlobalVelocity(MeasurementModel):
    """
    Global, world-frame, or "absolute" position measurement.

    Compatible with SE23State, IMUState
    """

    def __init__(self, R: np.ndarray):
        self.R = R

    def evaluate(self, x: MatrixLieGroupState):
        return x.velocity

    def jacobian(self, x: MatrixLieGroupState):
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        if x.direction == "right":
            return x.jacobian_from_blocks(velocity=x.attitude)
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=att_group.odot(x.velocity),
                velocity=np.identity(x.velocity.size),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.velocity.size)
        else:
            return self.R
        

class Altitude(MeasurementModel):
    """
    A model that returns that z component of a position vector.
    """

    def __init__(self, R: np.ndarray, minimum=None, bias=0.0):
        """

        Parameters
        ----------
        R : np.ndarray
            variance associated with the measurement
        minimum : float, optional
            Minimal height for the measurement to be valid, by default None
        bias : float, optional
            Fixed sensor bias, by default 0.0. This bias will be added to the
            z component of position to create the modelled measurement.
        """
        self.R = R
        if minimum is None:
            minimum = -np.inf
        self.minimum = minimum
        self.bias = bias

    def evaluate(self, x: MatrixLieGroupState):
        h = x.position[2] + self.bias
        return h if h > self.minimum else None

    def jacobian(self, x: MatrixLieGroupState):

        if x.direction == "right":
            return x.jacobian_from_blocks(
                position=x.attitude[2, :].reshape((1, -1))
            )
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=SO3.odot(x.position)[2, :].reshape((1, -1)),
                position=np.array(([[0, 0, 1]])),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self.R


class Gravitometer(MeasurementModel):
    """
    Gravitometer model of the form

    .. math::

        \mathbf{y} = \mathbf{C}_{ab}^T \mathbf{g}_a + \mathbf{v}

    where :math:`\mathbf{g}_a` is the magnetic field vector in a world frame `a`.
    """

    def __init__(
        self, R: np.ndarray, gravity_vector: List[float] = None
    ):
        """
        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        gravity_vector : list[float] or numpy.ndarray, optional
            local magnetic field vector, by default [0, 0, -9.80665]
        """
        if gravity_vector is None:
            gravity_vector = [0, 0, -9.80665]

        self.R = R
        self._g_a = np.array(gravity_vector).reshape((-1, 1))

    def evaluate(self, x: MatrixLieGroupState):
        return x.attitude.T @ self._g_a

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._g_a)
            )
        elif x.direction == "left":

            return x.jacobian_from_blocks(
                attitude=x.attitude.T @ -SO3.odot(self._g_a)
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R


class Magnetometer(MeasurementModel):
    """
    Magnetometer model of the form

    .. math::

        \mathbf{y} = \mathbf{C}_{ab}^T \mathbf{m}_a + \mathbf{v}

    where :math:`\mathbf{m}_a` is the magnetic field vector in a world frame `a`.
    """

    def __init__(self, R: np.ndarray, magnetic_vector: List[float] = None):
        """

        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        magnetic_vector : list[float] or numpy.ndarray, optional
            local magnetic field vector, by default [1, 0, 0]
        """
        if magnetic_vector is None:
            magnetic_vector = [1, 0, 0]

        self.R = R
        self._m_a = np.array(magnetic_vector).reshape((-1, 1))

    def evaluate(self, x: MatrixLieGroupState):
        return x.attitude.T @ self._m_a

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._m_a)
            )
        elif x.direction == "left":

            return x.jacobian_from_blocks(
                attitude=-x.attitude.T @ SO3.odot(self._m_a)
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R


class _InvariantInnovation(MeasurementModel):
    def __init__(
        self, y: np.ndarray, model: MeasurementModel, direction="right"
    ):
        self.measurement_model = model
        self.y = y.ravel()
        self.direction = direction

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        y_hat = self.measurement_model.evaluate(x)
        e: np.ndarray = y_hat.ravel() - self.y.ravel()

        direction = self._compute_direction(x)
        if direction == "left":
            z = x.attitude.T @ e
        elif direction == "right":
            z = x.attitude @ e

        return z

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        G = self.measurement_model.jacobian(x)

        direction = self._compute_direction(x)
        if direction == "left":
            jac = x.attitude.T @ G
        elif direction == "right":
            jac = x.attitude @ G

        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:

        R = np.atleast_2d(self.measurement_model.covariance(x))

        direction = self._compute_direction(x)
        if direction == "left":
            M = x.attitude.T
            cov = M @ R @ M.T
        elif direction == "right":
            M = x.attitude
            cov = M @ R @ M.T
        return cov

    def _compute_direction(self, x: MatrixLieGroupState) -> str:
        if self.direction == "left":
            direction = self.direction
        elif self.direction == "right":
            direction = self.direction
        elif self.direction == "auto":
            if x.direction == "left":
                direction = "right"
            elif x.direction == "right":
                direction = "left"
        else:
            raise ValueError(
                "Invalid direction. Must be 'left', 'right' or 'auto'"
            )
        return direction


class InvariantMeasurement(Measurement):
    """
    Given a Measurement object, the class will construct a
    left- or right-invariant innovation ready to be fused into a state estimator.

    If a right-invariant innovation is chosen then the following will be formed.

    .. math::
        \mathbf{z} &= \\bar{\mathbf{X}}(\mathbf{y} - \\bar{\mathbf{y}})

        &= \\bar{\mathbf{X}}(\mathbf{g}(\mathbf{X}) +
        \mathbf{v} - \mathbf{g}(\\bar{\mathbf{X}}))

        &\\approx \\bar{\mathbf{X}}( \mathbf{g}(\\bar{\mathbf{X}})
        + \mathbf{G}\delta \mathbf{\\xi} + \mathbf{v}
        - \mathbf{g}(\\bar{\mathbf{X}}))

        &= \\bar{\mathbf{X}}\mathbf{G}\delta \mathbf{\\xi}
        + \\bar{\mathbf{X}}\mathbf{v}

    and hence :math:`\\bar{\mathbf{X}}\mathbf{G}` is the Jacobian of
    :math:`\mathbf{z}`, where :math:`\mathbf{G}` is the Jacobian of
    :math:`\mathbf{g}(\mathbf{X})`.  Similarly, if a left-invariant innovation is chosen,

     .. math::
        \mathbf{z} &= \\bar{\mathbf{X}}^{-1}(\mathbf{y} - \\bar{\mathbf{y}})

        &\\approx \\bar{\mathbf{X}}^{-1}\mathbf{G}\delta \mathbf{\\xi}
        + \\bar{\mathbf{X}}^{-1}\mathbf{v}

    and hence :math:`\\bar{\mathbf{X}}^{-1}\mathbf{G}` is the Jacobian of
    :math:`\mathbf{z}`.
    """

    def __init__(self, meas: Measurement, direction="auto", model=None):
        """
        Parameters
        ----------
        meas : Measurement
            Measurement value
        direction : "left" or "right" or "auto"
            whether to form a left- or right-invariant innovation, by default "auto".
            If "auto" is chosen, the direction will be chosen to be the opposite of
            the direction of the state.
        model : MeasurementModel, optional
            a measurement model that directly returns the innovation and
            Jacobian and covariance of the innovation. If none is supplied,
            the default InvariantInnovation will be used, which computes the
            Jacobian of the innovation indirectly via chain rule.
        """

        if model is None:
            model = _InvariantInnovation(meas.value, meas.model, direction)

        super(InvariantMeasurement, self).__init__(
            value=np.zeros((meas.value.size,)),
            stamp=meas.stamp,
            model=model,
            state_id=meas.state_id,
        )
