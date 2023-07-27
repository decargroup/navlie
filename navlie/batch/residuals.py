"""
A set of commonly-used residuals in batch estimation. 

These residuals are
    - the PriorResidual, to assign a prior estimate on the state,
    - the ProcessResidual, which uses a navlie `ProcessModel` to compute an error between
      a predicted state and the actual state,
    - a MeasurementResidual, which uses a navlie `Measurement` to compare 
      a true measurement to the measurement predicted by the `MeasurementModel`.

"""

from abc import ABC, abstractmethod
from typing import Hashable, List, Tuple
from navlie.types import State, ProcessModel, StampedValue, Measurement
import numpy as np


class Residual(ABC):
    """
    Abstract class for a residual to be used in batch estimation.

    Each residual must implement an evaluate(self, states) method,
    which returns an error and Jacobian of the error with
    respect to each of the states.

    Each residual must contain a list of keys, where each key corresponds to a
    variable for optimization.
    """

    def __init__(self, keys: List[Hashable]):
        # If the hasn't supplied a list, make a list
        if isinstance(keys, list):
            self.keys = keys
        else:
            self.keys = [keys]

    @abstractmethod
    def evaluate(
        self,
        states: List[State],
        compute_jacobian: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the residual and Jacobians.

        Parameters
        ----------
        states : List[State]
            List of states for optimization.
        compute_jacobian : List[bool], optional
            optional flag to compute Jacobians, by default None

        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray]]
            Returns the error and a list of Jacobians.
        """
        # TODO: seems more appropriate to receive states as a dict with
        # corresponding state names than as a list.
        pass


class PriorResidual(Residual):
    """
    A generic prior error.
    """

    def __init__(
        self,
        keys: List[Hashable],
        prior_state: State,
        prior_covariance: np.ndarray,
    ):
        super().__init__(keys)
        self._cov = prior_covariance
        self._x0 = prior_state
        # Precompute square-root of info matrix
        self._L = np.linalg.cholesky(np.linalg.inv(self._cov))

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the prior error of the form

            e = x.minus(x0),

        where :math:`\mathbf{x}` is our operating point and
        :math:`\mathbf{x}_0` is a prior guess.
        """
        x = states[0]
        error = x.minus(self._x0)
        # Weight the error
        error = self._L.T @ error
        # Compute Jacobian of error w.r.t x
        if compute_jacobians:
            jacobians = [None]

            if compute_jacobians[0]:
                jacobians[0] = self._L.T @ x.minus_jacobian(self._x0)
            return error, jacobians

        return error


class ProcessResidual(Residual):
    """
    A generic process residual.

    Can be used with any :class:`navlie.types.ProcessModel`.
    """

    def __init__(
        self,
        keys: List[Hashable],
        process_model: ProcessModel,
        u: StampedValue,
    ):
        super().__init__(keys)
        self._process_model = process_model
        self._u = u

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the process residual.

        An input :math:`\mathbf{u}` is used to propagate the state
        :math:\mathbf{x}_{k-1}` through the process model, to generate
        :math:`\hat{\mathbf{x}}_{k}`. This operation is written as

        .. math::
            \hat{\mathbf{x}}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t).

        An error is then created as

            e = x_k.minus(x_k_hat),

        where :math:`\mathbf{x}_k` is our current operating point at time :math:`t_k`.
        """
        x_km1 = states[0]
        x_k = states[1]
        dt = x_k.stamp - x_km1.stamp

        # Evaluate the process model, compute the error
        x_k_hat = self._process_model.evaluate(x_km1.copy(), self._u, dt)
        e = x_k.minus(x_k_hat)

        # Scale the error by the square root of the info matrix
        L = self._process_model.sqrt_information(x_km1, self._u, dt)
        e = L.T @ e

        # Compute the Jacobians of the residual w.r.t x_km1 and x_k
        if compute_jacobians:
            jac_list = [None] * len(states)
            if compute_jacobians[0]:
                jac_list[0] = -L.T @ self._process_model.jacobian(
                    x_km1, self._u, dt
                )
            if compute_jacobians[1]:
                jac_list[1] = L.T @ x_k.minus_jacobian(x_k_hat)

            return e, jac_list

        return e


class MeasurementResidual(Residual):
    """
    A generic measurement residual.

    Can be used with any :class:`navlie.Measurement`.
    """

    def __init__(self, keys: List[Hashable], measurement: Measurement):
        super().__init__(keys)
        self._y = measurement

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evaluates the measurement residual.

        The error is computed as

        .. math::
            \mathbf{e} = \mathbf{y} - \mathbf{g} (\mathbf{x}).

        The Jacobian of the residual with respect to the state
        is then the negative of the measurement model Jacobian.
        """
        # Extract state
        x = states[0]

        # Compute predicted measurement
        y_check = self._y.model.evaluate(x)
        e = self._y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))

        # Weight error by square root of information matrix
        L = self._y.model.sqrt_information(x)
        e = L.T @ e

        if compute_jacobians:
            jacobians = [None] * len(states)

            if compute_jacobians[0]:
                jacobians[0] = -L.T @ self._y.model.jacobian(x)
            return e, jacobians

        return e
