"""Batch estimation related functionality, using pysquares.

A set of commonly-used residuals in batch estimation are defined here,
that all inherit from the `Residual` type defined in pysquares. These residuals
are :
    - the PriorResidual,
    - the ProcessResidual, which uses a pynav `ProcessModel` to compute an error between
    a predicted state and the actual state,
    - a MeasurementResidual, which uses a pynav `Measurement` to compare 
    a true measurement to the measurement predicted by the `MeasurementModel`.

The run_batch() function can also be used to construct a batch problem given an initial estimate 
(x0, P0), a list of input data and a corresponding process model, and a list of measurements.
"""

from dataclasses import dataclass
from typing import Hashable, List, Tuple

import numpy as np

from pynav.types import (
    Input,
    Measurement,
    ProcessModel,
    StampedValue,
    State,
    StateWithCovariance,
)
from pynav.utils import find_nearest_stamp_idx
from pysquares.problem import Problem
from pysquares.types import Residual


class PriorResidual(Residual):
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
        """A prior error in the form of 
        
            e = x.minus(x0),

        where x is our operating point and x0 is a prior guess.
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
        """A generic process residual.

        The input self._u is used to propagate the state
        at time x_km1 forward in time to genrate x_k_hat.
        An error is created as 

            e = x_k.minus(x_k_hat),
        where x_k is our current operating point at time k.
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
    def __init__(self, keys: List[Hashable], measurement: Measurement):
        super().__init__(keys)
        self._y = measurement

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """A generic residual for a measurement.

        Computes a residual by comparing the true measurement
        to the predicted measurement given by the model.

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


@dataclass
class OptimizationSettings:
    """A dataclass to hold some default optimizaiton settings."""
    solver: str = "GN" # Solver type, either "GN" or "LM"
    max_iters: int = 100 # maximum number of iters
    step_tol: float = 1e-7 # Convergence tolerance, norm of the step size.
    tau: float = 1e-11 # tau parameter in LM
    verbose: bool = True # Print convergence during runtime 


def run_batch(
    x0: State,
    P0: np.ndarray,
    input_data: List[Input],
    meas_data: List[Measurement],
    process_model: ProcessModel,
    return_opt_results: bool = False,
    opt_settings: OptimizationSettings = OptimizationSettings(),
) -> List[StateWithCovariance]:
    """Creates and solves a batch problem using pysquares.

    The input data is used to propagate the initial state x0 forward in time
    using the process model, to generate an initial estimate of the state
    at estimate timestep.

    The batch problem created involves a PriorResidual, a ProcessResidual 
    for each input used to connect subsequent states through the process model,
    and MeasurementResiduals for each measurement. 

    Parameters
    ----------
    x0 : State
        x0: Initial state.
    P0 : np.ndarray
        Initial covariance
    input_data : List[Input]
        List of input data.
    meas_data : List[Measurement]
        List of measurements.
    process_model : ProcessModel
        Process model used to propagate the initial estimate
        and form ProcessResiduals.
    return_opt_results : bool, optional
        Flag to optionally return the results dictionary
        from the batch problem, by default False
    opt_settings : OptimizationSettings, optional
        Settings used by the optimizer, by default OptimizationSettings()

    Returns
    -------
    List[StateWithCovariance]
        List of estimates with covariance.
    """

    # Sort the data by time
    input_data.sort(key=lambda x: x.stamp)
    meas_data.sort(key=lambda x: x.stamp)

    # Remove all that are before the current time
    for idx, u in enumerate(input_data):
        if u.stamp >= x0.stamp:
            input_data = input_data[idx:]
            break

    for idx, y in enumerate(meas_data):
        if y.stamp >= x0.stamp:
            meas_data = meas_data[idx:]
            break

    # We want to generate state estimates at
    # each input and measurement timestamp
    input_stamps = [round(u.stamp, 5) for u in input_data]
    meas_stamps = [round(meas.stamp, 5) for meas in meas_data]
    stamps = input_stamps + meas_stamps

    # Get unique stamps
    stamps = list(np.unique(np.array(stamps)))

    # Propagate states through process model to create
    # initial estimate
    state_list: List[State] = [None] * len(stamps)
    state_list[0] = x0.copy()
    input_idx = 0
    x = x0.copy()
    for k in range(len(stamps) - 1):
        u = input_data[input_idx]

        dt = stamps[k + 1] - x.stamp
        if dt < 0:
            raise RuntimeError("dt is negative!")

        x = state_list[k].copy()
        x = process_model.evaluate(x, u, dt)
        x.stamp = x.stamp + dt
        state_list[k + 1] = x

        if stamps[k + 1] < input_stamps[input_idx + 1]:
            continue
        else:
            input_idx += 1

    # Create problem and add all variables to the problem.
    problem = Problem(
        max_iters=opt_settings.max_iters,
        solver=opt_settings.solver,
        step_tol=opt_settings.step_tol,
        tau=opt_settings.tau,
        verbose=opt_settings.verbose,
    )

    # The key used is just the index in the state list
    for i, state in enumerate(state_list):
        problem.add_variable(i, state)

    # Add prior residual
    prior_residual = PriorResidual(0, x0, P0)
    problem.add_residual(prior_residual)

    # Add process residuals
    for k in range(len(input_data) - 1):
        # Get current input and next input to compute 
        u_k = input_data[k]
        u_kp1 = input_data[k + 1]

        # Find the states that are connected by this measurement
        keys = [None] * 2
        keys[0] = find_nearest_stamp_idx(stamps, u_k.stamp)
        keys[1] = find_nearest_stamp_idx(stamps, u_kp1.stamp)
        process_residual = ProcessResidual(keys, process_model, u_k)
        problem.add_residual(process_residual)

    # Add measurement residuals
    for meas in meas_data:
        state_idx = find_nearest_stamp_idx(stamps, meas.stamp)
        meas_residual = MeasurementResidual(state_idx, meas)
        problem.add_residual(meas_residual)

    # Solve problem
    opt_results = problem.solve()

    # Extract state and covariance
    results_list = []
    for key, state in opt_results["variables"].items():
        cov = problem.get_covariance_block(key, key)
        results_list.append(StateWithCovariance(state, cov))

    if return_opt_results:
        return results_list, opt_results
    else:
        return results_list
