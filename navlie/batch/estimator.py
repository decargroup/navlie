"""Batch estimator to construct batch problems composed of prior residuals, process residuals, and 
measurement residuals.

The BatchEstimator.solve() method constructs and solves a batch problem over a sequence
of input and measurement data. Process and measurement residuals are automatically created
and added to the problem, using the generic definitions of process and measurement residuals
defined in the module `navlie.batch.residuals`.
"""

from typing import List

import numpy as np

from navlie.batch.problem import Problem
from navlie.batch.residuals import (
    MeasurementResidual,
    PriorResidual,
    ProcessResidual,
)
from navlie.types import (
    Input,
    Measurement,
    ProcessModel,
    State,
    StateWithCovariance,
)
from navlie.utils import find_nearest_stamp_idx


class BatchEstimator:
    """Main class for the batch estimator."""

    def __init__(
        self,
        solver_type: str = "GN",
        max_iters: int = 100,
        step_tol: float = 1e-7,
        tau: float = 1e-11,
        verbose: bool = True,
    ):
        """Instantiate :class:`BatchEstiamtor`.

        Parameters
        ----------
        solver : str, optional
            Solver type, either "GN" or "LM", by default "GN".
        max_iters : int, optional
            Maximum number of optimization iterations, by default 100.
        step_tol : float, optional
            Convergence tolerance, by default 1e7.
        tau : float, optional
            tau parameter in LM, by default 1e-11.
        verbose : bool, optional
            Print convergence during runtime, by default True.
        """
        self.solver_type = solver_type
        self.max_iters = max_iters
        self.step_tol = step_tol
        self.tau = tau
        self.verbose = verbose

    def solve(
        self,
        x0: State,
        P0: np.ndarray,
        input_data: List[Input],
        meas_data: List[Measurement],
        process_model: ProcessModel,
        return_opt_results: bool = False,
    ) -> List[StateWithCovariance]:
        """Creates and solves a batch problem.

        The input data is used to propagate the initial state x0 forward in time
        using the process model, to generate an initial estimate of the state
        at estimate timestep.

        The batch problem created involves a :class:`PriorResidual`, a
        :class:`ProcessResidual` for each input used to connect subsequent
        states through the process model, and a :class:`MeasurementResiduals`
        for each measurement.

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
        input_stamps = [round(u.stamp, 12) for u in input_data]
        meas_stamps = [round(meas.stamp, 12) for meas in meas_data]
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
            max_iters=self.max_iters,
            solver=self.solver_type,
            step_tol=self.step_tol,
            tau=self.tau,
            verbose=self.verbose,
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
