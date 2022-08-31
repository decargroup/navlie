from typing import List
from .types import (
    StampedValue,
    State,
    ProcessModel,
    Measurement,
    StateWithCovariance,
)
import numpy as np
from scipy.stats.distributions import chi2


def check_outlier(error: np.ndarray, covariance: np.ndarray):
    """
    Performs the Normalized-Innovation-Squared (NIS) test to identify
    an outlier.
    """
    error = error.reshape((-1, 1))
    md = np.ndarray.item(error.T @ np.linalg.solve(covariance, error))
    if md > chi2.ppf(0.99, df=error.size):
        is_outlier = True
    else:
        is_outlier = False

    return is_outlier


class ExtendedKalmanFilter:
    """
    On-manifold nonlinear Kalman filter.
    """

    __slots__ = ["process_model", "_u", "reject_outliers"]

    def __init__(self, process_model: ProcessModel, reject_outliers=False):
        """
        Parameters
        ----------
        process_model : ProcessModel
            process model to be used in the prediction step
        reject_outliers : bool, optional
            whether to apply the NIS test to measurements, by default False
        """
        self.process_model = process_model
        self._u = None
        self.reject_outliers = reject_outliers

    def predict(
        self,
        x: StateWithCovariance,
        u: StampedValue,
        dt: float =None,
        x_jac: State = None,
        use_last_input=True,
    ) -> StateWithCovariance:
        """
        Propagates the state forward in time using a process model.


        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u : StampedValue
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, the `.stamp` value in 
            `u` will be used.
        x_jac : State, optional
            Evaluation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        use_last_input : bool, optional
            Whether to use the previously-fed input in the process model or the
            current one, by default True. This is essentially the difference
            between 

            .. math:
                x_k = f(x_{k-1}, u_{k-1}) \\text{ and } x_k = f(x_{k-1}, u_{k})
            


        Returns
        -------
        StateWithCovariance
            New predicted state
        """

        # Make a copy so we dont modify the input
        x = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = u.stamp

        if dt is None:
            dt = u.stamp - x.state.stamp

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        # Use current input provided, or the one previously.
        if use_last_input:
            u_eval = self._u
        else:
            u_eval = u

        if u_eval is not None:
            A = self.process_model.jacobian(x_jac, u_eval, dt)
            Q = self.process_model.covariance(x_jac, u_eval, dt)
            x.state = self.process_model.evaluate(x.state, u_eval, dt)
            x.covariance = A @ x.covariance @ A.T + Q
            x.symmetrize()
            x.state.stamp += dt

        self._u = u

        return x

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        x_jac: State = None,
        reject_outlier: bool =None,
    ) -> StateWithCovariance:
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        y : Measurement
            Measurement to be fused into the current state estimate.
        x_jac : State, optional
            valuation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None

        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # Load default outlier rejection option 
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt > 0 and self._u is not None:
                x = self.predict(x, self._u, dt)

        if x_jac is None:
            x_jac = x.state
        P = x.covariance
        R = np.atleast_2d(y.model.covariance(x_jac))
        G = np.atleast_2d(y.model.jacobian(x_jac))
        y_check = y.model.evaluate(x.state)

        if y_check is not None:
            z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
            S = G @ P @ G.T + R

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:

                # Do the correction
                K = np.linalg.solve(S.T, (P @ G.T).T).T
                dx = K @ z
                x.state.plus(dx)
                x.covariance = (np.identity(x.state.dof) - K @ G) @ P
                x.symmetrize()

        return x


class IteratedKalmanFilter(ExtendedKalmanFilter):
    """
    On-manifold iterated extended Kalman filter.
    """
    __slots__ = ["process_model", "_u", "reject_outliers", "step_tol", "max_iters"]

    def __init__(
        self,
        process_model: ProcessModel,
        step_tol=1e-4,
        max_iters=100,  # TODO. implement max iters
        line_search=True,  # TODO implement line search
        reject_outliers=False,
    ):
        super(IteratedKalmanFilter, self).__init__(process_model)
        self.step_tol = step_tol
        self.max_iters = max_iters
        self.reject_outliers = reject_outliers

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        x_jac: State = None,
        reject_outlier=None,
    ):
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        y : Measurement
            Measurement to be fused into the current state estimate.
        x_jac : State, optional
            valuation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None

        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        # Load default outlier rejection option 
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If state has no time stamp, load form measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt > 0 and self._u is not None:
                x = self.predict(x, self._u, dt)

        dx = 10
        x_op = x.state.copy()  # Operating point
        while np.linalg.norm(dx) > self.step_tol:

            # Load a dedicated state evaluation point for jacobian
            # if the user supplied it.
            if x_jac is not None:
                x_op_jac = x_jac
            else:
                x_op_jac = x_op

            R = np.atleast_2d(y.model.covariance(x_op_jac))
            G = np.atleast_2d(y.model.jacobian(x_op_jac))
            y_check = y.model.evaluate(x_op)
            z = y.value.reshape((-1, 1)) - y_check.reshape((-1, 1))
            S = G @ x.covariance @ G.T + R
            S = 0.5 * (S + S.T)
            e = x.state.minus(x_op).reshape((-1, 1))

            # Test for outlier if requested.
            outlier = False
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                K = np.linalg.solve(S.T, (x.covariance @ G.T).T).T
                dx = e + K @ (z - G @ e)
                x_op.plus(0.2 * dx)
            else:
                break

        x.state = x_op
        x.covariance = (np.identity(x.state.dof) - K @ G) @ x.covariance
        x.symmetrize()
        return x

    def _cost(
        self,
        prior_error: np.ndarray,
        prior_covariance: np.ndarray,
        meas_error: np.ndarray,
        meas_covariance: np.ndarray,
    ):
        e = prior_error
        P = prior_covariance
        z = meas_error
        R = meas_covariance
        cost_prior = np.ndarray.item(0.5 * e.T @ np.linalg.solve(P, e))
        cost_meas = np.ndarray.item(0.5 * z.T @ np.linalg.solve(R, z))
        return cost_prior + cost_meas, cost_prior, cost_meas


def run_filter(
    filter: ExtendedKalmanFilter,
    x0: State,
    P0: np.ndarray,
    input_data: List[StampedValue],
    meas_data: List[Measurement],
) -> List[StateWithCovariance]:
    """
    Executes a predict-correct-style filter given lists of input and measurement
    data.

    Parameters
    ----------
    filter : ExtendedKalmanFilter
        _description_
    x0 : State
        _description_
    P0 : np.ndarray
        _description_
    input_data : List[StampedValue]
        _description_
    meas_data : List[Measurement]
        _description_
    """
    x = StateWithCovariance(x0, P0)

    # Sort the data by time
    input_data.sort(key = lambda x: x.stamp)
    meas_data.sort(key = lambda x: x.stamp)

    # Remove all that are before the current time
    for idx, u in enumerate(input_data):
        if u.stamp >= x.state.stamp:
            input_data = input_data[idx:]
            break

    for idx, y in enumerate(meas_data):
        if y.stamp >= x.state.stamp:
            meas_data = meas_data[idx:]
            break


    meas_idx = 0 
    if len(meas_data) > 0:
        y = meas_data[meas_idx]

    results_list = []
    for k in range(len(input_data) - 1):
        u = input_data[k]

        # Fuse any measurements that have occurred.
        if len(meas_data) > 0:
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

                x = filter.correct(x, y)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]

        x = filter.predict(x, u)
        results_list.append(x)
    return results_list